import time
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import or_
from database import get_db
import models
from services.ingestion import extract_file_text, get_chunks_from_text
from services.embedding import get_embedding, get_embeddings_batch
from services.scraper import scrape_url_content
from services.auth import get_current_user

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge Ingestion"])

@router.get("/sources/{bot_id}")
def get_sources(bot_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Security: Ensure bot belongs to the user's organization
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    sources = db.query(models.BotSource).filter(
        models.BotSource.botId == bot_id,
        models.BotSource.isUniversal == False
    ).order_by(models.BotSource.createdAt.desc()).all()
    bot_tokens = sum(s.tokenCount or 0 for s in sources)

    # Universal sources inherited by all bots in this organization
    universal_sources = db.query(models.BotSource).filter(
        models.BotSource.isUniversal == True,
        or_(models.BotSource.orgId == user.orgId, models.BotSource.orgId == None)
    ).order_by(models.BotSource.createdAt.desc()).all()
    universal_tokens = sum(s.tokenCount or 0 for s in universal_sources)

    return {
        "sources": [{"id": s.id, "kind": s.kind, "title": s.title, "tokens": s.tokenCount or 0, "url": s.url or ""} for s in sources],
        "universal_sources": [{"id": s.id, "kind": s.kind, "title": s.title, "tokens": s.tokenCount or 0, "url": s.url or ""} for s in universal_sources],
        "bot_tokens": bot_tokens,
        "universal_tokens": universal_tokens,
        "used_tokens": bot_tokens + universal_tokens,
        "max_tokens": 60000,
    }

# ----------------- UNIVERSAL KNOWLEDGE BASE ENDPOINTS ----------------- #

@router.get("/universal")
def get_universal_sources(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Fetches all Organization Universal Knowledge sources inherited by all bots"""
    sources = db.query(models.BotSource).filter(
        models.BotSource.isUniversal == True,
        or_(models.BotSource.orgId == user.orgId, models.BotSource.orgId == None)
    ).order_by(models.BotSource.createdAt.desc()).all()

    total_tokens = sum(s.tokenCount or 0 for s in sources)
    bot_count = db.query(models.Bot).filter(models.Bot.orgId == user.orgId).count()

    source_ids = [s.id for s in sources]
    total_chunks = db.query(models.DocumentChunk).filter(models.DocumentChunk.sourceId.in_(source_ids)).count() if source_ids else 0

    return {
        "sources": [{
            "id": s.id,
            "kind": s.kind,
            "title": s.title,
            "tokens": s.tokenCount or 0,
            "url": s.url or "",
            "createdAt": s.createdAt.strftime("%b %d, %Y") if s.createdAt else ""
        } for s in sources],
        "total_tokens": total_tokens,
        "total_chunks": total_chunks,
        "active_bots_count": bot_count
    }

@router.delete("/universal/{source_id}")
def delete_universal_source(source_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    source = db.query(models.BotSource).filter(
        models.BotSource.id == source_id,
        models.BotSource.isUniversal == True,
        or_(models.BotSource.orgId == user.orgId, models.BotSource.orgId == None)
    ).first()
    if not source:
        raise HTTPException(status_code=404, detail="Universal source not found")

    db.delete(source)
    db.commit()
    return {"ok": True}

@router.post("/universal/ingest-mixed")
async def ingest_universal_mixed(
    pasted_text: str = Form(""),
    title: str = Form(""),
    file: UploadFile = File(None),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB Enterprise Bulk Limit
    if file and file.filename:
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=400, detail="File size exceeds maximum allowed limit of 500 MB.")
        text_content = extract_file_text(file_bytes, file.filename)
        source_title = file.filename
        kind = "FILE"
    elif pasted_text.strip():
        text_content = pasted_text.strip()
        source_title = title.strip() or "Universal Grounding Policy"
        kind = "TEXT"
    else:
        raise HTTPException(status_code=400, detail="Provide a file or text content.")

    return process_and_save_source(
        db=db,
        bot_id=None,
        title=source_title,
        content=text_content,
        kind=kind,
        is_universal=True,
        org_id=user.orgId
    )

@router.post("/universal/scrape")
def scrape_universal_site(
    url: str = Form(...),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        title, text_content, _ = scrape_url_content(url)
        return process_and_save_source(
            db=db,
            bot_id=None,
            title=title,
            content=text_content,
            kind="PAGE",
            original_url=url,
            is_universal=True,
            org_id=user.orgId
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ----------------- BOT-SPECIFIC INGESTION ENDPOINTS ----------------- #

@router.delete("/sources/{source_id}")
def delete_source(source_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    source = db.query(models.BotSource).filter(models.BotSource.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    # Security: Check if user owns the bot connected to this source
    if source.botId:
        bot = db.query(models.Bot).filter(models.Bot.id == source.botId, models.Bot.orgId == user.orgId).first()
        if not bot:
            raise HTTPException(status_code=403, detail="Not authorized to delete this source")
    elif source.orgId != user.orgId:
        raise HTTPException(status_code=403, detail="Not authorized to delete this source")

    db.delete(source)
    db.commit()
    return {"ok": True}

@router.post("/ingest-mixed")
async def ingest_mixed(
    bot_id: str = Form(...),
    pasted_text: str = Form(""),
    title: str = Form(""),
    gap_id: str = Form(None),
    gap_question: str = Form(None),
    file: UploadFile = File(None),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB Enterprise Bulk Limit
    if file and file.filename:
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=400, detail="File size exceeds maximum allowed limit of 500 MB.")
        text_content = extract_file_text(file_bytes, file.filename)
        source_title = file.filename
        kind = "FILE"
    elif pasted_text.strip():
        text_content = pasted_text.strip()
        source_title = title.strip() or "Hand-written Note"
        kind = "TEXT"
    else:
        raise HTTPException(status_code=400, detail="Provide a file or text content.")

    res = process_and_save_source(
        db=db,
        bot_id=bot.id,
        title=source_title,
        content=text_content,
        kind=kind,
        is_universal=False,
        org_id=user.orgId
    )

    # Automatically resolve content gap if this grounding fact answered one
    if gap_id or gap_question:
        try:
            conv_ids = [c.id for c in db.query(models.Conversation).filter(models.Conversation.botId == bot.id).all()]
            if conv_ids:
                if gap_id:
                    msgs = db.query(models.Message).filter(models.Message.id == gap_id).all()
                    for m in msgs:
                        m.unanswered = False
                if gap_question:
                    clean_gq = gap_question.strip('?.,! ')
                    msgs = db.query(models.Message).filter(
                        models.Message.conversationId.in_(conv_ids),
                        models.Message.content.ilike(f"%{clean_gq}%")
                    ).all()
                    for m in msgs:
                        m.unanswered = False
                db.commit()

            if gap_question:
                clean_gq = gap_question.strip('?.,! ')
                leads = db.query(models.Lead).filter(
                    models.Lead.botId == bot.id,
                    models.Lead.note.ilike(f"%{clean_gq}%")
                ).all()
                for ld in leads:
                    if not (ld.note or '').startswith("[RESOLVED]"):
                        ld.note = f"[RESOLVED] {ld.note}"
                db.commit()
            print(f"✅ [Content Gap Resolved]: gap_id={gap_id}, question='{gap_question}'")
        except Exception as g_err:
            print(f"⚠️ [Gap Resolution Error]: {g_err}")

    return res

@router.post("/scrape")
def scrape_site(
    bot_id: str = Form(...),
    url: str = Form(...),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    try:
        # Scrape data via BeautifulSoup/Wikipedia API
        title, text_content, logo_url = scrape_url_content(url)

        # Save fetched logo automatically if bot doesn't have one yet
        if logo_url and hasattr(bot, "logoUrl") and not bot.logoUrl:
            bot.logoUrl = logo_url
            db.commit()

        return process_and_save_source(
            db=db,
            bot_id=bot.id,
            title=title,
            content=text_content,
            kind="PAGE",
            original_url=url,
            is_universal=False,
            org_id=user.orgId
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def process_and_save_source(
    db: Session,
    bot_id: str | None,
    title: str,
    content: str,
    kind: str,
    original_url: str = "",
    is_universal: bool = False,
    org_id: str | None = None
):
    """Helper function to save source, create chunks, and generate PostgreSQL vectors with telemetry metrics"""
    start_time = time.perf_counter()

    # Sanitize NUL (0x00) characters for PostgreSQL safety
    title = title.replace('\x00', '').replace('\0', '').strip()
    content = content.replace('\x00', '').replace('\0', '').strip()
    if original_url:
        original_url = original_url.replace('\x00', '').replace('\0', '').strip()

    if not content:
        raise HTTPException(status_code=400, detail="No readable text extracted.")

    # Deduplication: If this URL or title was already indexed, delete the old version cleanly
    if is_universal and org_id:
        filter_cond = [models.BotSource.isUniversal == True, models.BotSource.orgId == org_id]
        if original_url:
            filter_cond.append((models.BotSource.url == original_url) | (models.BotSource.title == title))
        elif title:
            filter_cond.append(models.BotSource.title == title)
        old_sources = db.query(models.BotSource).filter(*filter_cond).all()
        for old_s in old_sources:
            db.delete(old_s)
        db.commit()
    elif bot_id:
        filter_cond = [models.BotSource.botId == bot_id]
        if original_url:
            filter_cond.append((models.BotSource.url == original_url) | (models.BotSource.title == title))
        elif title:
            filter_cond.append(models.BotSource.title == title)
        old_sources = db.query(models.BotSource).filter(*filter_cond).all()
        for old_s in old_sources:
            db.delete(old_s)
        db.commit()

    # 1. Save Parent Source
    token_count = len(content.split())
    source = models.BotSource(
        botId=bot_id,
        isUniversal=is_universal,
        orgId=org_id,
        kind=kind,
        title=title,
        url=original_url or None,
        tokenCount=token_count
    )
    db.add(source)
    db.commit()
    db.refresh(source)

    # 2. Chunking & Embeddings (Scaled with Batch Vectorization)
    chunks = get_chunks_from_text(content)
    prepared_chunks = []
    for text_chunk in chunks:
        clean_chunk = text_chunk.replace('\x00', '').replace('\0', '').strip()
        if not clean_chunk:
            continue
        context_title = f"{title} ({original_url})" if original_url else title
        chunk_with_title = f"{context_title}\n{clean_chunk}".replace('\x00', '').replace('\0', '')
        prepared_chunks.append(chunk_with_title)

    if prepared_chunks:
        # Vectorize all chunks rapidly in batches
        vectors = get_embeddings_batch(prepared_chunks)
        for chunk_text_content, vec in zip(prepared_chunks, vectors):
            db.add(models.DocumentChunk(sourceId=source.id, content=chunk_text_content, embedding=vec))
        db.commit()

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "status": "success",
        "title": title,
        "chunks_created": len(prepared_chunks),
        "tokens": token_count,
        "kind": kind,
        "is_universal": is_universal,
        "embedding_time_ms": round(elapsed_ms, 1)
    }
