import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import or_, text
from app.db.database import get_db, SessionLocal
from app.db import models
from app.services.ingestion import extract_file_text, get_chunks_from_text
from app.services.embedding import get_embedding, get_embeddings_batch
from app.services.scraper import scrape_url_content
from app.core.dependencies import get_current_user
from app.services.quality import analyze_knowledge_quality
from app.services.audit import log_audit_event
from app.services.bulk_ingestion import manage_vector_index, bulk_insert_chunks, parallel_embed_and_save
from app.services.multimodal import process_image_multimodal, process_video_multimodal
from app.services.hybrid_search import three_layer_hybrid_search

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge Ingestion & Quality"])

@router.get("/quality/report", summary="Retrieve real-time Knowledge Base Health & Quality Audit Report")
def get_knowledge_quality_report(
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Returns automated quality findings: stale documents (>180d), duplicate chunks,
    conflicting engineering specifications, and 0-100 overall health score.
    """
    return analyze_knowledge_quality(
        db=db,
        org_id=user.orgId,
        current_user=user,
        client_ip="internal"
    )

@router.post("/quality/scan", summary="Trigger an on-demand Knowledge Quality re-scan and audit log")
def trigger_knowledge_quality_scan(
    request: Request,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Executes a full re-scan and records audit log event."""
    client_ip = request.client.host if request.client else "unknown"
    report = analyze_knowledge_quality(
        db=db,
        org_id=user.orgId,
        current_user=user,
        client_ip=client_ip
    )
    log_audit_event(
        db=db,
        org_id=user.orgId,
        action="KNOWLEDGE_QUALITY_SCAN_RUN",
        resource_type="knowledge_quality",
        resource_id=user.orgId,
        user=user,
        details={
            "health_score": report["health_score"],
            "stale_docs": report["metrics"]["stale_documents_count"],
            "conflicting_specs": report["metrics"]["conflicting_specs_count"],
            "duplicates": report["metrics"]["duplicate_clusters_count"]
        },
        ip_address=client_ip
    )
    return report


def resolve_source_display_title(s: models.BotSource, db: Session) -> str:
    title = (s.title or "").strip()
    is_generic = not title or any(title.lower().startswith(p) for p in ['test.', 'upload_', 'img_', 'image.', 'scan_', 'file.', 'doc.'])
    if is_generic:
        fc = db.query(models.DocumentChunk).filter(models.DocumentChunk.sourceId == s.id).first()
        if fc and fc.content:
            import re
            t_match = re.search(r'(?:poster for\s*["\']([^"\']+)["\']|Title[:\s*]+["\']?([^"\']+)["\']?)', fc.content, re.IGNORECASE)
            if t_match:
                extracted = (t_match.group(1) or t_match.group(2)).strip()
                if extracted and len(extracted) > 3:
                    return extracted
            for line in fc.content.splitlines():
                l_s = line.strip()
                if l_s and not l_s.startswith('![') and not l_s.startswith('[') and not l_s.lower().endswith(('.jpeg', '.jpg', '.png', '.pdf')):
                    if len(l_s) > 4 and len(l_s) < 60:
                        return l_s
    import re
    return re.sub(r'\.(pdf|docx?|txt|json|zip|csv|xlsx?|vsixpackage|html?|jpeg|jpg|png|webp)$', '', title, flags=re.IGNORECASE).strip() or title

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
        "sources": [{"id": s.id, "kind": s.kind, "title": s.title, "display_title": resolve_source_display_title(s, db), "tokens": s.tokenCount or 0, "url": s.url or ""} for s in sources],
        "universal_sources": [{"id": s.id, "kind": s.kind, "title": s.title, "display_title": resolve_source_display_title(s, db), "tokens": s.tokenCount or 0, "url": s.url or ""} for s in universal_sources],
        "bot_tokens": bot_tokens,
        "universal_tokens": universal_tokens,
        "used_tokens": bot_tokens + universal_tokens,
        "max_tokens": 1000000,
    }

@router.get("/sources/{source_id}/view", summary="Inspect source details, extracted text and vector chunks")
def view_source_content(
    source_id: str,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Fetches complete document metadata and all generated text chunks for inspection."""
    source = db.query(models.BotSource).filter(models.BotSource.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Document source not found.")

    if not source.isUniversal and source.orgId and source.orgId != user.orgId:
        raise HTTPException(status_code=403, detail="Access denied to this source.")

    chunks = db.query(models.DocumentChunk).filter(
        models.DocumentChunk.sourceId == source.id
    ).order_by(models.DocumentChunk.id.asc()).all()

    return {
        "id": source.id,
        "title": source.title,
        "display_title": resolve_source_display_title(source, db),
        "kind": source.kind,
        "url": source.url,
        "isUniversal": source.isUniversal,
        "tokenCount": source.tokenCount or sum(len(c.content.split()) for c in chunks),
        "createdAt": source.createdAt.strftime("%b %d, %Y %H:%M") if source.createdAt else None,
        "chunks_count": len(chunks),
        "chunks": [
            {
                "id": c.id,
                "content": c.content,
                "language": c.language or "en",
                "document_type": c.document_type or source.kind or "DOC",
                "status": c.status.value if hasattr(c.status, "value") else str(c.status)
            }
            for c in chunks
        ]
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
            "display_title": resolve_source_display_title(s, db),
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

def async_embed_and_index_chunks(chunk_items: List[Dict[str, Any]], drop_recreate_index: bool = False):
    """Background task to generate dense embeddings and update PostgreSQL vector records"""
    if not chunk_items:
        return
    db = SessionLocal()
    try:
        if drop_recreate_index and len(chunk_items) > 500:
            manage_vector_index(action="drop", db=db)

        all_texts = [it["content"] for it in chunk_items]
        all_vectors = get_embeddings_batch(all_texts, batch_size=128)

        # Batch update embeddings in PostgreSQL
        batch_size = 500
        for i in range(0, len(chunk_items), batch_size):
            batch_slice = chunk_items[i : i + batch_size]
            batch_vecs = all_vectors[i : i + batch_size]
            param_dicts = []
            for it, vec in zip(batch_slice, batch_vecs):
                vec_str = "[" + ",".join(str(x) for x in vec) + "]"
                param_dicts.append({
                    "id": it["id"],
                    "embedding": vec_str
                })

            stmt = text("""
                UPDATE document_chunks
                SET embedding = CAST(:embedding AS vector)
                WHERE id = :id
            """)
            db.execute(stmt, param_dicts)
            db.commit()

        if drop_recreate_index and len(chunk_items) > 500:
            manage_vector_index(action="recreate", db=db)

        print(f"✅ [Background Ingestion]: Successfully vectorized and indexed {len(chunk_items)} chunks.")
    except Exception as e:
        db.rollback()
        print(f"❌ [Background Ingestion Error]: {e}")
    finally:
        db.close()

def _parse_file_worker(item):
    filename, file_bytes = item
    try:
        ext = (filename.split('.')[-1] or '').lower()
        is_image = ext in ["png", "jpg", "jpeg", "webp"]
        is_video = ext in ["mp4", "mov", "avi", "mkv"]

        if is_image:
            img_res = process_image_multimodal(file_bytes, filename)
            source_title = img_res.get("title", filename)
            tok_cnt = len(img_res["chunk_content"].split())
            return {
                "filename": filename,
                "kind": "DIAGRAM",
                "title": source_title,
                "url": img_res["image_url"],
                "token_count": tok_cnt,
                "chunks": [{
                    "id": str(uuid.uuid4()),
                    "content": img_res["chunk_content"],
                    "document_type": "DIAGRAM",
                    "status": models.DocumentStatus.APPROVED,
                    "source_url": img_res["image_url"],
                    "language": "en"
                }]
            }

        elif is_video:
            video_chunks = process_video_multimodal(video_bytes=file_bytes, filename=filename, sample_interval_seconds=5)
            source_title = f"Video Guide: {filename}"
            tok_cnt = sum(len(c["content"].split()) for c in video_chunks)
            full_vid_url = video_chunks[0]["video_url"] if video_chunks else None
            return {
                "filename": filename,
                "kind": "VIDEO",
                "title": source_title,
                "url": full_vid_url,
                "token_count": tok_cnt,
                "chunks": [{
                    "id": str(uuid.uuid4()),
                    "content": vc["content"],
                    "document_type": "VIDEO",
                    "status": models.DocumentStatus.APPROVED,
                    "source_url": vc["frame_url"],
                    "language": "en"
                } for vc in video_chunks]
            }

        else:
            text_content = extract_file_text(file_bytes, filename)
            if not text_content or not text_content.strip():
                print(f"⚠️ [Batch Ingestion Warning]: No text extracted from {filename}")
                return None

            tok_cnt = len(text_content.split())
            source_title = filename
            chunks = get_chunks_from_text(text_content)
            chunk_items = []
            for text_chunk in chunks:
                clean_chunk = text_chunk.replace('\x00', '').replace('\0', '').strip()
                if not clean_chunk:
                    continue
                chunk_with_title = f"{source_title}\n{clean_chunk}"
                chunk_items.append({
                    "id": str(uuid.uuid4()),
                    "content": chunk_with_title,
                    "document_type": "FILE",
                    "status": models.DocumentStatus.APPROVED,
                    "source_url": None,
                    "language": "en"
                })

            return {
                "filename": filename,
                "kind": "FILE",
                "title": source_title,
                "url": None,
                "token_count": tok_cnt,
                "chunks": chunk_items
            }
    except Exception as file_err:
        print(f"❌ [Batch Ingestion Worker Error] '{filename}': {file_err}")
        return None

async def ingest_files_batch_core(
    files: List[UploadFile],
    bot_id: Optional[str],
    is_universal: bool,
    org_id: str,
    db: Session,
    background_tasks: Optional[BackgroundTasks] = None
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB Enterprise Bulk Limit

    valid_files = [f for f in files if f and f.filename]
    if not valid_files:
        raise HTTPException(status_code=400, detail="No files provided for batch ingestion.")

    # 1. Asynchronously read all file bytes upfront
    raw_files_data = []
    for file in valid_files:
        try:
            file_bytes = await file.read()
            if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' exceeds 500 MB limit.")
            raw_files_data.append((file.filename, file_bytes))
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ [Batch Ingestion Read Error] '{file.filename}': {e}")

    if not raw_files_data:
        raise HTTPException(status_code=400, detail="Could not read uploaded files.")

    # 2. Parallel multi-worker document extraction and semantic chunking
    num_workers = min(len(raw_files_data), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        parsed_results = list(executor.map(_parse_file_worker, raw_files_data))

    processed_files = []
    items_to_embed = []
    total_tokens = 0

    # 3. Single Batched Database Transaction for Sources
    for parsed in parsed_results:
        if not parsed or not parsed["chunks"]:
            continue

        source_title = parsed["title"]
        # Deduplication within transaction
        if is_universal:
            db.query(models.BotSource).filter(
                models.BotSource.isUniversal == True,
                models.BotSource.orgId == org_id,
                models.BotSource.title == source_title
            ).delete(synchronize_session=False)
        else:
            db.query(models.BotSource).filter(
                models.BotSource.botId == bot_id,
                models.BotSource.title == source_title
            ).delete(synchronize_session=False)

        source = models.BotSource(
            botId=bot_id if not is_universal else None,
            isUniversal=is_universal,
            orgId=org_id,
            kind=parsed["kind"],
            title=source_title,
            url=parsed.get("url"),
            tokenCount=parsed["token_count"]
        )
        db.add(source)
        db.flush()  # Allocates source.id without full commit roundtrip

        total_tokens += parsed["token_count"]
        processed_files.append({
            "filename": parsed["filename"],
            "kind": parsed["kind"],
            "chunks_created": len(parsed["chunks"]),
            "tokens": parsed["token_count"],
            "source_id": source.id
        })

        for chunk_dict in parsed["chunks"]:
            if "id" not in chunk_dict or not chunk_dict["id"]:
                chunk_dict["id"] = str(uuid.uuid4())
            chunk_dict["sourceId"] = source.id
            chunk_dict["embedding"] = None
            items_to_embed.append(chunk_dict)

    db.commit()

    if not processed_files:
        raise HTTPException(status_code=400, detail="No readable text or valid media extracted from provided files.")

    # 4. Immediate Foreground Chunk Commit & Asynchronous Background Vectorization
    if items_to_embed:
        bulk_insert_chunks(items_to_embed, batch_size=500, db=db)
        bg_payload = [{"id": it["id"], "content": it["content"]} for it in items_to_embed]
        if background_tasks:
            background_tasks.add_task(async_embed_and_index_chunks, bg_payload, drop_recreate_index=(len(bg_payload) > 500))
        else:
            async_embed_and_index_chunks(bg_payload, drop_recreate_index=(len(bg_payload) > 500))

    total_time = time.perf_counter() - start_time

    return {
        "status": "success",
        "total_files": len(processed_files),
        "files_processed": processed_files,
        "total_chunks": len(items_to_embed),
        "chunks_created": len(items_to_embed),
        "total_tokens": total_tokens,
        "tokens": total_tokens,
        "is_universal": is_universal,
        "background_processing": True,
        "total_time_seconds": round(total_time, 2)
    }

@router.post("/universal/ingest-batch", summary="Batch multi-file ingestion for Universal Knowledge")
async def ingest_universal_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return await ingest_files_batch_core(
        files=files,
        bot_id=None,
        is_universal=True,
        org_id=user.orgId,
        db=db,
        background_tasks=background_tasks
    )

@router.post("/universal/ingest-mixed")
async def ingest_universal_mixed(
    background_tasks: BackgroundTasks,
    pasted_text: str = Form(""),
    title: str = Form(""),
    file: UploadFile = File(None),
    files: List[UploadFile] = File(None),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    all_upload_files = []
    if files:
        all_upload_files.extend([f for f in files if f and f.filename])
    if file and file.filename and file not in all_upload_files:
        all_upload_files.append(file)

    if all_upload_files:
        return await ingest_files_batch_core(
            files=all_upload_files,
            bot_id=None,
            is_universal=True,
            org_id=user.orgId,
            db=db,
            background_tasks=background_tasks
        )

    if pasted_text.strip():
        text_content = pasted_text.strip()
        source_title = title.strip() or "Universal Grounding Policy"
        kind = "TEXT"
        return process_and_save_source(
            db=db,
            bot_id=None,
            title=source_title,
            content=text_content,
            kind=kind,
            is_universal=True,
            org_id=user.orgId,
            background_tasks=background_tasks
        )

    raise HTTPException(status_code=400, detail="Provide at least one file or text content.")

@router.post("/universal/scrape")
def scrape_universal_site(
    background_tasks: BackgroundTasks,
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
            org_id=user.orgId,
            background_tasks=background_tasks
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

@router.post("/ingest-batch", summary="Batch multi-file ingestion for Bot-specific Knowledge")
async def ingest_bot_batch(
    background_tasks: BackgroundTasks,
    bot_id: str = Form(...),
    files: List[UploadFile] = File(...),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    return await ingest_files_batch_core(
        files=files,
        bot_id=bot.id,
        is_universal=False,
        org_id=user.orgId,
        db=db,
        background_tasks=background_tasks
    )

@router.post("/ingest-mixed")
async def ingest_mixed(
    background_tasks: BackgroundTasks,
    bot_id: str = Form(...),
    pasted_text: str = Form(""),
    title: str = Form(""),
    gap_id: str = Form(None),
    gap_question: str = Form(None),
    file: UploadFile = File(None),
    files: List[UploadFile] = File(None),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    all_upload_files = []
    if files:
        all_upload_files.extend([f for f in files if f and f.filename])
    if file and file.filename and file not in all_upload_files:
        all_upload_files.append(file)

    if all_upload_files:
        res = await ingest_files_batch_core(
            files=all_upload_files,
            bot_id=bot.id,
            is_universal=False,
            org_id=user.orgId,
            db=db,
            background_tasks=background_tasks
        )
    elif pasted_text.strip():
        text_content = pasted_text.strip()
        source_title = title.strip() or "Hand-written Note"
        kind = "TEXT"
        res = process_and_save_source(
            db=db,
            bot_id=bot.id,
            title=source_title,
            content=text_content,
            kind=kind,
            is_universal=False,
            org_id=user.orgId,
            background_tasks=background_tasks
        )
    else:
        raise HTTPException(status_code=400, detail="Provide at least one file or text content.")

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
    background_tasks: BackgroundTasks,
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
            org_id=user.orgId,
            background_tasks=background_tasks
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
    org_id: str | None = None,
    background_tasks: Optional[BackgroundTasks] = None
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

    # 2. Chunking & Embeddings (Immediate Foreground Chunks & Background Vectorization)
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
        chunk_items = [{
            "id": str(uuid.uuid4()),
            "sourceId": source.id,
            "content": chunk_text_content,
            "embedding": None,
            "source_url": source.url,
            "document_type": source.kind or "DOC",
            "status": models.DocumentStatus.APPROVED,
            "language": "en"
        } for chunk_text_content in prepared_chunks]
        bulk_insert_chunks(chunk_items, batch_size=500, db=db)

        bg_payload = [{"id": it["id"], "content": it["content"]} for it in chunk_items]
        if background_tasks:
            background_tasks.add_task(async_embed_and_index_chunks, bg_payload, drop_recreate_index=(len(bg_payload) > 500))
        else:
            async_embed_and_index_chunks(bg_payload, drop_recreate_index=(len(bg_payload) > 500))

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "status": "success",
        "title": title,
        "chunks_created": len(prepared_chunks),
        "total_chunks": len(prepared_chunks),
        "tokens": token_count,
        "total_tokens": token_count,
        "kind": kind,
        "is_universal": is_universal,
        "background_processing": True,
        "total_time_seconds": round(time.perf_counter() - start_time, 2),
        "embedding_time_ms": round(elapsed_ms, 1)
    }

# =============================================================================
# BULK INGESTION PIPELINE & VECTOR INDEX LIFECYCLE (THE GOLDEN TRICK)
# =============================================================================

@router.get("/index/status", summary="Inspect pgvector HNSW index status")
def get_vector_index_status(
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Checks whether the HNSW index on document_chunks is active and query-ready."""
    return manage_vector_index(action="status", db=db)

@router.post("/index/lifecycle", summary="Trigger vector index drop or rebuild")
def trigger_vector_index_lifecycle(
    action: str = Form(..., description="'drop' to disable index before massive upload, or 'recreate' post-load"),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Executes The Golden Trick:
    - 'drop': Drops HNSW index to prevent 50x slowdown during GB/TB data uploads.
    - 'recreate': Rebuilds the index with optimized parameters (m=16, ef_construction=64).
    """
    if action not in ["drop", "recreate", "status"]:
        raise HTTPException(status_code=400, detail="Action must be 'drop', 'recreate', or 'status'")
    return manage_vector_index(action=action, db=db)

@router.post("/bulk-ingest", summary="High-performance async bulk ingestion for large documents and datasets")
async def bulk_ingest_dataset(
    bot_id: str = Form(None),
    is_universal: bool = Form(False),
    title: str = Form(""),
    file: UploadFile = File(None),
    pasted_text: str = Form(""),
    auto_manage_index: bool = Form(True),
    max_workers: int = Form(8),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    High-Throughput Bulk Ingestion:
    - Drops HNSW index if large dataset (The Golden Trick).
    - Multi-threaded parallel embedding generation.
    - Parameterized multi-row batch inserts (500 chunks/batch).
    - Automatically rebuilds HNSW index upon completion.
    """
    if not is_universal and not bot_id:
        raise HTTPException(status_code=400, detail="Must provide bot_id or set is_universal=True.")

    bot = None
    if bot_id:
        bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")

    content_text = ""
    source_title = title.strip()

    if file:
        file_bytes = await file.read()
        source_title = source_title or file.filename
        content_text = extract_file_text(file_bytes, file.filename)
    elif pasted_text.strip():
        content_text = pasted_text.strip()
        source_title = source_title or "Bulk Text Document"
    else:
        raise HTTPException(status_code=400, detail="Provide a file or text content.")

    if not content_text:
        raise HTTPException(status_code=400, detail="No readable text extracted from input.")

    # 1. Create Source record
    source = models.BotSource(
        botId=bot.id if bot else None,
        isUniversal=is_universal,
        orgId=user.orgId,
        kind="FILE" if file else "TEXT",
        title=source_title,
        tokenCount=len(content_text.split())
    )
    db.add(source)
    db.commit()
    db.refresh(source)

    # 2. Chunk document
    raw_chunks = get_chunks_from_text(content_text)
    items_to_embed = []
    for c_text in raw_chunks:
        clean_c = c_text.replace('\x00', '').replace('\0', '').strip()
        if clean_c:
            items_to_embed.append({
                "sourceId": source.id,
                "content": f"{source_title}\n{clean_c}",
                "document_type": source.kind or "DOC",
                "status": models.DocumentStatus.APPROVED,
                "source_url": source.url,
                "language": "en"
            })

    # 3. Parallel Embed & Bulk Insert
    result = parallel_embed_and_save(
        items=items_to_embed,
        max_workers=max_workers,
        auto_manage_index=auto_manage_index
    )
    result["source_id"] = source.id
    result["source_title"] = source_title
    return result

# =============================================================================
# MULTIMODAL RAG INGESTION (IMAGES, DIAGRAMS & VIDEOS)
# =============================================================================

@router.post("/multimodal/upload-media", summary="Ingest technical diagrams, architectural images, and operational videos")
async def upload_multimodal_media(
    bot_id: str = Form(None),
    is_universal: bool = Form(False),
    title: str = Form(""),
    file: UploadFile = File(...),
    sample_interval_seconds: int = Form(5),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Multimodal RAG Ingestion Pipeline:
    - Diagrams/Images: Deep technical description via Llama-3.2-Vision (entities, flowcharts, relationships), embedded into vector DB.
    - Videos: Keyframe sampling (OpenCV every N sec) + Audio transcription (Whisper) -> Multimodal Fusion Chunking.
    """
    if not is_universal and not bot_id:
        raise HTTPException(status_code=400, detail="Must provide bot_id or set is_universal=True.")

    bot = None
    if bot_id:
        bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")

    file_bytes = await file.read()
    filename = file.filename
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    is_image = ext in ["png", "jpg", "jpeg", "webp"]
    is_video = ext in ["mp4", "mov", "avi", "mkv"]

    if not is_image and not is_video:
        raise HTTPException(status_code=400, detail="Unsupported media format. Please upload PNG, JPG, WEBP, MP4, MOV, or AVI.")

    if is_image:
        # Process Image / Diagram
        img_res = process_image_multimodal(file_bytes, filename)
        source_title = title.strip() or img_res["title"]

        source = models.BotSource(
            botId=bot.id if bot else None,
            isUniversal=is_universal,
            orgId=user.orgId,
            kind="DIAGRAM",
            title=source_title,
            url=img_res["image_url"],
            tokenCount=len(img_res["chunk_content"].split())
        )
        db.add(source)
        db.commit()
        db.refresh(source)

        # Generate embedding for the rich technical description
        emb = get_embedding(img_res["chunk_content"])
        chunk = models.DocumentChunk(
            sourceId=source.id,
            content=img_res["chunk_content"],
            embedding=emb,
            source_url=img_res["image_url"],
            document_type="DIAGRAM",
            status=models.DocumentStatus.APPROVED,
            language="en"
        )
        db.add(chunk)
        db.commit()

        return {
            "status": "success",
            "media_type": "image/diagram",
            "source_id": source.id,
            "title": source_title,
            "image_url": img_res["image_url"],
            "vision_description": img_res["vision_description"],
            "chunks_created": 1
        }

    else:
        # Process Video with OpenCV + Whisper
        video_chunks = process_video_multimodal(
            video_bytes=file_bytes,
            filename=filename,
            sample_interval_seconds=sample_interval_seconds
        )

        source_title = title.strip() or f"Video Guide: {filename}"
        source = models.BotSource(
            botId=bot.id if bot else None,
            isUniversal=is_universal,
            orgId=user.orgId,
            kind="VIDEO",
            title=source_title,
            tokenCount=sum(len(c["content"].split()) for c in video_chunks)
        )
        db.add(source)
        db.commit()
        db.refresh(source)

        # Batch embed and insert video fusion chunks
        items_to_embed = [
            {
                "sourceId": source.id,
                "content": c["content"],
                "document_type": "VIDEO",
                "status": models.DocumentStatus.APPROVED,
                "source_url": c["frame_url"],
                "language": "en"
            }
            for c in video_chunks
        ]

        inserted = parallel_embed_and_save(items_to_embed, max_workers=4, auto_manage_index=False)

        return {
            "status": "success",
            "media_type": "video",
            "source_id": source.id,
            "title": source_title,
            "frames_sampled": len(video_chunks),
            "chunks_created": inserted.get("total_chunks_processed", len(video_chunks))
        }

# =============================================================================
# THREE-LAYER HYBRID SEARCH ENDPOINT (RRF + FLASHRANK)
# =============================================================================

@router.post("/hybrid-search", summary="Execute 3-Layer Hybrid Search with Mathematical RRF")
def execute_hybrid_search(
    query: str = Form(...),
    bot_id: str = Form(None),
    top_k_final: int = Form(4),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Executes Three-Layer Hybrid Search:
    Layer 1: Dense Vector (pgvector)
    Layer 2: PostgreSQL Full-Text & Lexical ILIKE
    Layer 3: Reciprocal Rank Fusion: 1/(60+vec_rank) + 1/(60+lex_rank)
    Layer 4: FlashRank Deep Cross-Encoder
    """
    return three_layer_hybrid_search(
        query=query,
        bot_id=bot_id,
        org_id=user.orgId,
        top_k_final=top_k_final,
        db=db
    )

