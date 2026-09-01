from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from database import get_db
import models
from services.ingestion import extract_file_text, get_chunks_from_text
from services.embedding import get_embedding
from services.scraper import scrape_url_content
from services.auth import get_current_user

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge Ingestion"])

@router.get("/sources/{bot_id}")
def get_sources(bot_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Security: Ensure bot belongs to the user's organization
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    sources = db.query(models.BotSource).filter(models.BotSource.botId == bot_id).order_by(models.BotSource.createdAt.desc()).all()
    used_tokens = sum(s.tokenCount or 0 for s in sources)
    
    return {
        "sources": [{"id": s.id, "kind": s.kind, "title": s.title, "tokens": s.tokenCount or 0} for s in sources],
        "used_tokens": used_tokens,
        "max_tokens": 60000,
    }

@router.delete("/sources/{source_id}")
def delete_source(source_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    source = db.query(models.BotSource).filter(models.BotSource.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
        
    # Security: Check if user owns the bot connected to this source
    bot = db.query(models.Bot).filter(models.Bot.id == source.botId, models.Bot.orgId == user.orgId).first()
    if not bot:
        raise HTTPException(status_code=403, detail="Not authorized to delete this source")

    db.delete(source)
    db.commit()
    return {"ok": True}

@router.post("/ingest-mixed")
async def ingest_mixed(
    bot_id: str = Form(...),
    pasted_text: str = Form(""),
    title: str = Form(""),
    file: UploadFile = File(None),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    if file and file.filename:
        file_bytes = await file.read()
        text_content = extract_file_text(file_bytes, file.filename)
        source_title = file.filename
        kind = "FILE"
    elif pasted_text.strip():
        text_content = pasted_text.strip()
        source_title = title.strip() or "Hand-written Note"
        kind = "TEXT"
    else:
        raise HTTPException(status_code=400, detail="Provide a file or text content.")

    return process_and_save_source(db, bot.id, source_title, text_content, kind)

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

        return process_and_save_source(db, bot.id, title, text_content, "PAGE", original_url=url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def process_and_save_source(db: Session, bot_id: str, title: str, content: str, kind: str, original_url: str = ""):
    """Helper function to save source, create chunks, and generate PostgreSQL vectors"""
    if not content.strip():
        raise HTTPException(status_code=400, detail="No readable text extracted.")

    # 1. Save Parent Source
    token_count = len(content.split())
    source = models.BotSource(
        botId=bot_id,
        kind=kind,
        title=title,
        url=original_url or None,
        tokenCount=token_count
    )
    db.add(source)
    db.commit()
    db.refresh(source)

    # 2. Chunking & Embeddings
    chunks = get_chunks_from_text(content)
    for text_chunk in chunks:
        context_title = f"{title} ({original_url})" if original_url else title
        chunk_with_title = f"{context_title}\n{text_chunk}"
        
        # HuggingFace MiniLM Embedding
        vec = get_embedding(chunk_with_title)
        
        # Save to PostgreSQL native vector column
        db.add(models.DocumentChunk(sourceId=source.id, content=chunk_with_title, embedding=vec))

    db.commit()
    return {"status": "success", "title": title, "chunks_created": len(chunks)}