from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Form, Request
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db import models
from app.services.rag import stream_rag_pipeline

router = APIRouter(tags=["Public Widget APIs"])

def verify_origin(request: Request, bot: models.Bot):
    """Security: Check if the request origin is in the bot's allowed list"""
    origin = request.headers.get("origin") or request.headers.get("referer") or ""
    origin_clean = origin.replace("https://", "").replace("http://", "").split("/")[0].strip()

    raw_origins = getattr(bot, "allowedOrigins", None) or bot.domain or ""
    allowed = [o.strip() for o in raw_origins.split(",") if o.strip()]

    # Permissive for local testing, dev servers (e.g. Live Server on :5500, :3000, :8000), file:// and wildcards
    if not allowed or "*" in allowed:
        return
    if not origin_clean or origin_clean in ["null", ""]:
        return
    if "localhost" in origin_clean or "127.0.0.1" in origin_clean or "0.0.0.0" in origin_clean:
        return

    if origin_clean not in allowed:
        raise HTTPException(status_code=403, detail="Origin not allowed")

@router.get("/w.js")
def serve_widget_script():
    """Serves the embeddable widget loader script directly from backend"""
    candidates = [
        Path("/app/public/w.js"),
        Path("/app/frontend_public/w.js"),
        Path(__file__).resolve().parent.parent.parent / "public" / "w.js",
        Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "public" / "w.js"
    ]
    for p in candidates:
        if p.is_file():
            return Response(
                content=p.read_text(encoding="utf-8"),
                media_type="application/javascript",
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
    raise HTTPException(status_code=404, detail="w.js not found")

@router.get("/widget/{public_key}")
@router.get("/widget.html")
def serve_widget_html(public_key: str = None):
    """Serves the widget iframe interface directly from backend"""
    candidates = [
        Path("/app/public/widget.html"),
        Path("/app/frontend_public/widget.html"),
        Path(__file__).resolve().parent.parent.parent / "public" / "widget.html",
        Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "public" / "widget.html"
    ]
    for p in candidates:
        if p.is_file():
            return HTMLResponse(
                content=p.read_text(encoding="utf-8"),
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
    raise HTTPException(status_code=404, detail="widget.html not found")


@router.get("/api/public/config/{public_key}")
def get_widget_config(public_key: str, request: Request, db: Session = Depends(get_db)):
    bot = db.query(models.Bot).filter(models.Bot.publicKey == public_key).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found or inactive")

    verify_origin(request, bot)

    suggestions = [s.strip() for s in (bot.suggestions or "").split('\n') if s.strip()]
    if not suggestions:
        import re
        from sqlalchemy import or_
        sources = db.query(models.BotSource).filter(
            or_(
                models.BotSource.botId == bot.id,
                (models.BotSource.isUniversal == True) & (models.BotSource.orgId == bot.orgId)
            )
        ).all()
        for s in sources:
            clean_t = (s.title or '').strip()
            if clean_t:
                clean_name = re.sub(r'\.[a-zA-Z0-9]+$', '', clean_t).split('|')[0].strip()
                if len(clean_name) > 30:
                    clean_name = clean_name[:28] + "..."
                pill = f"Explore {clean_name}" if not clean_name.lower().startswith("explore") else clean_name
                if pill not in suggestions:
                    suggestions.append(pill)
        if not suggestions:
            suggestions = ["👤 Talk to Real Human"]
        elif "👤 Talk to Real Human" not in suggestions:
            suggestions.append("👤 Talk to Real Human")

    return {
        "botId": bot.id,
        "name": bot.name,
        "logoUrl": getattr(bot, "logoUrl", None) or f"https://www.google.com/s2/favicons?domain={bot.domain}&sz=128",
        "template": bot.template,
        "accentColor": bot.accentColor,
        "greeting": bot.greeting,
        "suggestions": suggestions,
        "launcherPosition": bot.launcherPosition
    }

@router.post("/api/public/chat/stream")
async def public_chat_stream(
    request: Request,
    bot_id: str = Form(...),
    question: str = Form(...),
    conversation_id: str = Form(None),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    verify_origin(request, bot)

    # Manage Conversation History
    if not conversation_id:
        conv = models.Conversation(botId=bot.id)
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conversation_id = conv.id

    # Save User Message
    db.add(models.Message(conversationId=conversation_id, role="USER", content=question))
    db.commit()

    return StreamingResponse(
        stream_rag_pipeline(bot.id, question, db),
        media_type="text/event-stream",
        headers={"X-Conversation-Id": conversation_id}
    )

@router.post("/api/public/lead")
def capture_lead(
    request: Request,
    bot_id: str = Form(...),
    name: str = Form(""),
    email: str = Form(""),
    phone: str = Form(""),
    note: str = Form(""),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    verify_origin(request, bot)

    lead = models.Lead(botId=bot.id, name=name, email=email, phone=phone, note=note)
    db.add(lead)
    db.commit()
    return {"status": "success", "message": "Lead captured"}
