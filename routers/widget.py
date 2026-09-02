from fastapi import APIRouter, Depends, HTTPException, Form, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from sqlalchemy.orm import Session
from database import get_db
import models
from services.rag import stream_rag_pipeline

router = APIRouter(tags=["Public Widget APIs"])

def verify_origin(request: Request, bot: models.Bot):
    """Security: Check if the request origin is in the bot's allowed list"""
    origin = request.headers.get("origin") or request.headers.get("referer") or ""
    origin_clean = origin.replace("https://", "").replace("http://", "").split("/")[0].strip()

    raw_origins = getattr(bot, "allowedOrigins", None) or bot.domain or ""
    allowed = [o.strip() for o in raw_origins.split(",") if o.strip()]
    # For testing, we allow localhost/127.0.0.1
    if origin_clean and origin_clean not in allowed and "localhost" not in origin_clean and "127.0.0.1" not in origin_clean:
        raise HTTPException(status_code=403, detail="Origin not allowed")

@router.get("/api/public/config/{public_key}")
def get_widget_config(public_key: str, request: Request, db: Session = Depends(get_db)):
    bot = db.query(models.Bot).filter(models.Bot.publicKey == public_key).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found or inactive")

    verify_origin(request, bot)

    return {
        "botId": bot.id,
        "name": bot.name,
        "logoUrl": getattr(bot, "logoUrl", None) or f"https://www.google.com/s2/favicons?domain={bot.domain}&sz=128",
        "template": bot.template,
        "accentColor": bot.accentColor,
        "greeting": bot.greeting,
        "suggestions": [s.strip() for s in (bot.suggestions or "").split('\n') if s.strip()],
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
