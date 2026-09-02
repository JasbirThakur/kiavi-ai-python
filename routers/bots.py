from fastapi import APIRouter, Depends, HTTPException, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db
import models
from services.auth import get_current_user

router = APIRouter(prefix="/api/bots", tags=["Bot Management"])

class BotCreateRequest(BaseModel):
    name: str
    domain: str = "appdeft.ai"

class BotAppearanceUpdate(BaseModel):
    name: str
    template: str
    accentColor: str
    greeting: str
    suggestions: str
    launcherPosition: str
    webhookUrl: str = None

@router.get("/")
def get_user_bots(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    bots = db.query(models.Bot).filter(models.Bot.orgId == user.orgId).all()
    bot_list = []
    total_convs = 0
    total_leads = 0
    total_unanswered = 0
    total_messages = 0
    total_tokens = 0

    for b in bots:
        conv_count = db.query(models.Conversation).filter(models.Conversation.botId == b.id).count()
        lead_count = db.query(models.Lead).filter(models.Lead.botId == b.id).count()
        sources = db.query(models.BotSource).filter(models.BotSource.botId == b.id).all()
        bot_tokens = sum(s.tokenCount or 0 for s in sources)
        
        msgs = db.query(models.Message).join(models.Conversation).filter(models.Conversation.botId == b.id).all()
        unans = sum(1 for m in msgs if m.unanswered and m.role == "USER")
        
        total_convs += conv_count
        total_leads += lead_count
        total_messages += len(msgs)
        total_unanswered += unans
        total_tokens += bot_tokens

        bot_list.append({
            "id": b.id,
            "name": b.name,
            "domain": b.domain,
            "template": b.template,
            "accentColor": b.accentColor,
            "conversations": conv_count,
            "tokens": bot_tokens,
            "publicKey": b.publicKey
        })

    answered_pct = round(((total_messages - total_unanswered) / total_messages * 100)) if total_messages > 0 else 100

    return {
        "bots": bot_list,
        "metrics": {
            "conversations": total_convs,
            "leads": total_leads,
            "answered_pct": answered_pct,
            "tokens": f"{round(total_tokens / 1000, 1)}K",
            "unanswered_count": total_unanswered
        }
    }

@router.post("/")
def create_bot(req: BotCreateRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    bot = models.Bot(
        orgId=user.orgId,
        name=req.name,
        domain=req.domain,
        accentColor="#E30613",
        template="classic"
    )
    db.add(bot)
    db.commit()
    db.refresh(bot)
    return {"id": bot.id, "name": bot.name, "publicKey": bot.publicKey}

@router.get("/{bot_id}")
def get_bot_details(bot_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    conv_count = db.query(models.Conversation).filter(models.Conversation.botId == bot.id).count()
    return {
        "id": bot.id,
        "name": bot.name,
        "domain": bot.domain,
        "publicKey": bot.publicKey,
        "template": bot.template,
        "accentColor": bot.accentColor,
        "greeting": bot.greeting,
        "suggestions": bot.suggestions,
        "launcherPosition": bot.launcherPosition,
        "logoUrl": getattr(bot, "logoUrl", None) or "",
        "webhookUrl": getattr(bot, "webhookUrl", None) or "",
        "conversations": conv_count
    }

@router.put("/{bot_id}/appearance")
def update_appearance(bot_id: str, req: BotAppearanceUpdate, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    bot.name = req.name
    bot.template = req.template
    bot.accentColor = req.accentColor
    bot.greeting = req.greeting
    bot.suggestions = req.suggestions
    bot.launcherPosition = req.launcherPosition
    if req.webhookUrl is not None:
        bot.webhookUrl = req.webhookUrl.strip()

    db.commit()
    return {"status": "success", "message": "Appearance saved. It's live on your site now."}

from fastapi import UploadFile, File
from pathlib import Path
from datetime import datetime

LOGO_DIR = Path(__file__).resolve().parent.parent / "static" / "bot_logos"
LOGO_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/{bot_id}/logo")
async def upload_bot_logo(
    bot_id: str,
    file: UploadFile = File(...),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    ext = file.filename.split(".")[-1].lower() if "." in file.filename else "png"
    if ext not in ["png", "jpg", "jpeg", "svg", "webp"]:
        ext = "png"

    filename = f"logo_{bot.id}_{int(datetime.now().timestamp())}.{ext}"
    target_path = LOGO_DIR / filename
    contents = await file.read()
    target_path.write_bytes(contents)

    logo_url = f"/static/bot_logos/{filename}"
    bot.logoUrl = logo_url
    db.commit()

    return {"status": "success", "logoUrl": logo_url}

@router.post("/{bot_id}/test-webhook")
def test_bot_webhook(bot_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    import requests
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.orgId == user.orgId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
        
    url = getattr(bot, "webhookUrl", None)
    if not url:
        return {"status": "skipped", "message": "No webhook URL configured for this bot."}
        
    payload = {
        "event": "lead.created",
        "bot": {"id": bot.id, "name": bot.name, "domain": bot.domain},
        "lead": {
            "name": "Test Lead (Verification)",
            "email": "test@lead.ai",
            "phone": "+1 (555) 019-2834",
            "note": "Test lead dispatched from Kiavi IQ verification tool."
        },
        "text": f"🚀 *[Kiavi IQ Webhook Verification]*: New simulated lead from *{bot.name}* (`test@lead.ai`)."
    }
    
    try:
        res = requests.post(url, json=payload, timeout=6)
        return {"status": "success", "http_status": res.status_code, "message": f"Webhook dispatched successfully (HTTP {res.status_code})."}
    except Exception as e:
        return {"status": "error", "message": f"Webhook failed: {str(e)}"}