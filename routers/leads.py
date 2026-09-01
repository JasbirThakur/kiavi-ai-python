import csv
import io
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db
import models
from services.auth import get_current_user

router = APIRouter(tags=["Leads"])

@router.get("/api/leads/{bot_id}")
def get_bot_leads(
    bot_id: str,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify bot belongs to user's organization
    bot = db.query(models.Bot).filter(
        models.Bot.id == bot_id,
        models.Bot.orgId == user.orgId
    ).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found.")

    leads = (
        db.query(models.Lead)
        .filter(models.Lead.botId == bot_id)
        .order_by(models.Lead.createdAt.desc())
        .all()
    )

    return [
        {
            "id": l.id,
            "name": l.name or "Anonymous Visitor",
            "email": l.email or "-",
            "phone": l.phone or "-",
            "note": l.note or "Captured from widget conversation",
            "date": l.createdAt.strftime("%b %d, %Y %I:%M %p") if l.createdAt else "-"
        }
        for l in leads
    ]

@router.get("/api/leads/{bot_id}/export")
def export_leads_csv(
    bot_id: str,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(
        models.Bot.id == bot_id,
        models.Bot.orgId == user.orgId
    ).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found.")

    leads = (
        db.query(models.Lead)
        .filter(models.Lead.botId == bot_id)
        .order_by(models.Lead.createdAt.desc())
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)
    
    # CSV Header Row
    writer.writerow(["Lead ID", "Name", "Email", "Phone", "Context / Note", "Captured At (UTC)"])

    # CSV Data Rows
    for l in leads:
        writer.writerow([
            l.id,
            l.name or "",
            l.email or "",
            l.phone or "",
            l.note or "",
            l.createdAt.strftime("%Y-%m-%d %H:%M:%S") if l.createdAt else ""
        ])

    output.seek(0)
    filename = f"leads_{bot.name.replace(' ', '_').lower()}_{bot_id[:8]}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.post("/api/public/lead")
def submit_public_lead(
    bot_id: str = Form(...),
    name: str = Form(None),
    email: str = Form(None),
    phone: str = Form(None),
    note: str = Form("Captured via widget prompt"),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found.")

    new_lead = models.Lead(
        botId=bot_id,
        name=name.strip() if name else "Visitor",
        email=email.strip() if email else None,
        phone=phone.strip() if phone else None,
        note=note.strip() if note else "Website lead"
    )
    db.add(new_lead)
    db.commit()
    db.refresh(new_lead)

    # Asynchronously dispatch webhook notification if configured
    webhook_url = getattr(bot, "webhookUrl", None)
    if webhook_url:
        import threading, requests
        def send_webhook():
            try:
                payload = {
                    "event": "lead.created",
                    "bot": {"id": bot.id, "name": bot.name, "domain": bot.domain},
                    "lead": {
                        "id": new_lead.id,
                        "name": new_lead.name,
                        "email": new_lead.email or "-",
                        "phone": new_lead.phone or "-",
                        "note": new_lead.note or ""
                    },
                    "text": f"🔥 *New Lead Captured from {bot.name}!*\n• *Name:* {new_lead.name}\n• *Email:* {new_lead.email or '-'}\n• *Phone:* {new_lead.phone or '-'}\n• *Note:* {new_lead.note or '-'}"
                }
                requests.post(webhook_url, json=payload, timeout=8)
            except Exception as err:
                print(f"[Lead Webhook Error]: {err}")
        threading.Thread(target=send_webhook, daemon=True).start()

    return {"status": "success", "lead_id": new_lead.id}