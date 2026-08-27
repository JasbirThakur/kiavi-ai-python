from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
import models
from services.auth import get_current_user

router = APIRouter(prefix="/api/insights", tags=["Insights"])

@router.get("/{bot_id}")
def get_bot_insights(
    bot_id: str,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        gaps = []
        seen = set()

        # 1. Fetch leads captured during conversations as missing info gaps
        leads = db.query(models.Lead).filter(models.Lead.botId == bot_id).order_by(models.Lead.createdAt.desc()).all()
        for l in leads:
            note_text = l.note or "Pricing / general query"
            if note_text not in seen:
                seen.add(note_text)
                gaps.append({
                    "id": str(l.id),
                    "question": note_text,
                    "date": l.createdAt.strftime("%b %d, %Y %I:%M %p") if getattr(l, "createdAt", None) else "Recently"
                })

        # 2. Fetch unanswered conversation messages
        try:
            conv_ids = [c.id for c in db.query(models.Conversation).filter(models.Conversation.botId == bot_id).all()]
            if conv_ids:
                msgs = db.query(models.Message).filter(models.Message.conversationId.in_(conv_ids)).all()
                for m in msgs:
                    if getattr(m, "unanswered", False) and m.content not in seen:
                        seen.add(m.content)
                        gaps.append({
                            "id": str(m.id),
                            "question": m.content,
                            "date": m.createdAt.strftime("%b %d, %Y %I:%M %p") if getattr(m, "createdAt", None) else "Recently"
                        })
        except Exception:
            pass

        return {
            "total_unanswered": len(gaps),
            "gaps": gaps
        }
    except Exception as e:
        print(f"Insights error: {e}")
        return {"total_unanswered": 0, "gaps": []}