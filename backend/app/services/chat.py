from typing import Optional, Tuple
from sqlalchemy.orm import Session
from app.db import models
from app.utils.logger import logger

def get_or_create_conversation(
    db: Session,
    bot_id: str,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    is_test: bool = False
) -> models.Conversation:
    """Retrieves an existing conversation or creates a new one."""
    conv = None
    if conversation_id:
        conv = db.query(models.Conversation).filter(models.Conversation.id == conversation_id).first()
    elif session_id:
        conv = (
            db.query(models.Conversation)
            .filter(models.Conversation.botId == bot_id, models.Conversation.sessionId == session_id)
            .order_by(models.Conversation.createdAt.desc())
            .first()
        )

    if not conv:
        conv = models.Conversation(botId=bot_id, sessionId=session_id, isTest=is_test)
        db.add(conv)
        db.commit()
        db.refresh(conv)

    return conv

def save_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    unanswered: bool = False
) -> models.Message:
    """Persists a new message in the database."""
    msg = models.Message(
        conversationId=conversation_id,
        role=role.upper(),
        content=content.strip(),
        unanswered=unanswered
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg

def set_conversation_status(
    db: Session,
    conversation_id: str,
    status: str,
    is_handed_off: Optional[bool] = None
) -> Optional[models.Conversation]:
    """Updates the status of a conversation (e.g. AI_ACTIVE, HUMAN_REQUESTED, HUMAN_TAKEN_OVER)."""
    conv = db.query(models.Conversation).filter(models.Conversation.id == conversation_id).first()
    if conv:
        conv.status = status
        if is_handed_off is not None:
            conv.isHandedOff = is_handed_off
        db.commit()
        db.refresh(conv)
    return conv

