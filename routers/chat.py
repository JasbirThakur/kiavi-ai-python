from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db
import models
from services.auth import get_current_user
from services.rag import stream_rag_pipeline

router = APIRouter(tags=["Chat"])

@router.post("/api/chat/stream")
async def chat_stream_auth(
    bot_id: str = Form(...),
    question: str = Form(...),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    conv = models.Conversation(botId=bot_id, isTest=True)
    db.add(conv)
    db.commit()
    db.refresh(conv)

    user_msg = models.Message(
        conversationId=conv.id,
        role=models.RoleEnum.USER if hasattr(models, "RoleEnum") else "USER",
        content=question.strip()
    )
    if hasattr(user_msg, "unanswered"):
        user_msg.unanswered = False
    db.add(user_msg)
    db.commit()

    return StreamingResponse(
        stream_rag_pipeline(bot_id, question, db),
        media_type="text/event-stream"
    )

@router.post("/api/public/chat/stream")
async def chat_stream_public(
    bot_id: str = Form(...),
    question: str = Form(...),
    session_id: str = Form(None),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    conv = models.Conversation(botId=bot_id, sessionId=session_id, isTest=False)
    db.add(conv)
    db.commit()
    db.refresh(conv)

    user_msg = models.Message(
        conversationId=conv.id,
        role=models.RoleEnum.USER if hasattr(models, "RoleEnum") else "USER",
        content=question.strip()
    )
    if hasattr(user_msg, "unanswered"):
        user_msg.unanswered = False
    db.add(user_msg)
    db.commit()

    return StreamingResponse(
        stream_rag_pipeline(bot_id, question, db),
        media_type="text/event-stream"
    )

@router.post("/api/public/chat/request-agent")
def request_human_agent(
    bot_id: str = Form(...),
    conversation_id: str = Form(None),
    session_id: str = Form(None),
    db: Session = Depends(get_db)
):
    conv = None
    if conversation_id:
        conv = db.query(models.Conversation).filter(models.Conversation.id == conversation_id).first()
    if not conv:
        conv = models.Conversation(botId=bot_id, sessionId=session_id, isTest=False)
        db.add(conv)
        db.commit()
        db.refresh(conv)

    conv.isHandedOff = True
    conv.status = "HUMAN_REQUESTED"
    
    sys_msg = models.Message(
        conversationId=conv.id,
        role="SYSTEM",
        content="🔔 Visitor requested live human agent."
    )
    db.add(sys_msg)
    db.commit()
    return {"status": "success", "conversation_id": conv.id}


@router.post("/api/chat/conversations/{conv_id}/agent-reply")
def send_agent_reply(
    conv_id: str,
    message: str = Form(...),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conv = db.query(models.Conversation).filter(models.Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.isHandedOff = True
    conv.status = "HUMAN_TAKEN_OVER"

    agent_msg = models.Message(
        conversationId=conv.id,
        role="AGENT",
        content=message.strip()
    )
    db.add(agent_msg)
    db.commit()
    return {"status": "success", "message_id": agent_msg.id}


@router.get("/api/public/chat/poll/{conv_id}")
def poll_conversation_messages(
    conv_id: str,
    db: Session = Depends(get_db)
):
    conv = db.query(models.Conversation).filter(models.Conversation.id == conv_id).first()
    if not conv:
        return {"status": "error", "messages": []}

    msgs = (
        db.query(models.Message)
        .filter(models.Message.conversationId == conv.id)
        .order_by(models.Message.createdAt.asc())
        .all()
    )

    return {
        "status": "success",
        "conversation_id": conv.id,
        "is_handed_off": conv.isHandedOff,
        "conv_status": conv.status or "AI_ACTIVE",
        "messages": [
            {
                "id": m.id,
                "role": str(m.role),
                "content": m.content,
                "time": m.createdAt.strftime("%I:%M %p") if getattr(m, "createdAt", None) else ""
            }
            for m in msgs
        ]
    }


@router.get("/api/chat/history/{bot_id}")
def get_chat_history(
    bot_id: str,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        conversations = (
            db.query(models.Conversation)
            .filter(models.Conversation.botId == bot_id)
            .order_by(models.Conversation.createdAt.desc())
            .limit(30)
            .all()
        )

        history = []
        for conv in conversations:
            msgs = (
                db.query(models.Message)
                .filter(models.Message.conversationId == conv.id)
                .order_by(models.Message.createdAt.asc())
                .all()
            )
            history.append({
                "id": str(conv.id),
                "isTest": getattr(conv, "isTest", False),
                "isHandedOff": getattr(conv, "isHandedOff", False),
                "status": getattr(conv, "status", "AI_ACTIVE") or "AI_ACTIVE",
                "date": conv.createdAt.strftime("%b %d, %Y • %I:%M %p") if getattr(conv, "createdAt", None) else "Recently",
                "messages": [
                    {
                        "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                        "content": m.content,
                        "unanswered": getattr(m, "unanswered", False),
                        "time": m.createdAt.strftime("%I:%M %p") if getattr(m, "createdAt", None) else ""
                    }
                    for m in msgs
                ]
            })
        return history
    except Exception as e:
        print(f"Chat history error: {e}")
        return []