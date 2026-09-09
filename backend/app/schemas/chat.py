from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class ChatRequest(BaseModel):
    botId: str
    question: str

class PublicChatRequest(BaseModel):
    key: str
    question: str
    conversationId: Optional[str] = None
    origin: Optional[str] = ""

class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    createdAt: Optional[datetime] = None

    class Config:
        from_attributes = True

class ConversationResponse(BaseModel):
    id: str
    botId: str
    sessionId: Optional[str] = None
    isHandedOff: bool = False
    status: str = "AI_ACTIVE"
    createdAt: Optional[datetime] = None
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True

