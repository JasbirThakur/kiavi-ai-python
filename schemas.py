from pydantic import BaseModel
from typing import Optional

# Frontend jab naya User banayega, toh use ye 3 cheezein bhejni hongi
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

# Frontend jab naya Bot banayega, toh use ye data bhejna hoga
class BotCreate(BaseModel):
    name: str
    orgId: str
    systemPrompt: Optional[str] = None # Ye optional hai

# Frontend jab naya Knowledge Base banayega
class KnowledgeBaseCreate(BaseModel):
    name: str
    botId: str

# Frontend jab document ka text (chunk) bhejega
class DocumentChunkCreate(BaseModel):
    content: str
    knowledgeBaseId: str
    # Note: Embedding (AI vectors) hum frontend se nahi lenge, 
    # wo hum baad mein backend mein hi generate karenge.

# Frontend jab Bot se koi sawaal puchega
class ChatRequest(BaseModel):
    question: str
    botId: str