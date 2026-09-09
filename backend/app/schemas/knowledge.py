from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class WebsiteScrapeRequest(BaseModel):
    url: str
    botId: Optional[str] = None
    isUniversal: bool = False

class KnowledgeSourceResponse(BaseModel):
    id: str
    botId: Optional[str]
    isUniversal: bool
    kind: str
    title: str
    url: Optional[str] = None
    tokenCount: int = 0
    createdAt: Optional[datetime] = None

    class Config:
        from_attributes = True

