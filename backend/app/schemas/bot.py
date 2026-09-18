from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class BotCreate(BaseModel):
    name: str
    domain: Optional[str] = ""
    greeting: Optional[str] = "Hi! How can I help?"
    suggestions: Optional[str] = "What do you offer?\nHow much does it cost?\nHow do I get in touch?"
    accentColor: Optional[str] = "#00c48c"
    launcherPosition: Optional[str] = "right"
    webhookUrl: Optional[str] = None
    supportEmail: Optional[str] = "jasbirsingh17050@gmail.com"
    supportPhone: Optional[str] = None

class BotUpdate(BaseModel):
    name: Optional[str] = None
    domain: Optional[str] = None
    greeting: Optional[str] = None
    suggestions: Optional[str] = None
    accentColor: Optional[str] = None
    launcherPosition: Optional[str] = None
    logoUrl: Optional[str] = None
    webhookUrl: Optional[str] = None
    supportEmail: Optional[str] = None
    supportPhone: Optional[str] = None

class BotResponse(BaseModel):
    id: str
    name: str
    domain: str
    publicKey: str
    accentColor: str
    greeting: str
    suggestions: str
    launcherPosition: str
    logoUrl: Optional[str] = None
    webhookUrl: Optional[str] = None
    supportEmail: Optional[str] = None
    supportPhone: Optional[str] = None
    createdAt: Optional[datetime] = None

    class Config:
        from_attributes = True

