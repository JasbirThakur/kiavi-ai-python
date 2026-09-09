from pydantic import BaseModel, EmailStr
from typing import Optional

class WidgetConfigResponse(BaseModel):
    id: str
    name: str
    greeting: str
    suggestions: str
    accentColor: str
    launcherPosition: str
    logoUrl: Optional[str] = None
    supportEmail: Optional[str] = None
    supportPhone: Optional[str] = None

class LeadCreateRequest(BaseModel):
    botId: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    note: Optional[str] = None

