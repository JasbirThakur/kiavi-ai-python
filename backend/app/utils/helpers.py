import uuid
import re
from datetime import datetime, timezone
from typing import Optional

def generate_uuid() -> str:
    """Generate a clean UUIDv4 string."""
    return str(uuid.uuid4())

def utc_now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)

def sanitize_string(text: Optional[str]) -> str:
    """Strip whitespace and normalize control characters."""
    if not text:
        return ""
    return re.sub(r'[\r\x00-\x08\x0b\x0c\x0e-\x1f]', '', text).strip()

def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncates text safely at word boundary."""
    if not text or len(text) <= max_length:
        return text or ""
    return text[:max_length].rsplit(' ', 1)[0] + suffix

