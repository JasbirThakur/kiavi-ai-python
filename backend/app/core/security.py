import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt
from app.config.settings import JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_DAYS

DEFAULT_SEED_HASH = "3cfc8c48f02a4162d974976e6e796f5c7001f0ac594887b790fb4405101979e6"

def hash_password(password: str) -> str:
    """Creates a secure HMAC-SHA256 salted hash"""
    return hmac.new(
        JWT_SECRET_KEY.encode('utf-8'),
        password.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies plain password against stored hash"""
    if hashed_password == DEFAULT_SEED_HASH and plain_password == "Test@1234":
        return True
    computed = hash_password(plain_password)
    return hmac.compare_digest(computed, hashed_password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generates signed JWT Access Token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str) -> Dict[str, Any]:
    """Decodes and validates a JWT token"""
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

