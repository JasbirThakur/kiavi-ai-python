from typing import Callable
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.database import get_db
from app.db import models
from app.db.models import UserRole

security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> models.User:
    """Extracts and validates user from Bearer Token"""
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub") or payload.get("user_id")
        if not user_id:
            raise credentials_exception
    except Exception:
        raise credentials_exception

    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise credentials_exception

    return user

def require_roles(*allowed_roles: str) -> Callable:
    """
    Factory that returns a FastAPI dependency checking if current_user has one of the allowed_roles.
    PLATFORM_ADMIN bypasses all role checks.
    """
    def role_checker(current_user: models.User = Depends(get_current_user)) -> models.User:
        user_role = (current_user.role or UserRole.INTERNAL_EMPLOYEE).strip()
        # Platform Admin and legacy OWNER have unrestricted access across all modules
        if user_role in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin", "SUPERADMIN", "admin"]:
            return current_user
        
        # Legacy mapping: MEMBER gets INTERNAL_EMPLOYEE permissions
        effective_role = UserRole.INTERNAL_EMPLOYEE if user_role.upper() == "MEMBER" else user_role
        
        if effective_role in allowed_roles or user_role in allowed_roles or user_role.lower() in allowed_roles:
            return current_user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Requires one of roles {list(allowed_roles)}. Current role: '{user_role}'"
        )
    return role_checker

# Pre-defined convenience role dependencies
require_org_admin = require_roles(UserRole.PLATFORM_ADMIN, UserRole.ORG_ADMIN)
require_sme = require_roles(UserRole.PLATFORM_ADMIN, UserRole.ORG_ADMIN, UserRole.SME_APPROVER)
require_knowledge_admin = require_roles(UserRole.PLATFORM_ADMIN, UserRole.ORG_ADMIN, UserRole.KNOWLEDGE_ADMIN)
require_trainer = require_roles(UserRole.PLATFORM_ADMIN, UserRole.ORG_ADMIN, UserRole.TRAINER)
require_auditor = require_roles(UserRole.PLATFORM_ADMIN, UserRole.ORG_ADMIN, UserRole.AUDITOR)
