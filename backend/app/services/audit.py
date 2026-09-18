import json
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.db.models import AuditLog, User
from app.utils.logger import logger

def log_audit_event(
    db: Session,
    org_id: str,
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    user: User | None = None,
    details: dict | None = None,
    ip_address: str | None = None
) -> AuditLog:
    """
    Writes an immutable event record into the audit_logs table for regulatory compliance
    (EU MDR, IATF 16949, EU AI Act, GDPR).
    """
    try:
        user_id = user.id if user else "SYSTEM"
        user_email = user.email if user else "system@kiavi.internal"
        user_role = user.role if user else "SYSTEM"

        log_entry = AuditLog(
            orgId=org_id,
            userId=user_id,
            userEmail=user_email,
            userRole=user_role,
            action=action,
            resourceType=resource_type,
            resourceId=str(resource_id) if resource_id else None,
            detailsJson=json.dumps(details) if details else None,
            ipAddress=ip_address,
            createdAt=datetime.now(timezone.utc)
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        logger.info(f"📋 [AUDIT LOG] {action} on {resource_type}:{resource_id} by {user_email} ({user_role})")
        return log_entry
    except Exception as e:
        logger.error(f"❌ Failed to write audit log: {e}")
        db.rollback()
        return None

