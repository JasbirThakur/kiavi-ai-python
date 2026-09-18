import json
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.db import models
from app.db.models import UserRole
from app.core.dependencies import get_current_user, require_auditor

router = APIRouter(prefix="/api/audit", tags=["Compliance & Audit Trail"])

@router.get("/logs", summary="Inspect immutable compliance audit trail (Auditor/Admin)")
def get_audit_logs(
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    user_email: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    current_user: models.User = Depends(require_auditor),
    db: Session = Depends(get_db)
):
    """
    Returns an append-only audit trail meeting EU MDR, IATF 16949, and ISO 9001 requirements.
    Auditors can inspect who viewed, approved, or downloaded documents, and who completed certifications.
    """
    q = db.query(models.AuditLog)
    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"]:
        q = q.filter(models.AuditLog.orgId == current_user.orgId)

    if action:
        q = q.filter(models.AuditLog.action == action)
    if resource_type:
        q = q.filter(models.AuditLog.resourceType == resource_type)
    if user_email:
        q = q.filter(models.AuditLog.userEmail.ilike(f"%{user_email}%"))

    total = q.count()
    logs = q.order_by(models.AuditLog.createdAt.desc()).offset(offset).limit(limit).all()

    return {
        "total_records": total,
        "offset": offset,
        "limit": limit,
        "audit_logs": [
            {
                "id": l.id,
                "action": l.action,
                "resource_type": l.resourceType,
                "resource_id": l.resourceId,
                "user_email": l.userEmail,
                "user_role": l.userRole,
                "ip_address": l.ipAddress,
                "details": json.loads(l.detailsJson or "{}"),
                "timestamp": l.createdAt.isoformat() if l.createdAt else ""
            }
            for l in logs
        ]
    }

@router.get("/summary", summary="Get compliance statistics and metrics")
def get_audit_summary(
    current_user: models.User = Depends(require_auditor),
    db: Session = Depends(get_db)
):
    """Provides high-level compliance metrics for regulatory inspectors."""
    q = db.query(models.AuditLog)
    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"]:
        q = q.filter(models.AuditLog.orgId == current_user.orgId)

    total_events = q.count()
    doc_approvals = q.filter(models.AuditLog.action == "DOCUMENT_APPROVED").count()
    role_changes = q.filter(models.AuditLog.action == "ROLE_ASSIGNED").count()
    certs_issued = q.filter(models.AuditLog.action == "COURSE_PASSED_CERTIFICATE_AWARDED").count()

    return {
        "organization_id": current_user.orgId,
        "total_audit_events": total_events,
        "document_approvals_count": doc_approvals,
        "role_changes_count": role_changes,
        "certificates_awarded_count": certs_issued,
        "eu_regulatory_frameworks": [
            "EU MDR 2017/745 (Medical Devices)",
            "IATF 16949 / VDA 6.3 (Automotive)",
            "EU CPR 305/2011 (Construction Products)",
            "EU AI Act Regulation 2024/1689 (High-Risk AI)",
            "GDPR Article 32 (Security of Processing)"
        ]
    }

