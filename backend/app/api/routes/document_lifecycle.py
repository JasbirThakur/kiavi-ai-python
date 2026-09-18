from datetime import datetime, timezone, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.db import models
from app.db.models import DocumentStatus, UserRole
from app.core.dependencies import get_current_user, require_roles, require_sme
from app.services.audit import log_audit_event

router = APIRouter(prefix="/api/documents", tags=["Document Lifecycle & Verification"])

class ReviewActionRequest(BaseModel):
    notes: Optional[str] = None
    expiry_date: Optional[str] = None # ISO format e.g. "2027-12-31T00:00:00Z"
    version: Optional[str] = None # e.g. "v1.1"

class DocumentSummary(BaseModel):
    id: str
    title: str
    kind: str
    status: str
    version: str
    approvedBy: Optional[str]
    approvedAt: Optional[str]
    expiryDate: Optional[str]
    tokenCount: int
    createdAt: str

@router.get("/approvals/pending", response_model=List[DocumentSummary], summary="List all documents awaiting SME verification")
def list_pending_approvals(
    current_user: models.User = Depends(require_sme),
    db: Session = Depends(get_db)
):
    """
    Returns documents currently in UNDER_REVIEW status that require formal SME gatekeeping.
    Only SME Approvers, Org Admins, and Platform Admins can access this queue.
    """
    query = db.query(models.BotSource).filter(models.BotSource.status == DocumentStatus.UNDER_REVIEW)
    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"]:
        query = query.filter(models.BotSource.orgId == current_user.orgId)

    pending_docs = query.order_by(models.BotSource.createdAt.asc()).all()
    return [
        DocumentSummary(
            id=d.id,
            title=d.title,
            kind=d.kind,
            status=d.status or DocumentStatus.DRAFT,
            version=d.version or "v1.0",
            approvedBy=d.approvedBy,
            approvedAt=d.approvedAt.isoformat() if d.approvedAt else None,
            expiryDate=d.expiryDate.isoformat() if d.expiryDate else None,
            tokenCount=d.tokenCount or 0,
            createdAt=d.createdAt.isoformat() if d.createdAt else ""
        )
        for d in pending_docs
    ]

@router.post("/{source_id}/submit-for-review", summary="Submit a draft document for formal SME review")
def submit_for_review(
    source_id: str,
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Transitions document status from DRAFT to UNDER_REVIEW."""
    doc = db.query(models.BotSource).filter(models.BotSource.id == source_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"] and doc.orgId != current_user.orgId:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")

    doc.status = DocumentStatus.UNDER_REVIEW
    # Ensure chunks are also marked UNDER_REVIEW so AI cannot retrieve them yet
    db.query(models.DocumentChunk).filter(models.DocumentChunk.sourceId == doc.id).update(
        {"status": DocumentStatus.UNDER_REVIEW}
    )
    db.commit()

    client_ip = request.client.host if request.client else "unknown"
    log_audit_event(
        db=db,
        org_id=doc.orgId,
        action="DOCUMENT_SUBMITTED_FOR_REVIEW",
        resource_type="document",
        resource_id=doc.id,
        user=current_user,
        details={"title": doc.title, "version": doc.version},
        ip_address=client_ip
    )

    return {"status": "success", "message": f"Document '{doc.title}' submitted for SME review", "doc_status": doc.status}

@router.post("/{source_id}/approve", summary="SME formally approves document for AI retrieval")
def approve_document(
    source_id: str,
    req: ReviewActionRequest,
    request: Request,
    current_user: models.User = Depends(require_sme),
    db: Session = Depends(get_db)
):
    """
    SME Approval Gate:
    1. Formally sets document status to APPROVED.
    2. Records approvedBy, approvedAt, and optional expiryDate.
    3. Promotes all linked document_chunks to APPROVED so AI assistant can ground answers on it.
    4. Writes immutable compliance audit log.
    """
    doc = db.query(models.BotSource).filter(models.BotSource.id == source_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"] and doc.orgId != current_user.orgId:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")

    now = datetime.now(timezone.utc)
    doc.status = DocumentStatus.APPROVED
    doc.approvedBy = current_user.id
    doc.approvedAt = now
    if req.version:
        doc.version = req.version
    if req.expiry_date:
        try:
            doc.expiryDate = datetime.fromisoformat(req.expiry_date.replace("Z", "+00:00"))
        except Exception:
            pass

    # Promote all chunks to APPROVED
    db.query(models.DocumentChunk).filter(models.DocumentChunk.sourceId == doc.id).update(
        {"status": DocumentStatus.APPROVED}
    )
    db.commit()

    client_ip = request.client.host if request.client else "unknown"
    log_audit_event(
        db=db,
        org_id=doc.orgId,
        action="DOCUMENT_APPROVED",
        resource_type="document",
        resource_id=doc.id,
        user=current_user,
        details={
            "title": doc.title,
            "version": doc.version,
            "notes": req.notes,
            "expiry_date": doc.expiryDate.isoformat() if doc.expiryDate else None
        },
        ip_address=client_ip
    )

    return {
        "status": "success",
        "message": f"Document '{doc.title}' ({doc.version}) formally approved by {current_user.email}",
        "doc_id": doc.id,
        "doc_status": doc.status,
        "approved_at": doc.approvedAt.isoformat()
    }

@router.post("/{source_id}/reject", summary="SME rejects document with feedback notes")
def reject_document(
    source_id: str,
    req: ReviewActionRequest,
    request: Request,
    current_user: models.User = Depends(require_sme),
    db: Session = Depends(get_db)
):
    """Rejects document back to DRAFT state. Keeps chunks locked from AI retrieval."""
    doc = db.query(models.BotSource).filter(models.BotSource.id == source_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"] and doc.orgId != current_user.orgId:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")

    doc.status = DocumentStatus.DRAFT
    db.query(models.DocumentChunk).filter(models.DocumentChunk.sourceId == doc.id).update(
        {"status": DocumentStatus.DRAFT}
    )
    db.commit()

    client_ip = request.client.host if request.client else "unknown"
    log_audit_event(
        db=db,
        org_id=doc.orgId,
        action="DOCUMENT_REJECTED",
        resource_type="document",
        resource_id=doc.id,
        user=current_user,
        details={"title": doc.title, "rejection_notes": req.notes},
        ip_address=client_ip
    )

    return {"status": "success", "message": f"Document '{doc.title}' returned to Draft", "doc_status": doc.status}

@router.get("/expiring", response_model=List[DocumentSummary], summary="List certificates and documents approaching expiration")
def list_expiring_documents(
    days: int = 60,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Returns documents whose expiryDate is within `days` from now.
    Crucial for CE Mark, ISO certificates, and EU MDR periodic validity.
    """
    now = datetime.now(timezone.utc)
    threshold = now + timedelta(days=days)

    query = db.query(models.BotSource).filter(
        models.BotSource.expiryDate.isnot(None),
        models.BotSource.expiryDate <= threshold,
        models.BotSource.status == DocumentStatus.APPROVED
    )
    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"]:
        query = query.filter(models.BotSource.orgId == current_user.orgId)

    expiring_docs = query.order_by(models.BotSource.expiryDate.asc()).all()
    return [
        DocumentSummary(
            id=d.id,
            title=d.title,
            kind=d.kind,
            status=d.status,
            version=d.version or "v1.0",
            approvedBy=d.approvedBy,
            approvedAt=d.approvedAt.isoformat() if d.approvedAt else None,
            expiryDate=d.expiryDate.isoformat() if d.expiryDate else None,
            tokenCount=d.tokenCount or 0,
            createdAt=d.createdAt.isoformat() if d.createdAt else ""
        )
        for d in expiring_docs
    ]

