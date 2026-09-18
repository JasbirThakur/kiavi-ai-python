from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional, List

from app.db.database import get_db
from app.db import models
from app.db.models import UserRole
from app.core.dependencies import get_current_user, require_org_admin
from app.core.rbac import can_assign_role, ROLE_HIERARCHY
from app.core.security import hash_password
from app.services.audit import log_audit_event

router = APIRouter(prefix="/api/admin/users", tags=["User Management & RBAC"])

class AssignRoleRequest(BaseModel):
    user_id: str
    role: str

class CreateUserRequest(BaseModel):
    email: EmailStr
    name: str
    password: str
    role: str = UserRole.INTERNAL_EMPLOYEE
    department: Optional[str] = None
    region: Optional[str] = "EU"
    partner_org_id: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    role: str
    department: Optional[str]
    region: Optional[str]
    partnerOrgId: Optional[str]
    createdAt: str

@router.get("/roles", summary="Get all 9 platform roles and their permissions")
def get_available_roles(current_user: models.User = Depends(get_current_user)):
    """Returns all 9 platform roles with descriptions and permission scopes."""
    roles_info = [
        {"role": UserRole.PLATFORM_ADMIN, "title": "Platform Administrator", "category": "System Admin", "description": "SaaS Super Admin managing global infrastructure and tenant provisioning."},
        {"role": UserRole.ORG_ADMIN, "title": "Organization Administrator", "category": "System Admin", "description": "Tenant-level Admin managing company users, roles, settings, and bots."},
        {"role": UserRole.KNOWLEDGE_ADMIN, "title": "Knowledge Administrator", "category": "Knowledge Curator", "description": "Manages document categorization, taxonomy, and stale content archiving."},
        {"role": UserRole.SME_APPROVER, "title": "Subject Matter Expert (SME)", "category": "Knowledge Curator", "description": "Gatekeeper reviewing and formally approving technical documents for AI retrieval."},
        {"role": UserRole.TRAINER, "title": "Trainer / Learning Admin", "category": "Training & LMS", "description": "Authors courses, configs video/doc lessons, sets quiz thresholds, and monitors completions."},
        {"role": UserRole.INTERNAL_EMPLOYEE, "title": "Internal Employee", "category": "Daily Consumer", "description": "Internal staff querying AI assistant and completing mandatory corporate training."},
        {"role": UserRole.CUSTOMER_PARTNER_ADMIN, "title": "Customer / Partner Admin", "category": "External Partner", "description": "Sub-tenant dealer/partner admin managing local technicians and viewing team certs."},
        {"role": UserRole.DISTRIBUTOR_DEALER_PARTNER, "title": "Distributor / Dealer / Partner", "category": "External Partner", "description": "Authorized workshops and dealers accessing partner-tier catalogs and installation guides."},
        {"role": UserRole.B2B_CUSTOMER, "title": "B2B Customer", "category": "Daily Consumer", "description": "Hospital procurement or contractors accessing public/approved safety datasheets."},
        {"role": UserRole.AUDITOR, "title": "Auditor / Compliance Reviewer", "category": "Governance", "description": "Read-only access to document version history, SME approval logs, and employee training proofs."}
    ]
    return {"roles": roles_info}

@router.get("", response_model=List[UserResponse], summary="List all users in the organization")
def list_organization_users(
    current_user: models.User = Depends(require_org_admin),
    db: Session = Depends(get_db)
):
    """Lists all users belonging to the current organization with their roles and metadata."""
    query = db.query(models.User)
    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"]:
        query = query.filter(models.User.orgId == current_user.orgId)
    
    users = query.order_by(models.User.createdAt.desc()).all()
    return [
        UserResponse(
            id=u.id,
            email=u.email,
            name=u.name,
            role=u.role or UserRole.INTERNAL_EMPLOYEE,
            department=u.department,
            region=u.region or "EU",
            partnerOrgId=u.partnerOrgId,
            createdAt=u.createdAt.isoformat() if u.createdAt else ""
        )
        for u in users
    ]

@router.post("/assign-role", summary="Assign role to a user with privilege escalation prevention")
def assign_user_role(
    req: AssignRoleRequest,
    request: Request,
    current_user: models.User = Depends(require_org_admin),
    db: Session = Depends(get_db)
):
    """
    Assigns a role to a user.
    Validates that:
    1. The target role is valid.
    2. Zero privilege escalation: current user cannot assign a role equal to or higher than their own.
    3. Writes an immutable audit event log.
    """
    if req.role not in UserRole.ALL_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{req.role}'. Valid roles: {UserRole.ALL_ROLES}"
        )

    # Privilege Escalation Guard
    if not can_assign_role(current_user.role, req.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Privilege escalation denied: A user with role '{current_user.role}' cannot assign role '{req.role}'"
        )

    target_user = db.query(models.User).filter(models.User.id == req.user_id).first()
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if current_user.role != UserRole.PLATFORM_ADMIN and target_user.orgId != current_user.orgId:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot modify user from another organization")

    prev_role = target_user.role
    target_user.role = req.role
    db.commit()

    # Log immutable audit event
    client_ip = request.client.host if request.client else "unknown"
    log_audit_event(
        db=db,
        org_id=current_user.orgId,
        action="ROLE_ASSIGNED",
        resource_type="user",
        resource_id=target_user.id,
        user=current_user,
        details={
            "target_user_email": target_user.email,
            "previous_role": prev_role,
            "new_role": req.role
        },
        ip_address=client_ip
    )

    return {
        "status": "success",
        "message": f"Assigned role '{req.role}' to {target_user.email}",
        "user_id": target_user.id,
        "new_role": target_user.role
    }

@router.post("/create", summary="Create or invite a new user in organization")
def create_organization_user(
    req: CreateUserRequest,
    request: Request,
    current_user: models.User = Depends(require_org_admin),
    db: Session = Depends(get_db)
):
    """Creates a new user within the organization with specified role, department, and region."""
    if req.role not in UserRole.ALL_ROLES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid role '{req.role}'")

    if not can_assign_role(current_user.role, req.role):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Cannot assign role '{req.role}' due to privilege restrictions")

    existing = db.query(models.User).filter(models.User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User with this email already exists")

    new_user = models.User(
        email=req.email,
        name=req.name,
        hashedPassword=hash_password(req.password),
        role=req.role,
        department=req.department,
        region=req.region or "EU",
        partnerOrgId=req.partner_org_id,
        orgId=current_user.orgId
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    client_ip = request.client.host if request.client else "unknown"
    log_audit_event(
        db=db,
        org_id=current_user.orgId,
        action="USER_CREATED",
        resource_type="user",
        resource_id=new_user.id,
        user=current_user,
        details={"email": new_user.email, "role": new_user.role, "department": new_user.department},
        ip_address=client_ip
    )

    return {
        "status": "success",
        "message": f"User {new_user.email} created with role '{new_user.role}'",
        "user_id": new_user.id
    }
