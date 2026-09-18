from fastapi import HTTPException, status
from app.db.models import UserRole

# Role hierarchy levels (higher number = higher privilege)
ROLE_HIERARCHY = {
    UserRole.PLATFORM_ADMIN: 100,
    "OWNER": 100,
    "platform_admin": 100,
    UserRole.ORG_ADMIN: 80,
    UserRole.KNOWLEDGE_ADMIN: 60,
    UserRole.SME_APPROVER: 50,
    UserRole.TRAINER: 40,
    UserRole.INTERNAL_EMPLOYEE: 30,
    UserRole.CUSTOMER_PARTNER_ADMIN: 25,
    UserRole.DISTRIBUTOR_DEALER_PARTNER: 20,
    UserRole.B2B_CUSTOMER: 10,
    UserRole.AUDITOR: 15,
    UserRole.MEMBER: 20,
}

def can_assign_role(assigner_role: str, target_role: str) -> bool:
    """
    Prevents privilege escalation.
    A user can only assign roles with a hierarchy level STRICTLY LOWER than their own,
    except PLATFORM_ADMIN who can assign any role.
    """
    if assigner_role in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"]:
        return True
    
    assigner_level = ROLE_HIERARCHY.get(assigner_role, 0)
    target_level = ROLE_HIERARCHY.get(target_role, 999)
    
    return assigner_level > target_level

def check_role_permission(user_role: str, required_roles: list[str]) -> bool:
    """Checks if the user role is within the list of required roles or is super admin."""
    if user_role in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"]:
        return True
    return user_role in required_roles

