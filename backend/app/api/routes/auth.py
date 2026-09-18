from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import models
from app.services.auth import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

from typing import Optional

class RegisterRequest(BaseModel):
    name: Optional[str] = None
    full_name: Optional[str] = None
    email: EmailStr
    password: str
    company_name: str = "AppDeft AI"

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class InviteRequest(BaseModel):
    email: EmailStr
    role: str = "MEMBER"

class UpdateNameRequest(BaseModel):
    name: str

class UpdateEmailRequest(BaseModel):
    new_email: EmailStr
    current_password: str

class UpdatePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class UpdateWorkspaceRequest(BaseModel):
    workspace_name: str

def get_user_pwd(user: models.User) -> str:
    return getattr(user, "passwordHash", None) or getattr(user, "hashedPassword", "")

@router.post("/register")
@router.post("/signup")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    email_clean = req.email.strip().lower()
    existing = db.query(models.User).filter(models.User.email == email_clean).first()
    if existing:
        raise HTTPException(status_code=400, detail="This email is already registered.")

    org = models.Organization(name=req.company_name.strip() or "AppDeft AI")
    db.add(org)
    db.commit()
    db.refresh(org)

    hashed = hash_password(req.password.strip())
    resolved_name = (req.name or req.full_name or email_clean.split("@")[0]).strip()
    user = models.User(
        email=email_clean,
        name=resolved_name or "User",
        role="OWNER",
        orgId=org.id
    )
    if hasattr(user, "passwordHash"):
        user.passwordHash = hashed
    if hasattr(user, "hashedPassword"):
        user.hashedPassword = hashed

    db.add(user)
    db.commit()
    db.refresh(user)

    starter_bot = models.Bot(
        orgId=org.id,
        name="Test Boat",
        domain="",
        template="classic",
        accentColor="#00c48c",
        greeting="Hi! How can I help?",
        suggestions="What do you offer?\nHow much does it cost?\nHow do I get in touch?",
        launcherPosition="right"
    )
    db.add(starter_bot)
    db.commit()
    db.refresh(starter_bot)

    token = create_access_token({"sub": user.id, "email": user.email, "org_id": user.orgId})
    return {
        "status": "success",
        "token": token,
        "access_token": token,
        "user": {"id": user.id, "name": user.name, "email": user.email},
        "bot": {"id": starter_bot.id, "name": starter_bot.name, "publicKey": starter_bot.publicKey}
    }

@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    email_clean = req.email.strip().lower()
    user = db.query(models.User).filter(models.User.email == email_clean).first()
    if not user or not verify_password(req.password.strip(), get_user_pwd(user)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")

    bot = db.query(models.Bot).filter(models.Bot.orgId == user.orgId).first()
    token = create_access_token({"sub": user.id, "email": user.email, "org_id": user.orgId})
    return {
        "status": "success",
        "token": token,
        "access_token": token,
        "user": {"id": user.id, "name": user.name, "email": user.email},
        "bot": {"id": bot.id, "name": bot.name, "publicKey": bot.publicKey} if bot else None
    }

@router.get("/me")
def get_me(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    bot = db.query(models.Bot).filter(models.Bot.orgId == user.orgId).first()
    org = db.query(models.Organization).filter(models.Organization.id == user.orgId).first()
    return {
        "user": {"id": user.id, "name": user.name, "email": user.email, "role": getattr(user, "role", "OWNER")},
        "organization": {"id": org.id, "name": org.name if org else "AppDeft AI"},
        "bot": {
            "id": bot.id,
            "name": bot.name,
            "publicKey": bot.publicKey,
            "domain": bot.domain,
            "accentColor": bot.accentColor,
            "template": bot.template
        } if bot else None
    }

# --- TEAM ENDPOINTS ---

@router.get("/team")
def get_team(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    members = db.query(models.User).filter(models.User.orgId == user.orgId).order_by(models.User.createdAt.asc()).all()
    return [
        {
            "id": m.id,
            "name": m.name or m.email.split("@")[0],
            "email": m.email,
            "role": getattr(m, "role", "MEMBER"),
            "isCurrent": m.id == user.id,
            "date": m.createdAt.strftime("%b %d, %Y") if getattr(m, "createdAt", None) else "Recently"
        }
        for m in members
    ]

@router.post("/team/invite")
def invite_team_member(req: InviteRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already registered.")

    hashed = hash_password("KiaviTempPassword2026!")
    new_m = models.User(
        email=req.email,
        name=req.email.split("@")[0].capitalize(),
        role=req.role.upper(),
        orgId=user.orgId
    )
    if hasattr(new_m, "passwordHash"):
        new_m.passwordHash = hashed
    if hasattr(new_m, "hashedPassword"):
        new_m.hashedPassword = hashed

    db.add(new_m)
    db.commit()
    return {"status": "success", "message": f"Member {req.email} invited."}

@router.delete("/team/{member_id}")
def remove_team_member(member_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    target = db.query(models.User).filter(models.User.id == member_id, models.User.orgId == user.orgId).first()
    if not target:
        raise HTTPException(status_code=404, detail="Member not found.")
    if getattr(target, "role", "") == "OWNER" or target.id == user.id:
        raise HTTPException(status_code=400, detail="Cannot remove owner or self.")
    db.delete(target)
    db.commit()
    return {"status": "success"}

# --- ACCOUNT SETTINGS ---

@router.put("/update-name")
def update_name(req: UpdateNameRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    user.name = req.name.strip()
    db.commit()
    return {"status": "success", "name": user.name}

@router.put("/update-email")
def update_email(req: UpdateEmailRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(req.current_password.strip(), get_user_pwd(user)):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    user.email = req.new_email.strip().lower()
    db.commit()
    return {"status": "success", "email": user.email}

@router.put("/update-password")
def update_password(req: UpdatePasswordRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(req.current_password.strip(), get_user_pwd(user)):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    hashed = hash_password(req.new_password.strip())
    if hasattr(user, "passwordHash"):
        user.passwordHash = hashed
    if hasattr(user, "hashedPassword"):
        user.hashedPassword = hashed
    db.commit()
    return {"status": "success", "message": "Password updated successfully."}

@router.put("/update-workspace")
def update_workspace(req: UpdateWorkspaceRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    org = db.query(models.Organization).filter(models.Organization.id == user.orgId).first()
    if org:
        org.name = req.workspace_name.strip()
        db.commit()
    return {"status": "success", "workspace": req.workspace_name}

@router.delete("/delete-account")
def delete_account(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    org = db.query(models.Organization).filter(models.Organization.id == user.orgId).first()
    if org:
        db.delete(org)
        db.commit()
    return {"status": "success", "message": "Account deleted."}
