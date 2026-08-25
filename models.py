import enum
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, Enum, UniqueConstraint
from sqlalchemy.orm import relationship
from database import Base

def generate_cuid():
    return "c_" + uuid.uuid4().hex[:20]

class RoleEnum(enum.Enum):
    OWNER = "OWNER"
    ADMIN = "ADMIN"
    MEMBER = "MEMBER"

# --- CORE TABLES ---
class User(Base):
    __tablename__ = "User"
    id = Column(String, primary_key=True, default=generate_cuid)
    email = Column(String, unique=True, index=True, nullable=False)
    passwordHash = Column(String, nullable=False)
    name = Column(String, nullable=False)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    organizations = relationship("Organization", back_populates="owner", cascade="all, delete-orphan")
    memberships = relationship("Membership", back_populates="user", cascade="all, delete-orphan")

class Organization(Base):
    __tablename__ = "Organization"
    id = Column(String, primary_key=True, default=generate_cuid)
    name = Column(String, nullable=False)
    ownerId = Column(String, ForeignKey("User.id", ondelete="CASCADE"), nullable=False, index=True)

    owner = relationship("User", back_populates="organizations")
    members = relationship("Membership", back_populates="org", cascade="all, delete-orphan")
    bots = relationship("Bot", back_populates="org", cascade="all, delete-orphan")

class Membership(Base):
    __tablename__ = "Membership"
    id = Column(String, primary_key=True, default=generate_cuid)
    orgId = Column(String, ForeignKey("Organization.id", ondelete="CASCADE"), nullable=False)
    userId = Column(String, ForeignKey("User.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(Enum(RoleEnum), default=RoleEnum.MEMBER)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    org = relationship("Organization", back_populates="members")
    user = relationship("User", back_populates="memberships")
    __table_args__ = (UniqueConstraint('orgId', 'userId', name='unique_org_user_membership'),)


# --- AI & KNOWLEDGE TABLES (NEW) ---
class Bot(Base):
    __tablename__ = "Bot"
    id = Column(String, primary_key=True, default=generate_cuid)
    name = Column(String, nullable=False)
    systemPrompt = Column(String, nullable=True) 
    orgId = Column(String, ForeignKey("Organization.id", ondelete="CASCADE"), nullable=False, index=True)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    org = relationship("Organization", back_populates="bots")
    knowledgeBases = relationship("KnowledgeBase", back_populates="bot", cascade="all, delete-orphan")

class KnowledgeBase(Base):
    __tablename__ = "KnowledgeBase"
    id = Column(String, primary_key=True, default=generate_cuid)
    name = Column(String, nullable=False) 
    botId = Column(String, ForeignKey("Bot.id", ondelete="CASCADE"), nullable=False, index=True)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    bot = relationship("Bot", back_populates="knowledgeBases")
    chunks = relationship("DocumentChunk", back_populates="knowledgeBase", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "DocumentChunk"
    id = Column(String, primary_key=True, default=generate_cuid)
    content = Column(String, nullable=False) 
    embedding = Column(String, nullable=True) 
    knowledgeBaseId = Column(String, ForeignKey("KnowledgeBase.id", ondelete="CASCADE"), nullable=False, index=True)

    knowledgeBase = relationship("KnowledgeBase", back_populates="chunks")