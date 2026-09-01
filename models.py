import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Boolean, Integer, Float, DateTime, ForeignKey
from pgvector.sqlalchemy import Vector
from database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    passwordHash = Column(String(255), nullable=True)
    hashedPassword = Column(String(255), nullable=True)
    role = Column(String(50), default="MEMBER")
    orgId = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Bot(Base):
    __tablename__ = "bots"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    orgId = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False, default="Test Boat")
    domain = Column(String(255), nullable=False, default="appdeft.ai")
    publicKey = Column(String(64), unique=True, default=generate_uuid, index=True)
    template = Column(String(50), default="classic")
    accentColor = Column(String(20), default="#00c48c")
    greeting = Column(Text, default="Hi! How can I help?")
    suggestions = Column(Text, default="What do you offer?\nHow much does it cost?\nHow do I get in touch?")
    launcherPosition = Column(String(20), default="right")
    webhookUrl = Column(String(500), nullable=True)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class BotSource(Base):
    __tablename__ = "bot_sources"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    botId = Column(String(36), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False)
    kind = Column(String(50), default="PAGE")
    title = Column(String(255), nullable=False)
    url = Column(String(1000), nullable=True)
    tokenCount = Column(Integer, default=0)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    sourceId = Column(String(36), ForeignKey("bot_sources.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    botId = Column(String(36), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False)
    sessionId = Column(String(100), nullable=True)
    isTest = Column(Boolean, default=False)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    conversationId = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    unanswered = Column(Boolean, default=False)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Lead(Base):
    __tablename__ = "leads"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    botId = Column(String(36), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(100), nullable=True)
    note = Column(Text, nullable=True)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))