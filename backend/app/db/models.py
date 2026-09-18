import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Boolean, Integer, Float, DateTime, ForeignKey
from pgvector.sqlalchemy import Vector
from app.db.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class UserRole:
    PLATFORM_ADMIN = "platform_admin"
    ORG_ADMIN = "org_admin"
    KNOWLEDGE_ADMIN = "knowledge_admin"
    SME_APPROVER = "sme_approver"
    TRAINER = "trainer"
    INTERNAL_EMPLOYEE = "internal_employee"
    DISTRIBUTOR_DEALER_PARTNER = "distributor_dealer_partner"
    B2B_CUSTOMER = "b2b_customer"
    AUDITOR = "auditor"
    CUSTOMER_PARTNER_ADMIN = "customer_partner_admin"
    OWNER = "OWNER" # Legacy workspace owner
    MEMBER = "MEMBER" # Legacy backward compatibility

    ALL_ROLES = [
        PLATFORM_ADMIN, ORG_ADMIN, KNOWLEDGE_ADMIN, SME_APPROVER,
        TRAINER, INTERNAL_EMPLOYEE, DISTRIBUTOR_DEALER_PARTNER,
        B2B_CUSTOMER, AUDITOR, CUSTOMER_PARTNER_ADMIN, OWNER, MEMBER
    ]

class DocumentStatus:
    DRAFT = "DRAFT"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    ARCHIVED = "ARCHIVED"

    ALL_STATUSES = [DRAFT, UNDER_REVIEW, APPROVED, REVIEW_REQUIRED, ARCHIVED]

# ----------------------------------------------------
# Multi-Tenancy & User Management
# ----------------------------------------------------
class Organization(Base):
    __tablename__ = "organizations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=True)
    domain = Column(String(255), nullable=True)
    industry = Column(String(100), default="MANUFACTURING") # MEDICAL, AUTOMOTIVE, HARDWARE, MANUFACTURING
    settingsJson = Column(Text, nullable=True)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    passwordHash = Column(String(255), nullable=True)
    hashedPassword = Column(String(255), nullable=True)
    role = Column(String(50), default=UserRole.INTERNAL_EMPLOYEE, index=True)
    department = Column(String(100), nullable=True) # e.g. Cardiology, Braking Systems, Regulatory
    region = Column(String(50), default="EU") # EU, DACH, NORDICS, NA
    partnerOrgId = Column(String(36), nullable=True) # For sub-tenant dealers
    mfaEnabled = Column(Boolean, default=False)
    orgId = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# ----------------------------------------------------
# Conversational Bot & Ingestion Sources
# ----------------------------------------------------
class Bot(Base):
    __tablename__ = "bots"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    orgId = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False, default="Test Boat")
    domain = Column(String(255), nullable=True, default="")
    publicKey = Column(String(64), unique=True, default=generate_uuid, index=True)
    template = Column(String(50), default="classic")
    accentColor = Column(String(20), default="#00c48c")
    greeting = Column(Text, default="Hi! How can I help?")
    suggestions = Column(Text, default="What do you offer?\nHow much does it cost?\nHow do I get in touch?")
    launcherPosition = Column(String(20), default="right")
    logoUrl = Column(String(500), nullable=True)
    webhookUrl = Column(String(500), nullable=True)
    supportEmail = Column(String(255), default="jasbirsingh17050@gmail.com", nullable=True)
    supportPhone = Column(String(50), nullable=True)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class BotSource(Base):
    __tablename__ = "bot_sources"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    botId = Column(String(36), ForeignKey("bots.id", ondelete="CASCADE"), nullable=True, index=True)
    isUniversal = Column(Boolean, default=False, nullable=False)
    orgId = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    kind = Column(String(50), default="PAGE")
    title = Column(String(255), nullable=False)
    url = Column(String(1000), nullable=True)
    tokenCount = Column(Integer, default=0)
    
    # Document Lifecycle & Governance
    status = Column(String(50), default=DocumentStatus.APPROVED, index=True)
    version = Column(String(20), default="v1.0")
    approvedBy = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approvedAt = Column(DateTime, nullable=True)
    expiryDate = Column(DateTime, nullable=True)
    permittedRoles = Column(Text, default='["internal_employee", "sme_approver", "knowledge_admin", "org_admin", "trainer", "distributor_dealer_partner", "b2b_customer", "auditor", "MEMBER"]')
    reviewIntervalDays = Column(Integer, default=180)
    
    # European Multilingual & Verification Governance
    language = Column(String(10), default="en", index=True) # en, de, fr, es, it, nl
    isAiTranslated = Column(Boolean, default=False)
    isFormallyReviewed = Column(Boolean, default=True)
    translationSourceDocId = Column(String(36), nullable=True)

    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    sourceId = Column(String(36), ForeignKey("bot_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False)
    source_url = Column(String(1000), nullable=True)
    document_type = Column(String(50), default="DOC", nullable=True)
    status = Column(String(50), default=DocumentStatus.APPROVED, index=True)
    permittedRoles = Column(Text, nullable=True)
    extracted_date = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=True)
    
    # Multilingual & Verification Tracking
    language = Column(String(10), default="en", nullable=True)
    isAiTranslated = Column(Boolean, default=False)
    isFormallyReviewed = Column(Boolean, default=True)

    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# ----------------------------------------------------
# Product Catalog & SKU 360°
# ----------------------------------------------------
class Product(Base):
    __tablename__ = "products"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    orgId = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    sku = Column(String(100), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False, index=True) # Door Hardware, Brakes, Surgical
    description = Column(Text, nullable=True)
    attributesJson = Column(Text, nullable=True) # JSON: dimensions, material, voltage, fireRating, torque
    certificationsJson = Column(Text, nullable=True) # JSON: ["CE Mark", "EN 1125", "ISO 13485"]
    mediaUrlsJson = Column(Text, nullable=True) # JSON: images, diagrams, CAD links
    udiDi = Column(String(100), nullable=True) # Basic UDI-DI (EU MDR / EUDAMED)
    imdsId = Column(String(100), nullable=True) # IMDS ID (Automotive)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updatedAt = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class ProductCompatibility(Base):
    __tablename__ = "product_compatibilities"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    orgId = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    productId = Column(String(36), ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    compatibleSku = Column(String(100), nullable=True)
    compatibleModel = Column(String(255), nullable=False) # e.g. "Honda Civic 2024", "Endoscope EM-200"
    compatibilityType = Column(String(50), default="OEM_FITMENT") # OEM_FITMENT, ACCESSORY, REPLACEMENT
    notes = Column(Text, nullable=True)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# ----------------------------------------------------
# Integrated Corporate LMS
# ----------------------------------------------------
class Course(Base):
    __tablename__ = "courses"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    orgId = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True)
    targetRolesJson = Column(Text, default='["internal_employee", "distributor_dealer_partner"]')
    isPublished = Column(Boolean, default=False)
    passingScore = Column(Integer, default=80) # Minimum % required to pass
    validityMonths = Column(Integer, default=24) # Certificate validity
    sourceDocId = Column(String(36), ForeignKey("bot_sources.id", ondelete="SET NULL"), nullable=True)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class CourseModule(Base):
    __tablename__ = "course_modules"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    courseId = Column(String(36), ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    orderIndex = Column(Integer, default=0)

class Lesson(Base):
    __tablename__ = "lessons"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    moduleId = Column(String(36), ForeignKey("course_modules.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    contentType = Column(String(50), default="DOC") # DOC, VIDEO, SLIDE
    mediaUrl = Column(String(500), nullable=True)
    durationSeconds = Column(Integer, default=300)
    orderIndex = Column(Integer, default=0)

class Quiz(Base):
    __tablename__ = "quizzes"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    courseId = Column(String(36), ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    passingScore = Column(Integer, default=80)

class QuizQuestion(Base):
    __tablename__ = "quiz_questions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    quizId = Column(String(36), ForeignKey("quizzes.id", ondelete="CASCADE"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    optionsJson = Column(Text, nullable=False) # JSON array of string options
    correctOptionIndex = Column(Integer, nullable=False)
    explanation = Column(Text, nullable=True)

class UserCourseProgress(Base):
    __tablename__ = "user_course_progress"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    userId = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    courseId = Column(String(36), ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    completedLessonsJson = Column(Text, default="[]") # JSON array of completed lesson IDs
    progressPercent = Column(Float, default=0.0)
    status = Column(String(50), default="IN_PROGRESS") # IN_PROGRESS, COMPLETED, EXPIRED
    quizScore = Column(Float, nullable=True)
    lastAccessedAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Certificate(Base):
    __tablename__ = "certificates"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    certificateNumber = Column(String(100), unique=True, nullable=False, index=True)
    userId = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    courseId = Column(String(36), ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    orgId = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    quizScore = Column(Float, nullable=False)
    verificationToken = Column(String(100), unique=True, nullable=False, index=True)
    issuedAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    expiresAt = Column(DateTime, nullable=True)

# ----------------------------------------------------
# Compliance & Audit Logging
# ----------------------------------------------------
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    orgId = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    userId = Column(String(36), nullable=True, index=True)
    userEmail = Column(String(255), nullable=True)
    userRole = Column(String(50), nullable=True)
    action = Column(String(100), nullable=False, index=True) # DOCUMENT_APPROVED, ROLE_ASSIGNED, etc.
    resourceType = Column(String(50), nullable=False) # document, user, course, product
    resourceId = Column(String(100), nullable=True)
    detailsJson = Column(Text, nullable=True)
    ipAddress = Column(String(50), nullable=True)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

# ----------------------------------------------------
# RAG Tracing & Chat History
# ----------------------------------------------------
class RAGTrace(Base):
    __tablename__ = "rag_traces"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    botId = Column(String(36), ForeignKey("bots.id", ondelete="CASCADE"), nullable=True)
    conversationId = Column(String(36), nullable=True)
    query = Column(Text, nullable=False)
    retrievedCandidatesCount = Column(Integer, default=0)
    topVectorScore = Column(Float, default=0.0)
    topRerankScore = Column(Float, default=0.0)
    rerankLatencyMs = Column(Float, default=0.0)
    totalLatencyMs = Column(Float, default=0.0)
    promptTokens = Column(Integer, default=0)
    contextTokens = Column(Integer, default=0)
    totalTokens = Column(Integer, default=0)
    guardrailStatus = Column(String(50), default="PASSED")
    candidatesJson = Column(Text, nullable=True)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    botId = Column(String(36), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False)
    sessionId = Column(String(100), nullable=True)
    isTest = Column(Boolean, default=False)
    isHandedOff = Column(Boolean, default=False)
    status = Column(String(50), default="AI_ACTIVE")
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

# ----------------------------------------------------
# EU Cyber Resilience Act (CRA) & Vulnerability Lifecycle
# ----------------------------------------------------
class CRAVulnerability(Base):
    __tablename__ = "cra_vulnerabilities"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    orgId = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    vulnerabilityId = Column(String(100), nullable=False, index=True)  # e.g. "CVE-2024-38816" or "KIAVI-SEC-004"
    componentName = Column(String(255), nullable=False)  # e.g. "FastAPI Framework", "pgvector extension", "Trafilatura"
    severity = Column(String(20), default="MEDIUM")  # CRITICAL, HIGH, MEDIUM, LOW
    cvssScore = Column(Float, default=5.0)  # 0.0 - 10.0
    status = Column(String(50), default="IDENTIFIED")  # IDENTIFIED, ASSESSING, MITIGATED, PATCHED, REPORTED_ENISA
    affectedVersions = Column(String(100), nullable=True)
    patchedVersion = Column(String(100), nullable=True)
    advisoryText = Column(Text, nullable=True)
    sbomComponent = Column(String(255), nullable=True)
    supportLifecycleUntil = Column(DateTime, nullable=True)
    enisaReported = Column(Boolean, default=False)
    enisaReportedAt = Column(DateTime, nullable=True)
    enisaReference = Column(String(100), nullable=True)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updatedAt = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

# ----------------------------------------------------
# EU AI Act Governance & Medical Device (MDR) Classification
# ----------------------------------------------------
class AICapabilityAssessment(Base):
    __tablename__ = "ai_capability_assessments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    orgId = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    capabilityName = Column(String(100), nullable=False, index=True)  # "AI Search", "AI Assistant", "AI Research", "AI Learning Assistant", "AI Agents", "AI-supported automation"
    intendedUse = Column(Text, nullable=False)
    riskTier = Column(String(50), default="LIMITED_RISK")  # MINIMAL_RISK, LIMITED_RISK, HIGH_RISK, PROHIBITED
    euAiActObligationsJson = Column(Text, nullable=True)  # JSON array of obligations (Art 50 transparency, human oversight, documentation)
    mdrClassification = Column(String(100), default="NOT_MEDICAL_DEVICE")  # NOT_MEDICAL_DEVICE, CLASS_I, CLASS_IIA_SAMD, CLASS_IIB, CLASS_III
    mdrRule11Justification = Column(Text, nullable=True)
    isCustomerFacing = Column(Boolean, default=True)
    humanOversightMeasures = Column(Text, nullable=True)
    watermarkingEnabled = Column(Boolean, default=True)
    assessedBy = Column(String(255), nullable=True)
    assessedAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updatedAt = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

