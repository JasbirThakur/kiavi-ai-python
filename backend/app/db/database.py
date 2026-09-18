from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config.settings import DATABASE_URL
from app.utils.logger import logger

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    """Ensures PostgreSQL vector extension is enabled and all tables are cleanly created"""
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            logger.info("✅ pgvector extension verified.")
            
            # Run schema migrations for Universal Knowledge Base and Bot features
            try:
                conn.execute(text("ALTER TABLE bot_sources ALTER COLUMN \"botId\" DROP NOT NULL;"))
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"isUniversal\" BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"orgId\" VARCHAR(36) REFERENCES organizations(id) ON DELETE CASCADE;"))
                conn.execute(text("UPDATE bot_sources SET \"orgId\" = bots.\"orgId\" FROM bots WHERE bot_sources.\"botId\" = bots.id AND bot_sources.\"orgId\" IS NULL;"))
                conn.execute(text("ALTER TABLE bots ADD COLUMN IF NOT EXISTS \"webhookUrl\" VARCHAR(500);"))
                conn.execute(text("ALTER TABLE bots ADD COLUMN IF NOT EXISTS \"supportEmail\" VARCHAR(255);"))
                conn.execute(text("ALTER TABLE bots ADD COLUMN IF NOT EXISTS \"supportPhone\" VARCHAR(50);"))
                
                # Enterprise Schema Migrations:
                # Organizations
                conn.execute(text("ALTER TABLE organizations ADD COLUMN IF NOT EXISTS \"slug\" VARCHAR(100);"))
                conn.execute(text("ALTER TABLE organizations ADD COLUMN IF NOT EXISTS \"domain\" VARCHAR(255);"))
                conn.execute(text("ALTER TABLE organizations ADD COLUMN IF NOT EXISTS \"industry\" VARCHAR(100) DEFAULT 'MANUFACTURING';"))
                conn.execute(text("ALTER TABLE organizations ADD COLUMN IF NOT EXISTS \"settingsJson\" TEXT;"))

                # Users
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS \"department\" VARCHAR(100);"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS \"region\" VARCHAR(50) DEFAULT 'EU';"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS \"partnerOrgId\" VARCHAR(36);"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS \"mfaEnabled\" BOOLEAN DEFAULT FALSE;"))

                # BotSources (Document Lifecycle)
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"status\" VARCHAR(50) DEFAULT 'APPROVED';"))
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"version\" VARCHAR(20) DEFAULT 'v1.0';"))
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"approvedBy\" VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL;"))
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"approvedAt\" TIMESTAMP;"))
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"expiryDate\" TIMESTAMP;"))
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"permittedRoles\" TEXT DEFAULT '[\"internal_employee\", \"sme_approver\", \"knowledge_admin\", \"org_admin\", \"trainer\", \"distributor_dealer_partner\", \"b2b_customer\", \"auditor\", \"MEMBER\"]';"))
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"reviewIntervalDays\" INTEGER DEFAULT 180;"))

                # DocumentChunks
                conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS \"status\" VARCHAR(50) DEFAULT 'APPROVED';"))
                conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS \"permittedRoles\" TEXT;"))

                # European Multilingual & Governance
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"language\" VARCHAR(10) DEFAULT 'en';"))
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"isAiTranslated\" BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"isFormallyReviewed\" BOOLEAN DEFAULT TRUE;"))
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"translationSourceDocId\" VARCHAR(36);"))

                conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS \"language\" VARCHAR(10) DEFAULT 'en';"))
                conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS \"isAiTranslated\" BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS \"isFormallyReviewed\" BOOLEAN DEFAULT TRUE;"))
            except Exception as mig_err:
                logger.warning(f"⚠️ Migration note: {mig_err}")
            logger.info("✅ Universal Knowledge, Bot, Enterprise & EU Governance schema verified.")

        from app.db import models
        models.Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully.")
    except Exception as e:
        logger.error(f"⚠️ Error initializing database: {e}")

def get_db():
    """FastAPI database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

