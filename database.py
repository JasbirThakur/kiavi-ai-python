import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DATABASE_URL

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
            print("✅ pgvector extension verified.")
            # Run schema migrations for Universal Knowledge Base
            try:
                conn.execute(text("ALTER TABLE bot_sources ALTER COLUMN \"botId\" DROP NOT NULL;"))
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"isUniversal\" BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS \"orgId\" VARCHAR(36) REFERENCES organizations(id) ON DELETE CASCADE;"))
                conn.execute(text("UPDATE bot_sources SET \"orgId\" = bots.\"orgId\" FROM bots WHERE bot_sources.\"botId\" = bots.id AND bot_sources.\"orgId\" IS NULL;"))
            except Exception as mig_err:
                print(f"⚠️ Migration note: {mig_err}")
            print("✅ Universal Knowledge schema verified.")

        import models
        models.Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully.")
    except Exception as e:
        print(f"⚠️ Error initializing database: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
