from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Humne PostgreSQL hata kar SQLite laga diya hai. Ye direct file banayega, koi server nahi chahiye!
SQLALCHEMY_DATABASE_URL = "sqlite:///./kiavidb.db"

# SQLite ke liye ye setting zaroori hai
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()