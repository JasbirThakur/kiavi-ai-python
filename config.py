import os
from pathlib import Path

# Load .env file
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# Native PostgreSQL Connection URL (Port 5433 to avoid local host conflicts)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgrespassword2026@127.0.0.1:5433/kiavidb"
)
if "@db:5432" in DATABASE_URL:
    try:
        import socket
        socket.gethostbyname("db")
    except Exception:
        DATABASE_URL = DATABASE_URL.replace("@db:5432", "@127.0.0.1:5433")

# NVIDIA NIM Primary LLM & Embedding Engine
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
NVIDIA_EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nemotron-3-embed-1b")

# Groq Engine Fallback
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Brandfetch & Wikipedia Settings
BRANDFETCH_API_KEY = os.getenv("BRANDFETCH_API_KEY", "")

# JWT Authentication Settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "kiavi-secret-production-key-2026")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

# Local / Backup Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# RAG Thresholds & Token Budgets
RELEVANCE_FLOOR = 0.22
RESCUE_FLOOR = 0.15
TOP_K_CHUNKS = 6
MAX_TOKEN_BUDGET = 60000