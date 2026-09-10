import os
from pathlib import Path
from typing import List

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = APP_DIR / "prompts"
STATIC_DIR = APP_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Load .env file
env_path = BASE_DIR / ".env"
if not env_path.exists():
    env_path = BASE_DIR.parent / ".env"

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

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

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
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.2-11b-vision-instruct")
NVIDIA_EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nemotron-3-embed-1b")

# Groq Engine Fallback
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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
TOP_K_CHUNKS = 8
MAX_TOKEN_BUDGET = 60000

# CORS Allowed Origins
def get_cors_origins() -> List[str]:
    origins_str = os.getenv("CORS_ORIGINS", "")
    if origins_str:
        return [origin.strip() for origin in origins_str.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ]

CORS_ORIGINS = get_cors_origins()

