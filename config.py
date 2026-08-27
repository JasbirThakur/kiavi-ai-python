import os
from pathlib import Path

# 🐘 Native PostgreSQL Connection URL
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgrespassword2026@127.0.0.1:5432/kiavidb"
)

# ⚡ LLM API Engines (Loaded dynamically from environment/.env)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
BRANDFETCH_API_KEY = os.getenv("BRANDFETCH_API_KEY", "")

# 🔐 JWT Authentication Settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "kiavi-secret-production-key-2026")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

# 🧠 Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# 🎯 RAG Thresholds & Token Budgets
RELEVANCE_FLOOR = 0.28
RESCUE_FLOOR = 0.18
TOP_K_CHUNKS = 6
MAX_TOKEN_BUDGET = 60000