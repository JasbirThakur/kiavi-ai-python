# Kiavi IQ - Backend Service (Standalone)

High-performance grounded AI chatbot backend built with **FastAPI**, **PostgreSQL 16**, **pgvector**, and **NVIDIA NIM / Groq**.

---

## Directory Layout

```text
backend/
├── .env                     # Local environment secrets (PORT=8000, DATABASE_URL, etc.)
├── .env.example             # Environment template
├── .dockerignore            # Ignores venv, pycache, .env from docker build
├── .gitignore               # Ignores venv, pycache, .env from git
├── .pre-commit-config.yaml  # Python syntax & secret leak checks
├── Dockerfile               # Backend API image configuration (Port 8000)
├── docker-compose.yml       # Standalone Docker Compose (PostgreSQL 5433 + Backend 8000)
├── backup.sql               # Certified vector database dump
├── restore_db.sh            # One-line database restore script
├── requirements.txt         # Python dependencies
├── app/
│   ├── api/routes/          # REST endpoints (auth, chat, knowledge, bots, leads, insights, widget, voice)
│   ├── config/              # settings & environment loader
│   ├── core/                # security, exceptions
│   ├── db/                  # SQLAlchemy models & database session
│   ├── prompts/             # System prompts & query rewrite templates
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # RAG, LLM, embedding, web scrapers
│   ├── static/              # uploads, logos, extracted diagrams
│   ├── utils/               # logger & helpers
│   └── main.py              # Pure API entrypoint (Root JSON + OpenAPI /docs)
└── tests/                   # Pytest unit and integration tests
```

---

## Ports & Endpoints

- **API Base Port**: `8000` (`http://localhost:8000`)
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **Health Check Endpoint**: `http://localhost:8000/api/health`
- **PostgreSQL Database Port**: `5433` (Host) -> `5432` (Container)

---

## Quick Start with Docker (Recommended)

To start both PostgreSQL (`pgvector`) and FastAPI Backend with a single command:

```bash
# 1. Inside the backend/ directory
cp .env.example .env

# 2. Start PostgreSQL and Backend services
docker compose up -d

# 3. Restore the vector database dump
./restore_db.sh
```

---

## Running Standalone on Local Python

### 1. Prerequisites
- Python 3.11 or 3.12
- PostgreSQL 16 with `pgvector` extension running on port 5433

### 2. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure `.env`
```bash
cp .env.example .env
# Edit .env with your database URL and API keys
```

### 4. Start Development Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Pre-Commit Hooks

To install and run standalone pre-commit checks:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
