# Kiavi IQ - Backend Service

High-performance grounded AI chatbot backend built with **FastAPI**, **PostgreSQL 16**, **pgvector**, and **NVIDIA NIM / Groq**.

## Architecture

```text
backend/
├── app/
│   ├── api/
│   │   └── routes/         # auth, bots, chat, knowledge, leads, insights, widget, voice
│   ├── config/             # settings & environment variables
│   ├── core/               # security, dependencies, exception handlers
│   ├── db/                 # database connection & SQLAlchemy ORM models
│   ├── prompts/            # 18-point charter & query rewrite prompt templates
│   ├── schemas/            # Pydantic schemas
│   ├── services/           # LLM, RAG, scrapers, ingestion, embedding, memory
│   ├── utils/              # structured logger & helpers
│   └── main.py             # FastAPI entrypoint
├── tests/                  # Unit and integration tests
├── Dockerfile              # Docker build configuration
└── requirements.txt        # Python package dependencies
```

## Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
python -m pytest tests/
```

3. Start development server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

