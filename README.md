# ⚡ Kiavi IQ — Enterprise Autonomous AI Agent Platform

Kiavi IQ is a production-ready, decoupled conversational AI agent platform featuring:
- **Clean Decoupled Architecture**: Independent `frontend/`, `backend/`, and `nginx/` layers.
- **Deep Web Ingestion**: TLS fingerprint rotation bypassing Akamai and Cloudflare WAFs.
- **Multi-Format Parsing**: Automatic parsing of PDF, DOCX, CSV, TXT, and ZIP/VSIX packages.
- **Native pgvector Storage**: High-dimensional vector search on PostgreSQL 16.
- **18-Point Conversational Intelligence**: Warm, natural, human-like dialogue grounded in truth with zero hallucinations.
- **Live Human Handoff**: Real-time two-way chat takeover with email alerts.
- **Embeddable Chat Widget**: One-line `<script src="/w.js"></script>` integration.

---

## 📁 Repository Structure

```text
kiavi-ai-python/
│
├── frontend/                          # Dedicated Frontend Application
│   ├── public/                        # Production pages (dashboard, widget, live-chat, etc.)
│   ├── src/                           # Modular React + Vite application
│   │   ├── components/                # ChatWindow, MessageBubble, Sidebar, etc.
│   │   ├── pages/                     # Login, Dashboard, Chat, KnowledgeBase
│   │   ├── services/api.js            # Unified API service
│   │   └── hooks/useChat.js           # Custom chat streaming hook
│   ├── package.json
│   ├── vite.config.js
│   ├── Dockerfile
│   └── README.md
│
├── backend/                           # Dedicated FastAPI Backend Service
│   ├── app/
│   │   ├── api/routes/                # auth, bots, chat, knowledge, leads, insights, widget, voice
│   │   ├── config/settings.py         # Environment configuration
│   │   ├── core/                      # Security (JWT/bcrypt), dependencies, exceptions
│   │   ├── db/                        # Database connection & SQLAlchemy models (pgvector)
│   │   ├── prompts/                   # 18-point charter & query rewriting prompts
│   │   ├── schemas/                   # Pydantic validation schemas
│   │   ├── services/                  # LLM, RAG, scrapers, ingestion, embedding, memory
│   │   ├── utils/                     # Structured logger & helpers
│   │   ├── static/                    # Uploads, logos, and generated PDFs
│   │   └── main.py                    # FastAPI entrypoint
│   ├── tests/                         # Backend unit tests
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── nginx/
│   └── nginx.conf                     # Production reverse proxy configuration
│
├── docker-compose.yml                 # Orchestrates db, backend, and frontend
├── .env
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (Docker Compose)

### 1. Configure Environment:
```bash
cp .env.example .env
```

### 2. Start Services:
```bash
docker compose up -d --build
```

### 3. Access Services:
- 📊 **Dashboard / Frontend UI**: [http://localhost:3000](http://localhost:3000)
- 🔑 **Login**: [http://localhost:3000/login](http://localhost:3000/login)
- ⚡ **Backend API**: [http://localhost:8000](http://localhost:8000)
- 📖 **API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 🧪 **Client Test Site**: [http://localhost:3000/test-site](http://localhost:3000/test-site)

### 🔑 Default Credentials:
- **Email**: `sahil@kiavi.com`
- **Password**: `Test@1234`

---

## 🛠️ Local Development Without Docker

### Backend:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (React / Vite):
```bash
cd frontend
npm install
npm run dev
```
Accessible at [http://localhost:5173](http://localhost:5173).
