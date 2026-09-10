# ⚡ Kiavi IQ — Enterprise Autonomous AI Agent Platform

Kiavi IQ is an enterprise-grade, modular conversational AI platform featuring two completely independent services:
- **Backend API**: High-performance FastAPI service with PostgreSQL 16 (`pgvector`), NVIDIA NIM & Groq LLMs.
- **Frontend UI**: Modular React + Vite application, Nginx production server, and embeddable live chat widget.

---

## 📁 Clean Repository Layout

```text
kiavi-ai-python/
│
├── backend/                  # 🐍 Standalone Backend API Repository
│   ├── .env                  # Backend credentials (PORT=8000, DATABASE_URL, NVIDIA_API_KEY)
│   ├── .env.example          # Template environment file
│   ├── .dockerignore         # Docker ignore rules
│   ├── .gitignore            # Python gitignore rules
│   ├── .pre-commit-config.yaml # Python linting & secret leak prevention
│   ├── Dockerfile            # Python 3.12 + PyTorch + FastAPI image
│   ├── docker-compose.yml    # Standalone Compose (PostgreSQL 5433 + Backend API 8000)
│   ├── backup.sql            # Certified vector database backup
│   ├── restore_db.sh         # One-line database restore script
│   ├── requirements.txt      # Python dependencies
│   ├── README.md             # Backend setup & API documentation
│   ├── app/                  # FastAPI application code
│   └── tests/                # Pytest unit and integration tests
│
├── frontend/                 # ⚛️ Standalone Frontend UI Repository
│   ├── .env                  # Frontend configuration (PORT=3000, VITE_API_BASE_URL)
│   ├── .env.example          # Template environment file
│   ├── .dockerignore         # Docker ignore rules
│   ├── .gitignore            # Node gitignore rules
│   ├── .pre-commit-config.yaml # Frontend linting & secret leak prevention
│   ├── Dockerfile            # Nginx production web server image
│   ├── docker-compose.yml    # Standalone Compose (Nginx Web Server Port 3000)
│   ├── nginx.conf            # Reverse proxy configuration with SSE streaming
│   ├── package.json          # Node.js dependencies
│   ├── vite.config.js        # Vite build & proxy configuration
│   ├── README.md             # Frontend setup & development guide
│   ├── public/               # Production HTML pages & widget assets
│   └── src/                  # React UI components & services
│
├── docker-compose.yml        # 🚀 Monorepo orchestrator (runs DB + Backend + Frontend together)
├── .gitignore                # 🛡️ Global gitignore rules
└── README.md                 # 📖 Main documentation & repository export guide
```

---

## 🚀 Running the Full Stack Together (Local Monorepo)

To run the full stack (PostgreSQL + Backend + Frontend) simultaneously from this repository:

```bash
docker compose up -d
```

- **Frontend UI**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000` (Docs: `http://localhost:8000/docs`)
- **Database**: `localhost:5433`

---

## 📦 Exporting to Separate Company Repositories

Each folder is 100% self-contained and ready to be pushed to its own git repository:

### 1. Export Frontend
```bash
cp -r frontend /home/devuser/Desktop/Kiavi-frontend
cd /home/devuser/Desktop/Kiavi-frontend
git init
git add .
git commit -m "feat: initial frontend repository setup"
git remote add origin <company-frontend-repo-url>
git push -u origin main
```

### 2. Export Backend
```bash
cp -r backend /home/devuser/Desktop/Kiavi-backend
cd /home/devuser/Desktop/Kiavi-backend
git init
git add .
git commit -m "feat: initial backend repository setup"
git remote add origin <company-backend-repo-url>
git push -u origin main
```
