# Kiavi IQ - Frontend Application (Standalone)

Clean React + Vite frontend application, web dashboard, and embeddable chat widget.

---

## Directory Layout

```text
frontend/
├── .env                     # Local frontend environment (PORT=3000, VITE_API_BASE_URL)
├── .env.example             # Environment template
├── .dockerignore            # Ignores node_modules, dist, .env from docker build
├── .gitignore               # Ignores node_modules, dist, .env from git
├── .pre-commit-config.yaml  # Frontend linting & secret leak prevention
├── Dockerfile               # Production Nginx container (Port 80 -> Host 3000)
├── docker-compose.yml       # Standalone Docker Compose (Nginx Web Server Port 3000)
├── nginx.conf               # Reverse proxy configuration with SSE streaming
├── package.json             # Node.js dependencies
├── vite.config.js           # Vite configuration with backend proxy
├── public/                  # Standalone production pages & widget assets
│   ├── dashboard.html       # Full-featured live dashboard
│   ├── login.html           # Authentication portal
│   ├── signup.html          # User registration portal
│   ├── live-chat.html       # Two-way human agent takeover room
│   ├── widget.html          # Embeddable widget iframe UI
│   └── w.js                 # One-line embed script
└── src/                     # Modular React + Vite UI
    ├── components/          # ChatWindow, MessageBubble, ChatInput, Sidebar, etc.
    ├── pages/               # Login, Dashboard, Chat, KnowledgeBase
    ├── services/api.js      # Unified API client with JWT & SSE streaming
    ├── hooks/useChat.js     # Custom chat streaming hook
    ├── css/                 # Main, auth, dashboard, and chat styles
    ├── App.jsx              # React application router
    └── main.jsx             # Entry point
```

---

## Ports & Environment

- **Frontend Port**: `3000` (`http://localhost:3000`)
- **Backend API URL**: `http://localhost:8000/api`
- **Configuration File**: `.env` (copied from `.env.example`)

```env
PORT=3000
VITE_PORT=3000
VITE_API_BASE_URL=http://localhost:8000/api
```

---

## Quick Start with Docker (Recommended)

To run the frontend Nginx web server on port 3000 with a single command:

```bash
# Inside the frontend/ directory
docker compose up -d
```
Access the application at `http://localhost:3000`.

---

## Running Locally on Node / Vite

### Option A: React Development Server (Vite)
```bash
npm install
npm run dev
```
Open your browser at `http://localhost:3000`.

### Option B: Direct Static Web Pages (public/)
```bash
npx serve public -l 3000
# Or using Python:
python3 -m http.server 3000 --directory public
```

---

## Pre-Commit Hooks

To install and run standalone pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
