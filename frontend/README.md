# Kiavi IQ - Frontend Application

Clean React / Vite frontend application and embeddable chat widget.

## Directory Layout

```text
frontend/
├── public/                 # Standalone production pages & widget assets
│   ├── dashboard.html      # Full-featured live dashboard
│   ├── login.html          # Authentication portal
│   ├── live_chat.html      # Two-way human agent takeover room
│   ├── widget.html         # Embeddable widget iframe UI
│   └── w.js                # One-line embed script
├── src/                    # Modular React + Vite UI
│   ├── components/         # ChatWindow, MessageBubble, ChatInput, Sidebar, etc.
│   ├── pages/              # Login, Dashboard, Chat, KnowledgeBase
│   ├── services/api.js     # Unified API client
│   ├── hooks/useChat.js    # Custom chat streaming hook
│   ├── css/                # Main, auth, dashboard, and chat styles
│   ├── App.jsx             # React application router
│   └── main.jsx            # Entry point
├── package.json            # Node.js dependencies
├── vite.config.js          # Vite build and proxy configuration
├── nginx.conf              # Reverse proxy configuration
└── Dockerfile              # Container deployment
```

## Running Locally

1. Install dependencies:
```bash
npm install
```

2. Run Vite development server:
```bash
npm run dev
```
Accessible at `http://localhost:5173`.

