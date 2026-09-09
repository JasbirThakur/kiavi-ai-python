from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config.settings import STATIC_DIR, CORS_ORIGINS
from app.db.database import init_db
from app.core.exceptions import (
    AppException,
    app_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    generic_exception_handler
)
from app.api.routes import (
    auth,
    bots,
    knowledge,
    chat,
    leads,
    insights,
    widget,
    voice
)
from app.utils.logger import logger

# 1. Initialize Database and verify pgvector extension
init_db()

# 2. Instantiate FastAPI App
app = FastAPI(
    title="Kiavi IQ - Grounded AI Agent Backend",
    description="High-performance grounded RAG backend with PostgreSQL & pgvector",
    version="2.0.0"
)

# 3. Exception Handlers for consistent JSON errors
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# 4. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Include API Routers
app.include_router(auth.router)
app.include_router(bots.router)
app.include_router(knowledge.router)
app.include_router(chat.router)
app.include_router(leads.router)
app.include_router(insights.router)
app.include_router(widget.router)
app.include_router(voice.router)

# 6. Static Uploads Mount (/static)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 7. Public Frontend Static Pages & Widget Integration
BASE_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = BASE_DIR / "public"
if not PUBLIC_DIR.exists():
    PUBLIC_DIR = BASE_DIR.parent / "frontend" / "public"

@app.get("/w.js")
def serve_widget_script():
    js_file = PUBLIC_DIR / "w.js"
    return FileResponse(js_file, media_type="application/javascript")

@app.get("/widget/{public_key}", response_class=HTMLResponse)
def serve_widget_ui(public_key: str):
    f = PUBLIC_DIR / "widget.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    return HTMLResponse("Widget UI Not Found", status_code=404)

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    f = PUBLIC_DIR / "dashboard.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    return HTMLResponse("Dashboard Not Found", status_code=404)

@app.get("/login", response_class=HTMLResponse)
@app.get("/signup", response_class=HTMLResponse)
def serve_login():
    f = PUBLIC_DIR / "login.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    return HTMLResponse("Login Not Found", status_code=404)

@app.get("/dashboard/account", response_class=HTMLResponse)
@app.get("/account", response_class=HTMLResponse)
def serve_account():
    f = PUBLIC_DIR / "account.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    return HTMLResponse("Account Not Found", status_code=404)

@app.get("/test-site", response_class=HTMLResponse)
def serve_test_site():
    f = PUBLIC_DIR / "test_site.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    return HTMLResponse("Test Site Not Found", status_code=404)

@app.get("/live-chat/{conv_id}", response_class=HTMLResponse)
@app.get("/visitor-chat/{conv_id}", response_class=HTMLResponse)
def serve_live_chat(conv_id: str):
    f = PUBLIC_DIR / "live_chat.html"
    if not f.exists():
        f = PUBLIC_DIR / "live-chat.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    return HTMLResponse("Live Chat Not Found", status_code=404)

# 8. Health Check API
@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "kiavi-iq-backend",
        "version": "2.0.0"
    }
