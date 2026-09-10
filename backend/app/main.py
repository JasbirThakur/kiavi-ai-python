from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
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

# 7. Root API Information Endpoint
@app.get("/", tags=["System"])
def root_endpoint():
    return {
        "service": "Kiavi IQ - Grounded AI Backend API",
        "version": "2.0.0",
        "status": "online",
        "frontend_dashboard": "http://localhost:3000/dashboard",
        "documentation": "/docs",
        "health": "/api/health"
    }

# Convenience browser redirects to Frontend UI (Port 3000)
@app.get("/dashboard", include_in_schema=False)
@app.get("/login", include_in_schema=False)
@app.get("/signup", include_in_schema=False)
@app.get("/account", include_in_schema=False)
def redirect_to_frontend():
    return RedirectResponse(url="http://localhost:3000/dashboard", status_code=307)

# 8. Health Check API
@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "kiavi-iq-backend",
        "version": "2.0.0"
    }
