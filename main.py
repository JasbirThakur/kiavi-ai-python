from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path

import database
import models
from routers import auth, bots, knowledge, chat, leads, insights, widget, voice

# 1. Initialize DB and create tables safely
database.init_db()

app = FastAPI(title="Kiavi IQ - Grounded AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(bots.router)
app.include_router(knowledge.router)
app.include_router(chat.router)
app.include_router(leads.router)
app.include_router(insights.router)
app.include_router(widget.router)
app.include_router(voice.router)

from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/w.js")
def serve_widget_script():
    js_file = BASE_DIR / "w.js"
    if not js_file.exists():
        js_file = TEMPLATE_DIR / "w.js"
    return FileResponse(js_file, media_type="application/javascript")

@app.get("/widget/{public_key}", response_class=HTMLResponse)
def serve_widget_ui(public_key: str):
    return (TEMPLATE_DIR / "widget.html").read_text(encoding="utf-8")

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    return (TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")

@app.get("/login", response_class=HTMLResponse)
def serve_login():
    return (TEMPLATE_DIR / "login.html").read_text(encoding="utf-8")

@app.get("/dashboard/account", response_class=HTMLResponse)
def serve_account():
    return (TEMPLATE_DIR / "account.html").read_text(encoding="utf-8")

@app.get("/test-site", response_class=HTMLResponse)
def serve_test_site():
    return (TEMPLATE_DIR / "test_site.html").read_text(encoding="utf-8")
