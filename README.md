# ⚡ Kiavi IQ — Enterprise Autonomous AI Agent Platform

Kiavi IQ is a production-ready, full-stack conversational AI agent platform. It features deep website web-scraping (supporting Next.js App Router RSC, React, Vue, WordPress), multi-modal OCR diagram extraction, vector chunking with PostgreSQL pgvector, hybrid lexical + vector retrieval, 3-tier resilient RAG streaming responses (NVIDIA Nemotron 70B & Groq), real-time human takeover, and instant 1-line embeddable web widgets.

---

## 🚀 Quick Start Guide (Run with 1 Command)

### 🐳 Method A: Run with Docker Compose (Recommended)

#### 1. Clone the repository:
```bash
git clone https://github.com/JasbirThakur/kiavi-ai-python.git
cd kiavi-ai-python
```

#### 2. Configure Environment Variables:
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional: Add your free trial `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com) or `GROQ_API_KEY` from [console.groq.com](https://console.groq.com) into `.env`).*

#### 3. Start the Full Stack (PostgreSQL 16 with pgvector + FastAPI Backend):
```bash
docker compose up --build
```
> **Note:** The backend mounts your local workspace (`volumes: - .:/app`) and runs with `--reload`, so code edits and uploads reflect in real time.

#### 4. Open the Application in Browser:
- 📊 **Admin Dashboard**: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
- 🔑 **Login / Register**: [http://localhost:8000/login](http://localhost:8000/login)
- 🧪 **Simulated Client Test Site**: [http://localhost:8000/test-site](http://localhost:8000/test-site)
- 📖 **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 🔑 Default Login Credentials

- **Email**: `sahil@kiavi.com`
- **Password**: `Test@1234`
*(Or click **Register** on the login screen to create a brand-new organization instantly).*

---

### 📦 Optional: Restore Seed Database (Pre-Loaded Bots & Data)
If you want to restore the pre-configured bots, metrics, and conversation history:
```bash
./restore_db.sh
```

---

### 🛠️ Method B: Run Locally with Python Virtual Environment

1. **Prerequisites**: Python 3.10+, PostgreSQL 16 with `pgvector` enabled (or run PostgreSQL via Docker: `docker compose up -d db`).

2. **Setup virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Configure Environment**:
```bash
cp .env.example .env
```

4. **Start the Backend Server**:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🌟 Core Features & Innovations

1. **🌐 Universal Multi-Framework Web Scraper**: Deep crawls any website (Next.js 13/14/15 App Router RSC, Nuxt, React, Vue, Angular, WordPress), parsing headers, footers, dropdown navigation, Schema.org microdata, physical office addresses, and phone numbers.
2. **🖼️ In-Chat Diagram & Visual Plate Rendering**: Extracts technical figures and charts from PDFs and images, rendering them directly as visual cards inside chat responses.
3. **📊 Tabular CSV Ingestion**: Transforms CSV tables into natural-language structured records, allowing vector search to query complex numerical rows.
4. **🧠 Hybrid Lexical + Vector Retrieval**: Combines 40-chunk dense vector cosine distance with direct SQL `ILIKE` keyword matching and typo normalization.
5. **🛡️ 3-Tier Zero-Failure LLM Cluster**:
   - **Tier 1**: NVIDIA NIM (`nvidia/llama-3.1-nemotron-70b-instruct`)
   - **Tier 2**: Groq LPU (`llama-3.3-70b-versatile`)
   - **Tier 3**: Direct Grounded Knowledge Extraction (100% zero-API-key fallback)
6. **🎨 Turnkey Appearance Studio**: Preset avatars, auto-fetching brand logos from target domains, custom color glow sliders, and live interactive preview.
7. **🔌 1-Line Embed Script Tag**:
   ```html
   <script src="http://localhost:8000/w.js" data-key="YOUR_BOT_PUBLIC_KEY"></script>
   ```
