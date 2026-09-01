# ⚡ Kiavi IQ — Enterprise Autonomous AI Agent & Embeddable Web Intelligence

Kiavi IQ is a full-stack, enterprise-grade AI chatbot platform featuring deep website web-scraping, OCR multimodal diagram extraction, vector chunking with pgvector, real-time streaming RAG responses, live human agent takeover, voice AI, and instant embeddable web widgets.

---

## 🚀 Quick Start Guide (Run with 1 Command)

### Method A: Run with Docker Compose (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JasbirThakur/kiavi-ai-python.git
   cd kiavi-ai-python
   ```

2. **Start the full stack (Postgres with pgvector + FastAPI Backend)**:
   ```bash
   docker compose up --build
   ```

3. **Access the application**:
   - **Dashboard**: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
   - **Login / Register**: [http://localhost:8000/login](http://localhost:8000/login)
   - **Simulated Test Site**: [http://localhost:8000/test-site](http://localhost:8000/test-site)
   - **API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Method B: Run Locally with Python Virtual Environment

1. **Prerequisites**: Python 3.10+, PostgreSQL with pgvector (or run postgres via `docker compose up -d db`).

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

4. **Start the Server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

---

## 🔑 Default Credentials & Access

- **Email**: `sahil@kiavi.com`
- **Password**: `Test@1234`
- (Or click **Register** on `/login` to create a new organization account instantly)

---

## 🌟 Key Features

1. **🌐 Deep Web Scraping Engine**: Automatically crawls websites, subpages, footers, social media links (LinkedIn, Instagram, X), emails, and metadata.
2. **🖼️ In-Chat Diagram & Visual Plate Rendering**: Extracts technical diagrams and figures from PDFs/documents and displays them inline in the chat.
3. **🔴 Live Human Takeover**: Visitors can click "Human" to request a live specialist; Admins can join in real-time and chat directly from the Dashboard.
4. **🎙️ Voice AI**: Real-time Speech-to-Text (Mic) and Text-to-Speech (Speaker) voice responses.
5. **⚡ Proactive Welcome Bubble**: Greets website visitors after 5 seconds to boost lead conversions.
6. **📊 AI Insights & Clustering**: Analyzes visitor sentiment, satisfaction score, top inquiry topics, and unanswered knowledge gaps.
7. **🔌 1-Line Embed Script Tag**: Embeds into any site, Shopify, WordPress, or Webflow:
   ```html
   <script src="http://localhost:8000/w.js" data-key="YOUR_BOT_PUBLIC_KEY"></script>
   ```
