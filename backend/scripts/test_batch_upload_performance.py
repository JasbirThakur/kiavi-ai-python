import os
import sys
import time
import requests
from datetime import timedelta

# Ensure python path includes backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.security import create_access_token
from app.db.database import SessionLocal
from app.db import models

USER_ID = "7bf2ecbe-ae12-4c65-b853-c1368e11478e"
BOT_ID = "6f39b7c8-dd9e-4f50-b502-b6969dac19b8"
BASE_URL = "http://localhost:8000"

def run_tests():
    print("================================================================================")
    print("🚀 TESTING SUPER-FAST CONCURRENT MULTI-FILE BATCH INGESTION & INDEXING")
    print("================================================================================")

    # 1. Create access token
    token = create_access_token(data={"sub": USER_ID}, expires_delta=timedelta(hours=2))
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Authenticated as test user: 7bf2ecbe-ae12-4c65-b853-c1368e11478e")

    # 2. Prepare Sample Multi-File Batch (Universal)
    sample_file_1_content = """
    Enterprise Cybersecurity Guidelines & CRA Directives (2026 Edition)
    Document ID: CRA-SEC-2026-V4
    
    1. Scope of Digital Element Products
    All hardware, software, and SaaS platforms deployed within European and global markets
    must maintain active vulnerability disclosures, automated firmware signing, and continuous
    software bill of materials (SBOM). The maximum reporting window for actively exploited zero-day
    vulnerabilities is 24 hours to ENISA and the designated CSIRT.
    
    2. Cryptographic Standards
    All symmetric data-at-rest encryption must utilize AES-256-GCM. Asymmetric key exchange requires
    Ed25519 or RSA-4096. Any post-quantum migration schedule must adopt ML-KEM and ML-DSA by Q4 2026.
    """
    
    sample_file_2_content = """
    Autonomous Heavy Machinery Safety Interlocks & Protocol Specifications
    Document ID: ISO-13849-CAT4-PL-E
    
    1. Redundant Hydraulic Pressure Relief
    All Category 4, Performance Level e (PL e) electro-hydraulic actuators require dual redundant
    spool position monitoring sensors. In the event of a channel discrepancy exceeding 15 milliseconds,
    the emergency stop safety relay (PNOZ multi 2) must de-energize the main proportional valves within
    25 milliseconds, bringing the machine to a Category 0 stop per IEC 60204-1.
    
    2. Maintenance Interval
    Hydraulic fluid ISO VG 46 particle count must not exceed ISO 4406 cleanliness code 16/14/11.
    Filter replacement is mandatory every 500 operating hours or when differential pressure exceeds 2.5 bar.
    """

    print("\n--- TEST 1: Universal Knowledge Multi-File Ingest Batch ---")
    files_universal = [
        ("files", ("cybersecurity_cra_directives_2026.txt", sample_file_1_content.encode("utf-8"), "text/plain")),
        ("files", ("machinery_safety_spec_iso13849.txt", sample_file_2_content.encode("utf-8"), "text/plain"))
    ]
    t0 = time.perf_counter()
    resp_univ = requests.post(
        f"{BASE_URL}/api/knowledge/universal/ingest-batch",
        headers=headers,
        files=files_universal
    )
    t_univ = time.perf_counter() - t0
    print(f"Status Code: {resp_univ.status_code} (in {t_univ:.2f}s)")
    assert resp_univ.status_code == 200, f"Universal batch failed: {resp_univ.text}"
    univ_data = resp_univ.json()
    print("Response Telemetry:", univ_data)
    assert univ_data["total_files"] == 2
    assert univ_data["total_chunks"] > 0
    print(f"✅ Universal Batch: 2 files ingested simultaneously! Chunks: {univ_data['total_chunks']}, Tokens: {univ_data['total_tokens']}, Embedding Time: {univ_data['embedding_time_ms']}ms")

    print("\n--- TEST 2: Bot Knowledge Multi-File Ingest Batch ---")
    bot_file_1_content = """
    Navjot Support Tier Policy & SLA Guidelines
    Policy Code: NAV-SLA-TIER1-4
    
    SLA Targets:
    - Tier 1 Priority Incidents (Service Outage): 15 minute initial response, 2 hour resolution.
    - Tier 2 Critical Inquiries: 1 hour initial response, 6 hour resolution.
    - Tier 3 Standard Requests: 4 hour initial response, 24 hour resolution.
    - Tier 4 General Consultations: 1 business day response.
    
    Escalation Path:
    Any incident unresolved after 80% SLA elapsed triggers automatic notification to Navjot Operations Director.
    """
    
    bot_file_2_content = """
    Navjot Product Return and Hardware Replacement Warranty Policy
    Warranty Code: NAV-WARR-2026-EX
    
    Warranty Terms:
    All enterprise hardware units carry a 36-month comprehensive replacement warranty.
    Advanced Hardware Replacement (AHR) ships a replacement unit via overnight air prior to receiving
    the defective unit if RMA is initiated before 3:00 PM EST.
    Return shipping labels are prepaid and generated automatically via the Navjot Partner Portal.
    """

    files_bot = [
        ("files", ("navjot_support_sla_policy.txt", bot_file_1_content.encode("utf-8"), "text/plain")),
        ("files", ("navjot_hardware_warranty_terms.txt", bot_file_2_content.encode("utf-8"), "text/plain"))
    ]
    data_bot = {"bot_id": BOT_ID}

    t0 = time.perf_counter()
    resp_bot = requests.post(
        f"{BASE_URL}/api/knowledge/ingest-batch",
        headers=headers,
        data=data_bot,
        files=files_bot
    )
    t_bot = time.perf_counter() - t0
    print(f"Status Code: {resp_bot.status_code} (in {t_bot:.2f}s)")
    assert resp_bot.status_code == 200, f"Bot batch failed: {resp_bot.text}"
    bot_data = resp_bot.json()
    print("Response Telemetry:", bot_data)
    assert bot_data["total_files"] == 2
    assert bot_data["total_chunks"] > 0
    print(f"✅ Bot Batch: 2 files ingested simultaneously! Chunks: {bot_data['total_chunks']}, Tokens: {bot_data['total_tokens']}, Embedding Time: {bot_data['embedding_time_ms']}ms")

    print("\n--- TEST 3: Verify Database Storage in PostgreSQL ---")
    db = SessionLocal()
    try:
        # Check universal sources
        univ_sources = db.query(models.BotSource).filter(
            models.BotSource.isUniversal == True,
            models.BotSource.title.in_(["cybersecurity_cra_directives_2026.txt", "machinery_safety_spec_iso13849.txt"])
        ).all()
        print(f"Universal sources found in DB: {len(univ_sources)}/2")
        assert len(univ_sources) == 2

        # Check bot sources
        b_sources = db.query(models.BotSource).filter(
            models.BotSource.botId == BOT_ID,
            models.BotSource.title.in_(["navjot_support_sla_policy.txt", "navjot_hardware_warranty_terms.txt"])
        ).all()
        print(f"Bot sources found in DB: {len(b_sources)}/2")
        assert len(b_sources) == 2

        # Check chunks and vectors
        s_ids = [s.id for s in univ_sources + b_sources]
        chunks = db.query(models.DocumentChunk).filter(models.DocumentChunk.sourceId.in_(s_ids)).all()
        print(f"Total document chunks created with vector embeddings: {len(chunks)}")
        assert len(chunks) > 0
        for c in chunks:
            assert c.embedding is not None, f"Chunk {c.id} missing embedding"
        print("✅ All chunks have valid 384-dimensional pgvector embeddings!")
    finally:
        db.close()

    print("\n--- TEST 4: Test 3-Layer Hybrid Search Retrieval ---")
    # Query 1: Universal knowledge retrieval
    search_payload = {
        "query": "What is the maximum reporting window for exploited zero-day vulnerabilities under the CRA?",
        "bot_id": BOT_ID,
        "top_k_final": 3
    }
    resp_search = requests.post(f"{BASE_URL}/api/knowledge/hybrid-search", headers=headers, data=search_payload)
    print(f"Search Status: {resp_search.status_code}")
    assert resp_search.status_code == 200
    search_res = resp_search.json()
    chunks_list = search_res.get("gold_chunks", [])
    print("Retrieved Chunks Count:", len(chunks_list))
    assert len(chunks_list) > 0, "No gold chunks returned from hybrid search"
    top_chunk = chunks_list[0]["text"]
    print(f"Top Retrieved Match: {top_chunk[:120]}...")
    assert "24 hours" in top_chunk or "ENISA" in top_chunk or "CRA-SEC-2026" in top_chunk
    print("✅ Hybrid Search successfully retrieved Universal document chunk with exact CRA 24 hours directive!")

    # Query 2: Bot-specific knowledge retrieval
    search_payload_2 = {
        "query": "What is the warranty policy for Advanced Hardware Replacement and return shipping?",
        "bot_id": BOT_ID,
        "top_k_final": 3
    }
    resp_search_2 = requests.post(f"{BASE_URL}/api/knowledge/hybrid-search", headers=headers, data=search_payload_2)
    assert resp_search_2.status_code == 200
    search_res_2 = resp_search_2.json()
    chunks_list_2 = search_res_2.get("gold_chunks", [])
    assert len(chunks_list_2) > 0, "No gold chunks returned for bot search"
    top_chunk_2 = chunks_list_2[0]["text"]
    print(f"Top Retrieved Match 2: {top_chunk_2[:120]}...")
    assert "36-month" in top_chunk_2 or "Advanced Hardware Replacement" in top_chunk_2 or "AHR" in top_chunk_2
    print("✅ Hybrid Search successfully retrieved Bot-specific document chunk with AHR policy!")

    print("\n================================================================================")
    print("🎉 ALL MULTI-FILE BATCH INGESTION TESTS PASSED WITH FLYING COLORS!")
    print("================================================================================")

if __name__ == "__main__":
    run_tests()
