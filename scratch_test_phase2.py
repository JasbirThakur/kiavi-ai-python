import urllib.request
import json
import uuid

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("=== PHASE 2 INTEGRATION SUITE ===")

    # 0. Seed test admin user
    from app.db.database import SessionLocal
    from app.db import models
    from app.db.models import UserRole, DocumentStatus
    from app.core.security import hash_password

    db_seed = SessionLocal()
    try:
        admin_user = db_seed.query(models.User).filter(models.User.email == "admin@kiavi.ai").first()
        if not admin_user:
            # check or create org
            org = db_seed.query(models.Organization).first()
            if not org:
                org = models.Organization(name="Kiavi Industrial Systems", slug="kiavi-sys")
                db_seed.add(org)
                db_seed.commit()
                db_seed.refresh(org)
            admin_user = models.User(
                email="admin@kiavi.ai",
                passwordHash=hash_password("AdminSecurePassword123!"),
                name="Chief Trainer & Quality Lead",
                role=UserRole.PLATFORM_ADMIN,
                orgId=org.id
            )
            db_seed.add(admin_user)
            db_seed.commit()
            print("✅ 0. Seeded admin@kiavi.ai user")
    finally:
        db_seed.close()

    # 1. Login as Admin/Trainer
    login_data = json.dumps({"email": "admin@kiavi.ai", "password": "AdminSecurePassword123!"}).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/api/auth/login", data=login_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        auth_res = json.loads(resp.read().decode('utf-8'))
        token = auth_res.get("token") or auth_res.get("access_token")
        print(f"✅ 1. Authenticated as Admin: token acquired ({token[:15]}...)")


    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # 2. Test Knowledge Quality Report
    req_qual = urllib.request.Request(f"{BASE_URL}/api/knowledge/quality/report", headers=headers)
    with urllib.request.urlopen(req_qual) as resp:
        qual_res = json.loads(resp.read().decode('utf-8'))
        print(f"✅ 2. Knowledge Quality Report Retrieved: Health Score = {qual_res['health_score']}/100, Rating = {qual_res['rating']}")
        print(f"   Metrics: {qual_res['metrics']}")
        print(f"   Recommendations: {qual_res['recommendations'][:2]}")

    # 3. Test Knowledge Quality On-Demand Scan
    req_scan = urllib.request.Request(f"{BASE_URL}/api/knowledge/quality/scan", data=b"{}", headers=headers)
    with urllib.request.urlopen(req_scan) as resp:
        scan_res = json.loads(resp.read().decode('utf-8'))
        print(f"✅ 3. Knowledge Quality Scan Executed: Score = {scan_res['health_score']}/100")

    # 4. Prepare 2 sample documents with chunks for Multi-Doc Synthesis and Course Generation testing
    # We can fetch universal documents or insert two test sources
    from app.db.database import SessionLocal
    from app.db import models
    from app.db.models import DocumentStatus
    from app.services.embedding import get_embedding

    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.email == "admin@kiavi.ai").first()
        org_id = user.orgId

        # Create Doc A: Medical Catheter
        doc_a = models.BotSource(
            orgId=org_id,
            isUniversal=True,
            kind="DOC",
            title="Surgical Catheter Model SC-200 Technical Manual",
            status=DocumentStatus.APPROVED,
            version="v2.1",
            tokenCount=450
        )
        db.add(doc_a)
        db.flush()

        t_a1 = "The SC-200 Surgical Steerable Catheter operates within 15°C to 42°C with a maximum torque rating of 3.2 Nm. Complies with EU MDR 2017/745 Class IIa requirements. Autoclave sterilization limit is 50 cycles."
        t_a2 = "Installation Procedure: Ensure sterile sheath seal integrity. Apply tightening torque of 3.2 Nm on connector collar. Verify luer-lock engagement with DIN EN ISO 80369-7 standard."

        chunk_a1 = models.DocumentChunk(
            sourceId=doc_a.id,
            content=t_a1,
            embedding=get_embedding(t_a1),
            status=DocumentStatus.APPROVED
        )
        chunk_a2 = models.DocumentChunk(
            sourceId=doc_a.id,
            content=t_a2,
            embedding=get_embedding(t_a2),
            status=DocumentStatus.APPROVED
        )
        db.add_all([chunk_a1, chunk_a2])

        # Create Doc B: Medical Catheter Next Gen
        doc_b = models.BotSource(
            orgId=org_id,
            isUniversal=True,
            kind="DOC",
            title="Surgical Catheter Model SC-300 High-Flow Manual",
            status=DocumentStatus.APPROVED,
            version="v3.0",
            tokenCount=520
        )
        db.add(doc_b)
        db.flush()

        t_b1 = "The SC-300 High-Flow Catheter supports extended thermal range 10°C to 45°C with enhanced titanium braid supporting torque of 4.5 Nm. Certified under EU MDR 2017/745 Class IIb. Autoclave sterilization limit extended to 100 cycles."
        t_b2 = "Installation Procedure: Ensure sterile sheath seal integrity. Apply tightening torque of 4.5 Nm on reinforced connector collar. Backward compatible with SC-200 docking console."

        chunk_b1 = models.DocumentChunk(
            sourceId=doc_b.id,
            content=t_b1,
            embedding=get_embedding(t_b1),
            status=DocumentStatus.APPROVED
        )
        chunk_b2 = models.DocumentChunk(
            sourceId=doc_b.id,
            content=t_b2,
            embedding=get_embedding(t_b2),
            status=DocumentStatus.APPROVED
        )
        db.add_all([chunk_b1, chunk_b2])
        db.commit()

        doc_a_id = doc_a.id
        doc_b_id = doc_b.id
        print(f"✅ Prepared test manuals: Doc A ({doc_a_id}) and Doc B ({doc_b_id})")
    finally:
        db.close()

    # 5. Test Multi-Document Comparative Synthesis
    synth_payload = json.dumps({
        "source_ids": [doc_a_id, doc_b_id],
        "query": "Compare operating temperature, tightening torque, EU MDR classification, and sterilization cycle limits between SC-200 and SC-300."
    }).encode('utf-8')
    req_synth = urllib.request.Request(f"{BASE_URL}/api/chat/multi-doc-synthesis", data=synth_payload, headers=headers)
    with urllib.request.urlopen(req_synth) as resp:
        synth_res = json.loads(resp.read().decode('utf-8'))
        print(f"✅ 4. Multi-Doc Comparative Synthesis Succeeded:")
        print(f"   Compared {len(synth_res['documents_compared'])} documents.")
        print(f"   Synthesis Preview:\n{synth_res['synthesis'][:350]}...\n")

    # 6. Test AI-Assisted Course Generation from Document
    course_gen_payload = json.dumps({
        "source_id": doc_b_id
    }).encode('utf-8')
    req_course = urllib.request.Request(f"{BASE_URL}/api/lms/courses/generate-from-doc", data=course_gen_payload, headers=headers)
    with urllib.request.urlopen(req_course) as resp:
        course_res = json.loads(resp.read().decode('utf-8'))
        print(f"✅ 5. AI Course Generation Succeeded:")
        print(f"   Generated Course ID: {course_res.get('course_id')}")
        print(f"   Title: {course_res.get('title')}")
        print(f"   Modules: {len(course_res.get('modules', []))}, Quiz Questions: {len(course_res.get('quiz', {}).get('questions', []))}")

    # 7. Re-run Quality Report to verify conflict / duplicate detection on new docs
    req_qual2 = urllib.request.Request(f"{BASE_URL}/api/knowledge/quality/report", headers=headers)
    with urllib.request.urlopen(req_qual2) as resp:
        qual_res2 = json.loads(resp.read().decode('utf-8'))
        print(f"✅ 6. Re-evaluated Knowledge Quality:")
        print(f"   Duplicate clusters detected: {qual_res2['metrics']['duplicate_clusters_count']}")
        print(f"   Conflicting specs detected: {qual_res2['metrics']['conflicting_specs_count']}")
        if qual_res2['conflicting_specifications']:
            print(f"   Detected Spec Conflict: {qual_res2['conflicting_specifications'][0]['summary']}")

    print("\nALL PHASE 2 BACKEND TEST SUITES COMPLETED WITH 100% SUCCESS!")

if __name__ == "__main__":
    run_tests()
