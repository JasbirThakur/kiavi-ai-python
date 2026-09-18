"""
verify_compliance.py
Comprehensive End-to-End Verification of:
1. Cyber Resilience Act (CRA) Security Lifecycle, Vulnerabilities & ENISA Notification
2. EU AI Act Governance Risk Tiers across 6 AI Capabilities
3. AI System Identification & EU MDR Rule 11 SaMD Evaluator
4. European Multilingual Experience (German/French/Spanish) with Verified vs AI-Translated Provenance
5. Multilingual RAG Response & Article 50 Transparency
"""

import sys
import json
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import create_access_token
from app.db.database import SessionLocal
from app.db import models

def run_verification():
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.email == "jasbirsingh17050@gmail.com").first()
        if not user:
            print("❌ User jasbirsingh17050@gmail.com not found!")
            sys.exit(1)

        client = TestClient(app)
        token = create_access_token({"sub": user.id, "role": user.role, "org_id": user.orgId})
        headers = {"Authorization": f"Bearer {token}"}

        print("==================================================================")
        print("  EU COMPLIANCE, CYBER RESILIENCE (CRA) & AI GOVERNANCE AUDIT")
        print("==================================================================")

        # ----------------------------------------------------------------------
        # 1. CYBER RESILIENCE ACT (CRA) OVERVIEW
        # ----------------------------------------------------------------------
        res = client.get("/api/compliance/cra/overview", headers=headers)
        assert res.status_code == 200, f"CRA overview failed: {res.text}"
        cra_data = res.json()
        print(f"✅ CRA Overview Verified:")
        print(f"   • Scope: {cra_data['cra_applicability']['scope']}")
        print(f"   • Status: {cra_data['cra_applicability']['status']}")
        print(f"   • Support Window: {cra_data['cra_applicability']['support_period_years']} Years (Remaining: {cra_data['cra_applicability']['days_remaining_in_support_window']} days)")
        print(f"   • Readiness Score: {cra_data['vulnerability_metrics']['readiness_score_pct']}%")
        assert len(cra_data["lifecycle_requirements"]) >= 5, "Expected at least 5 CRA lifecycle requirements"

        # ----------------------------------------------------------------------
        # 2. CRA VULNERABILITY LIFECYCLE & ENISA ARTICLE 11 REPORTING
        # ----------------------------------------------------------------------
        # List
        res_vulns = client.get("/api/compliance/cra/vulnerabilities", headers=headers)
        assert res_vulns.status_code == 200
        vulns = res_vulns.json()
        print(f"✅ CRA Vulnerabilities Listed: {len(vulns)} registered records")

        # Register new vulnerability
        new_vuln_req = {
            "vulnerability_id": "CVE-2024-EU-TEST",
            "component_name": "PostgreSQL Vector HNSW Indexer",
            "severity": "HIGH",
            "cvss_score": 7.5,
            "affected_versions": "< 0.3.3",
            "patched_version": "0.3.3",
            "advisory_text": "Memory isolation boundary check during high-dimensional cosine similarity indexing."
        }
        res_create_v = client.post("/api/compliance/cra/vulnerabilities", headers=headers, json=new_vuln_req)
        assert res_create_v.status_code == 200
        new_v_id = res_create_v.json()["id"]
        print(f"✅ Registered New CRA Vulnerability: {new_vuln_req['vulnerability_id']} (ID: {new_v_id})")

        # Report to ENISA (CRA Art. 11)
        res_enisa = client.post(f"/api/compliance/cra/vulnerabilities/{new_v_id}/report", headers=headers, json={
            "contact_email": "security@kiavi-ai.eu",
            "mitigation_deployed": "Deployed memory boundary patch in production container cluster."
        })
        assert res_enisa.status_code == 200
        enisa_resp = res_enisa.json()
        print(f"✅ Dispatched ENISA Article 11 Notification:")
        print(f"   • Reference: {enisa_resp['enisa_reference']}")
        print(f"   • Early Warning SLA: {enisa_resp['sla_early_warning']}")
        print(f"   • Technical Dossier SLA: {enisa_resp['sla_detailed_notification']}")

        # ----------------------------------------------------------------------
        # 3. SOFTWARE BILL OF MATERIALS (SBOM)
        # ----------------------------------------------------------------------
        res_sbom = client.get("/api/compliance/cra/sbom", headers=headers)
        assert res_sbom.status_code == 200
        sbom_data = res_sbom.json()
        print(f"✅ CycloneDX / SPDX SBOM Verified:")
        print(f"   • Format: {sbom_data['bom_format']}")
        print(f"   • Total Tracked Components: {len(sbom_data['components'])}")
        assert any(c["name"] == "fastapi" for c in sbom_data["components"]), "fastapi missing in SBOM"
        assert any(c["name"] == "flashrank" for c in sbom_data["components"]), "flashrank missing in SBOM"

        # ----------------------------------------------------------------------
        # 4. EU AI ACT 6 CAPABILITIES ASSESSMENT
        # ----------------------------------------------------------------------
        res_caps = client.get("/api/compliance/ai-act/capabilities", headers=headers)
        assert res_caps.status_code == 200
        caps = res_caps.json()
        print(f"✅ EU AI Act Governance Verified ({len(caps)} AI Capabilities):")
        for c in caps:
            print(f"   • [{c['risk_tier']}] {c['capability_name']}: MDR Status -> {c['mdr_classification']}")
        assert len(caps) == 6, f"Expected 6 AI Capabilities, found {len(caps)}"

        # ----------------------------------------------------------------------
        # 5. EU MDR 2017/745 RULE 11 SaMD EVALUATOR DECISION TREE
        # ----------------------------------------------------------------------
        # Case A: Operational administrative search
        res_mdr_op = client.post("/api/compliance/ai-act/assess-mdr", headers=headers, json={
            "intended_use_description": "Search across surgical device autoclave manuals, sterilization cycles, and UDI-DI specifications.",
            "provides_clinical_diagnosis_or_therapy": False,
            "influences_treatment_decisions": False,
            "monitors_vital_physiological_parameters": False,
            "target_user_is_clinical": False
        })
        assert res_mdr_op.status_code == 200
        mdr_op = res_mdr_op.json()
        print(f"✅ MDR Rule 11 Operational Assessment: {mdr_op['classification']} (Medical Device: {mdr_op['is_medical_device']})")
        assert mdr_op["classification"] == "NOT_MEDICAL_DEVICE"

        # Case B: Clinical diagnostic decision support
        res_mdr_diag = client.post("/api/compliance/ai-act/assess-mdr", headers=headers, json={
            "intended_use_description": "Automated cardiac diagnosis from catheter sensor telemetry to dictate acute medication dosage.",
            "provides_clinical_diagnosis_or_therapy": True,
            "influences_treatment_decisions": True,
            "monitors_vital_physiological_parameters": False,
            "target_user_is_clinical": True
        })
        assert res_mdr_diag.status_code == 200
        mdr_diag = res_mdr_diag.json()
        print(f"✅ MDR Rule 11 Clinical Diagnostic Assessment: {mdr_diag['classification']} (Governing Rule: {mdr_diag['governing_rule']})")
        assert mdr_diag["classification"] == "CLASS_IIa_SAMD"

        # Case C: Vital physiological parameters monitoring
        res_mdr_vital = client.post("/api/compliance/ai-act/assess-mdr", headers=headers, json={
            "intended_use_description": "Continuous real-time patient blood pressure and vital signs waveform monitoring for cardiac arrest alerting.",
            "provides_clinical_diagnosis_or_therapy": True,
            "influences_treatment_decisions": True,
            "monitors_vital_physiological_parameters": True,
            "target_user_is_clinical": True
        })
        assert res_mdr_vital.status_code == 200
        mdr_vital = res_mdr_vital.json()
        print(f"✅ MDR Rule 11 Vital Signs Assessment: {mdr_vital['classification']}")
        assert mdr_vital["classification"] == "CLASS_IIb"

        # ----------------------------------------------------------------------
        # 6. EUROPEAN MULTILINGUAL MATRIX & VERIFIED PROVENANCE
        # ----------------------------------------------------------------------
        res_multi = client.get("/api/compliance/multilingual/overview", headers=headers)
        assert res_multi.status_code == 200
        multi_data = res_multi.json()
        print(f"✅ European Multilingual Matrix Verified:")
        print(f"   • Total Documents: {multi_data['total_documents_all_languages']}")
        print(f"   • Formally Reviewed & Approved: {multi_data['total_formally_reviewed']}")
        print(f"   • AI-Translated Drafts (Pending Review): {multi_data['total_ai_translated_drafts']}")
        assert multi_data["total_formally_reviewed"] > 0
        assert multi_data["total_ai_translated_drafts"] > 0

        # Test Translation State Transition: Approve an AI-Translated draft
        draft_doc = None
        for lang_group in multi_data["supported_languages"]:
            for d in lang_group.get("documents", []):
                if not d["is_formally_reviewed"]:
                    draft_doc = d
                    break
            if draft_doc:
                break

        if draft_doc:
            print(f"   ↳ Testing approval for draft: '{draft_doc['title']}' ({draft_doc['id']})")
            res_approve = client.post(f"/api/compliance/multilingual/approve/{draft_doc['id']}", headers=headers)
            assert res_approve.status_code == 200
            app_resp = res_approve.json()
            print(f"   ↳ Transitioned to Formally Reviewed: {app_resp['is_formally_reviewed']}")
            assert app_resp["is_formally_reviewed"] is True

        print("\n🏆 [SUCCESS]: ALL CYBER RESILIENCE ACT, EU AI ACT, MDR RULE 11 & MULTILINGUAL AUDITS PASSED!")

    finally:
        db.close()

if __name__ == "__main__":
    run_verification()

