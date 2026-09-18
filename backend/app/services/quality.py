import re
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.db import models
from app.db.models import UserRole, DocumentStatus
from app.services.audit import log_audit_event
from app.utils.logger import logger

def analyze_knowledge_quality(
    db: Session,
    org_id: str,
    current_user: models.User,
    client_ip: str = "unknown"
) -> Dict[str, Any]:
    """
    Executes an automated Knowledge Health Audit across tenant documentation:
    1. Stale Documents (>180 days since approval/creation or expired CE/ISO certificates)
    2. Duplicate Content Clusters across distinct documents
    3. Conflicting Engineering Specifications (e.g. torque, voltage, temperature)
    4. Computes normalized Health Score (0-100) with remediation recommendations.
    """
    now = datetime.now(timezone.utc)
    stale_threshold = now - timedelta(days=180)
    expiring_soon_threshold = now + timedelta(days=60)

    # 1. Fetch organization documents
    doc_query = db.query(models.BotSource)
    if current_user.role != UserRole.PLATFORM_ADMIN:
        doc_query = doc_query.filter(
            or_(models.BotSource.orgId == org_id, models.BotSource.isUniversal == True)
        )
    documents = doc_query.all()
    doc_map = {d.id: d for d in documents}

    # 2. Check Stale & Expiring Documents
    stale_docs = []
    expiring_certs = []
    draft_pending_docs = []

    for d in documents:
        # Check draft status
        if d.status in [DocumentStatus.DRAFT, DocumentStatus.UNDER_REVIEW]:
            draft_pending_docs.append({
                "id": d.id,
                "title": d.title,
                "version": d.version or "v1.0",
                "status": d.status,
                "created_at": d.createdAt.isoformat() if d.createdAt else ""
            })

        # Check staleness: approved or created > 180 days ago without recent review
        check_date = d.approvedAt or d.createdAt
        if check_date and check_date.tzinfo is None:
            check_date = check_date.replace(tzinfo=timezone.utc)

        if check_date and check_date < stale_threshold:
            days_inactive = (now - check_date).days
            stale_docs.append({
                "id": d.id,
                "title": d.title,
                "version": d.version or "v1.0",
                "status": d.status,
                "last_active": check_date.isoformat(),
                "days_since_review": days_inactive,
                "reason": f"No SME verification or update in {days_inactive} days (>180 day SLA)"
            })

        # Check Certificate Expiration
        if d.expiryDate:
            exp_date = d.expiryDate
            if exp_date.tzinfo is None:
                exp_date = exp_date.replace(tzinfo=timezone.utc)

            if exp_date < now:
                expiring_certs.append({
                    "id": d.id,
                    "title": d.title,
                    "version": d.version or "v1.0",
                    "expiry_date": exp_date.isoformat(),
                    "state": "EXPIRED",
                    "days_remaining": (exp_date - now).days
                })
            elif exp_date <= expiring_soon_threshold:
                expiring_certs.append({
                    "id": d.id,
                    "title": d.title,
                    "version": d.version or "v1.0",
                    "expiry_date": exp_date.isoformat(),
                    "state": "EXPIRING_SOON",
                    "days_remaining": (exp_date - now).days
                })

    # 3. Detect Duplicate Chunks Across Documents
    doc_ids = list(doc_map.keys())
    duplicate_clusters = []
    chunk_hash_map = defaultdict(list)

    if doc_ids:
        chunks = db.query(models.DocumentChunk).filter(
            models.DocumentChunk.sourceId.in_(doc_ids)
        ).all()

        for chunk in chunks:
            if not chunk.content or len(chunk.content.strip()) < 80:
                continue
            # Normalize whitespace and create robust signature
            normalized = " ".join(chunk.content.lower().split()[:40])
            chunk_sig = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            chunk_hash_map[chunk_sig].append({
                "source_id": chunk.sourceId,
                "chunk_id": chunk.id,
                "preview": chunk.content[:140].strip()
            })

        # Find signatures that appear in more than 1 distinct source
        for sig, occurrences in chunk_hash_map.items():
            distinct_sources = {occ["source_id"] for occ in occurrences}
            if len(distinct_sources) > 1:
                source_titles = [doc_map[s_id].title for s_id in distinct_sources if s_id in doc_map]
                duplicate_clusters.append({
                    "signature": sig[:12],
                    "document_count": len(distinct_sources),
                    "documents": source_titles,
                    "sample_text": occurrences[0]["preview"] + "..."
                })

    # 4. Detect Conflicting Specifications
    # Regex patterns for high-impact industrial engineering specs
    SPEC_PATTERNS = {
        "torque": re.compile(r'(?:torque|tightening)\s*(?:of|is|:)?\s*([0-9.]+)\s*(nm|n-m|n\s*m|ft-lb)', re.IGNORECASE),
        "voltage": re.compile(r'(?:operating voltage|power supply|voltage)\s*(?:of|is|:)?\s*([0-9.]+)\s*(v|vdc|vac|volts)', re.IGNORECASE),
        "temperature": re.compile(r'(?:operating temperature|thermal range|temp)\s*(?:of|is|:)?\s*([-\d]+(?:\s*to\s*|\s*-\s*)[-\d]+|[0-9]+)\s*(?:°c|c|deg c)', re.IGNORECASE),
        "pressure": re.compile(r'(?:pressure|operating pressure)\s*(?:of|is|:)?\s*([0-9.]+)\s*(bar|psi|kpa)', re.IGNORECASE)
    }

    spec_findings = defaultdict(lambda: defaultdict(list))
    if doc_ids:
        for chunk in chunks:
            if not chunk.content:
                continue
            src = doc_map.get(chunk.sourceId)
            if not src:
                continue

            for spec_name, pat in SPEC_PATTERNS.items():
                matches = pat.findall(chunk.content)
                for m in matches:
                    val_str = m[0] if isinstance(m, tuple) else m
                    spec_findings[spec_name][src.title].append(val_str.strip())

    conflicting_specs = []
    for spec_name, doc_vals in spec_findings.items():
        if len(doc_vals) > 1:
            # Check if values differ across distinct documents
            val_to_docs = defaultdict(set)
            for d_title, vals in doc_vals.items():
                for v in vals:
                    val_to_docs[v].add(d_title)

            if len(val_to_docs) > 1:
                # Potential discrepancy
                conflict_summary = []
                for v, d_set in val_to_docs.items():
                    conflict_summary.append(f"Value '{v}' cited in {', '.join(d_set)}")

                conflicting_specs.append({
                    "specification_parameter": spec_name.capitalize(),
                    "divergent_values": list(val_to_docs.keys()),
                    "documents_involved": list(doc_vals.keys()),
                    "summary": f"Conflicting {spec_name} readings found across documents: {'; '.join(conflict_summary)}"
                })

    # 5. Calculate Health Score (0 - 100)
    score = 100

    # Penalties
    stale_penalty = min(len(stale_docs) * 5, 25)
    expired_penalty = min(sum(15 if c["state"] == "EXPIRED" else 5 for c in expiring_certs), 30)
    draft_penalty = min(len(draft_pending_docs) * 3, 15)
    dup_penalty = min(len(duplicate_clusters) * 4, 15)
    conflict_penalty = min(len(conflicting_specs) * 10, 20)

    total_penalty = stale_penalty + expired_penalty + draft_penalty + dup_penalty + conflict_penalty
    score = max(0, score - total_penalty)

    if score >= 85:
        rating = "EXCELLENT"
        color = "emerald"
    elif score >= 70:
        rating = "GOOD"
        color = "blue"
    elif score >= 50:
        rating = "WARNING"
        color = "amber"
    else:
        rating = "CRITICAL"
        color = "red"

    # 6. Actionable Recommendations
    recommendations = []
    if expiring_certs:
        recommendations.append(f"Renew {len(expiring_certs)} regulatory/compliance certificates nearing or past expiration.")
    if stale_docs:
        recommendations.append(f"Schedule SME recertification for {len(stale_docs)} stale documents inactive for >180 days.")
    if conflicting_specs:
        recommendations.append(f"Reconcile {len(conflicting_specs)} conflicting engineering specification parameters between active manuals.")
    if duplicate_clusters:
        recommendations.append(f"Consolidate {len(duplicate_clusters)} duplicate text sections to prevent search weight divergence.")
    if draft_pending_docs:
        recommendations.append(f"Review and approve {len(draft_pending_docs)} documents currently pending in the SME Approval Queue.")

    if not recommendations:
        recommendations.append("All technical documents comply with European ISO/EU MDR recertification freshness criteria.")

    return {
        "status": "success",
        "health_score": score,
        "rating": rating,
        "color": color,
        "metrics": {
            "total_documents": len(documents),
            "stale_documents_count": len(stale_docs),
            "expiring_certs_count": len(expiring_certs),
            "duplicate_clusters_count": len(duplicate_clusters),
            "conflicting_specs_count": len(conflicting_specs),
            "draft_pending_count": len(draft_pending_docs)
        },
        "stale_documents": stale_docs,
        "expiring_certificates": expiring_certs,
        "duplicate_clusters": duplicate_clusters[:10],
        "conflicting_specifications": conflicting_specs,
        "recommendations": recommendations,
        "scanned_at": now.isoformat()
    }

