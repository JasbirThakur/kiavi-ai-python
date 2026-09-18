"""
hybrid_search.py
Three-Layer Hybrid Search Pipeline with Reciprocal Rank Fusion (RRF) & FlashRank Reranker.
Architecture:
User Query ──► [ Layer 1: Dense Vector Search (Meaning) ] ──┐
           ──► [ Layer 2: PostgreSQL Full-Text (Exact IDs) ] ──┼──► [ Layer 3: Mathematical RRF Fusion ] ──► [ Layer 4: FlashRank Cross-Encoder ] ──► Gold Chunks
"""

import re
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models
from app.services.embedding import get_embedding
from app.services.reranker import rerank_chunks
from app.utils.logger import logger

def three_layer_hybrid_search(
    query: str,
    bot_id: Optional[str] = None,
    org_id: Optional[str] = None,
    allowed_source_ids: Optional[List[str]] = None,
    valid_industries: Optional[set] = None,
    matched_industry: Optional[str] = None,
    top_k_vector: int = 25,
    top_k_lexical: int = 25,
    top_k_final: int = 4,
    rrf_k: int = 60,
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """
    Executes Three-Layer Hybrid Search:
    1. Layer 1: Dense Vector Cosine Similarity Search in pgvector.
    2. Layer 2: PostgreSQL Full-Text Search (tsvector / websearch_to_tsquery + exact code ILIKE).
    3. Layer 3: Reciprocal Rank Fusion (RRF) combining vector and lexical rankings.
    4. Layer 4: FlashRank Deep Cross-Encoder Reranker for absolute precision.
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    start_time = time.time()
    clean_q = query.strip()

    # Scope condition
    scope_conditions = ["c.status = 'APPROVED'"]
    params: Dict[str, Any] = {"query_text": clean_q}

    if allowed_source_ids is not None:
        if not allowed_source_ids:
            # Empty list means no sources available
            if close_db:
                db.close()
            return {
                "query": clean_q,
                "total_vector_candidates": 0,
                "total_lexical_candidates": 0,
                "total_fused_candidates": 0,
                "gold_chunks": [],
                "top_rerank_score": 0.0,
                "total_latency_ms": 0.0
            }
        scope_conditions.append("s.id = ANY(:allowed_source_ids)")
        params["allowed_source_ids"] = [str(x) for x in allowed_source_ids]
    elif bot_id:
        scope_conditions.append("(s.\"botId\" = :bot_id OR s.\"isUniversal\" = TRUE)")
        params["bot_id"] = str(bot_id)
    elif org_id:
        scope_conditions.append("(s.\"orgId\" = :org_id OR s.\"isUniversal\" = TRUE)")
        params["org_id"] = str(org_id)

    scope_sql = " AND ".join(scope_conditions)

    # =========================================================================
    # LAYER 1: DENSE VECTOR SEARCH (Meaning & Concepts)
    # =========================================================================
    query_emb = get_embedding(clean_q)
    emb_str = "[" + ",".join(str(x) for x in query_emb) + "]"
    params["query_emb"] = emb_str
    params["top_k_vec"] = top_k_vector

    vector_sql = f"""
        SELECT 
            c.id, c.content, c."sourceId", s.title, s.url,
            (c.embedding <=> CAST(:query_emb AS vector)) AS distance,
            c.language, c."isAiTranslated", c."isFormallyReviewed"
        FROM document_chunks c
        JOIN bot_sources s ON c."sourceId" = s.id
        WHERE {scope_sql}
        ORDER BY distance ASC
        LIMIT :top_k_vec;
    """

    try:
        vec_rows = db.execute(text(vector_sql), params).fetchall()
    except Exception as ve:
        logger.warning(f"⚠️ [Vector Query Fallback]: {ve}")
        vec_rows = []

    vector_ranked = {}
    for rank_idx, row in enumerate(vec_rows, 1):
        cid = str(row[0])
        vector_ranked[cid] = {
            "rank": rank_idx,
            "id": cid,
            "content": row[1],
            "sourceId": str(row[2]),
            "title": row[3] or "Knowledge Document",
            "url": row[4],
            "distance": float(row[5]),
            "vec_sim": max(0.0, 1.0 - float(row[5])),
            "language": row[6] or "en",
            "isAiTranslated": bool(row[7]),
            "isFormallyReviewed": bool(row[8])
        }

    # =========================================================================
    # LAYER 2: POSTGRESQL FULL-TEXT & LEXICAL SEARCH (Exact IDs, SKUs, Numbers)
    # =========================================================================
    # Search terms extraction
    clean_words = [w for w in re.findall(r'[a-zA-Z0-9_\-\.]+', clean_q) if len(w) >= 2]
    lexical_rows = []

    if clean_words:
        # Build tsquery or exact terms ILIKE
        ts_query_str = " | ".join(clean_words[:6])
        params["ts_query"] = ts_query_str
        params["top_k_lex"] = top_k_lexical

        # Combine tsvector text search with substring ILIKE for codes/part numbers
        like_clauses = []
        for i, word in enumerate(clean_words[:3]):
            param_key = f"term_{i}"
            params[param_key] = f"%{word}%"
            like_clauses.append(f"c.content ILIKE :{param_key}")
        like_sql = " OR ".join(like_clauses) if like_clauses else "FALSE"

        lexical_sql = f"""
            SELECT 
                c.id, c.content, c."sourceId", s.title, s.url,
                ts_rank(to_tsvector('english', c.content), to_tsquery('english', :ts_query)) AS lex_rank,
                c.language, c."isAiTranslated", c."isFormallyReviewed"
            FROM document_chunks c
            JOIN bot_sources s ON c."sourceId" = s.id
            WHERE {scope_sql} AND (
                to_tsvector('english', c.content) @@ to_tsquery('english', :ts_query)
                OR {like_sql}
            )
            ORDER BY lex_rank DESC
            LIMIT :top_k_lex;
        """
        try:
            lexical_rows = db.execute(text(lexical_sql), params).fetchall()
        except Exception as le:
            # Fallback to simple ILIKE search if tsquery syntax fails on special chars
            fallback_lex_sql = f"""
                SELECT 
                    c.id, c.content, c."sourceId", s.title, s.url,
                    1.0 AS lex_rank,
                    c.language, c."isAiTranslated", c."isFormallyReviewed"
                FROM document_chunks c
                JOIN bot_sources s ON c."sourceId" = s.id
                WHERE {scope_sql} AND ({like_sql})
                LIMIT :top_k_lex;
            """
            try:
                lexical_rows = db.execute(text(fallback_lex_sql), params).fetchall()
            except Exception as fe2:
                lexical_rows = []

    lexical_ranked = {}
    for rank_idx, row in enumerate(lexical_rows, 1):
        cid = str(row[0])
        lexical_ranked[cid] = {
            "rank": rank_idx,
            "id": cid,
            "content": row[1],
            "sourceId": str(row[2]),
            "title": row[3] or "Knowledge Document",
            "url": row[4],
            "lex_score": float(row[5]) if row[5] is not None else 1.0,
            "language": row[6] or "en",
            "isAiTranslated": bool(row[7]),
            "isFormallyReviewed": bool(row[8])
        }

    # =========================================================================
    # LAYER 3: RECIPROCAL RANK FUSION (RRF)
    # Formula: RRF_Score(d) = 1.0 / (k + rank_vector(d)) + 1.0 / (k + rank_lexical(d))
    # =========================================================================
    all_chunk_ids = set(vector_ranked.keys()).union(set(lexical_ranked.keys()))
    fused_candidates = []

    for cid in all_chunk_ids:
        # Vector score
        vec_entry = vector_ranked.get(cid)
        vec_rank = vec_entry["rank"] if vec_entry else 9999
        vec_score = 1.0 / (rrf_k + vec_rank) if vec_entry else 0.0

        # Lexical score
        lex_entry = lexical_ranked.get(cid)
        lex_rank = lex_entry["rank"] if lex_entry else 9999
        lex_score = 1.0 / (rrf_k + lex_rank) if lex_entry else 0.0

        # Mathematical RRF Score
        rrf_score = vec_score + lex_score

        base_item = vec_entry or lex_entry
        fused_candidates.append({
            "id": cid,
            "text": base_item["content"],
            "title": base_item["title"],
            "url": base_item["url"],
            "sourceId": base_item["sourceId"],
            "language": base_item["language"],
            "isAiTranslated": base_item["isAiTranslated"],
            "isFormallyReviewed": base_item["isFormallyReviewed"],
            "vec_rank": vec_rank if vec_entry else None,
            "lex_rank": lex_rank if lex_entry else None,
            "vec_sim": vec_entry["vec_sim"] if vec_entry else 0.0,
            "rrf_score": rrf_score
        })

    # Sort descending by RRF score
    fused_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
    shortlisted = fused_candidates[:15]

    # =========================================================================
    # LAYER 4: FLASHRANK CROSS-ENCODER RERANKER
    # =========================================================================
    rerank_start = time.time()
    reranked_results = []
    
    if shortlisted:
        try:
            # Map candidates for FlashRank
            formatted_for_rerank = [
                {
                    "id": idx,
                    "text": item["text"],
                    "title": item["title"],
                    "url": item["url"],
                    "original_score": item["rrf_score"]
                }
                for idx, item in enumerate(shortlisted)
            ]

            rr_out = rerank_chunks(clean_q, formatted_for_rerank, top_k=top_k_final, min_score=0.002)
            
            # Map rerank scores back to original objects
            for r in rr_out:
                orig_idx = r.get("id")
                if orig_idx is not None and 0 <= orig_idx < len(shortlisted):
                    item_copy = dict(shortlisted[orig_idx])
                    item_copy["cross_encoder_score"] = r.get("rerank_score", 0.0)

                    # Extract visual & video metadata
                    img_match = re.search(r'!\[([^\]]*)\]\((/static/extracted_diagrams/[^\s\)]+)\)', item_copy["text"])
                    if img_match:
                        item_copy["has_visual"] = True
                        item_copy["visual_caption"] = img_match.group(1)
                        item_copy["visual_url"] = img_match.group(2)
                    else:
                        item_copy["has_visual"] = False

                    time_match = re.search(r'TIMESTAMP:\s*(\d{2}:\d{2})', item_copy["text"])
                    if time_match:
                        item_copy["video_timestamp"] = time_match.group(1)

                    reranked_results.append(item_copy)
        except Exception as re_err:
            logger.warning(f"⚠️ [FlashRank Cross-Encoder Fallback]: {re_err}")
            reranked_results = shortlisted[:top_k_final]

    total_latency_ms = (time.time() - start_time) * 1000.0

    if close_db:
        db.close()

    return {
        "query": clean_q,
        "total_vector_candidates": len(vec_rows),
        "total_lexical_candidates": len(lexical_rows),
        "total_fused_candidates": len(fused_candidates),
        "gold_chunks": reranked_results[:top_k_final],
        "top_rerank_score": reranked_results[0]["cross_encoder_score"] if reranked_results and "cross_encoder_score" in reranked_results[0] else (reranked_results[0]["rrf_score"] if reranked_results else 0.0),
        "total_latency_ms": round(total_latency_ms, 2)
    }
