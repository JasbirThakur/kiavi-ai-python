import time
from typing import List, Dict, Any, Optional

try:
    from flashrank import Ranker, RerankRequest
    HAS_FLASHRANK = True
except ImportError:
    HAS_FLASHRANK = False
    Ranker = None
    RerankRequest = None

class SmartReranker:
    """
    High-precision cross-encoder re-ranking service using FlashRank.
    Executes in ~2ms on CPU via ONNX, filtering out irrelevant vector search candidates
    before they enter the LLM context.
    """
    _instance: Optional['SmartReranker'] = None
    _ranker = None

    @classmethod
    def get_instance(cls) -> 'SmartReranker':
        if cls._instance is None:
            cls._instance = SmartReranker()
        return cls._instance

    def __init__(self):
        if HAS_FLASHRANK:
            try:
                # ms-marco-TinyBERT-L-2-v2 is ultra-lightweight (~3.2MB) and optimized for CPU inference
                self._ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
                print("⚡ [SmartReranker]: FlashRank TinyBERT ONNX model loaded successfully.")
            except Exception as e:
                print(f"⚠️ [SmartReranker Init Warning]: {e}")
                self._ranker = None
        else:
            print("⚠️ [SmartReranker Notice]: flashrank library not installed, fallback mode enabled.")

    def rerank_chunks(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 4,
        min_score: float = 0.0005
    ) -> List[Dict[str, Any]]:
        """
        Cross-encodes query and candidate chunks to rank by true semantic relevance.

        Each candidate in `candidates` should be a dict:
        {
            "id": int or str,
            "text": str,
            "title": str (optional),
            "url": str (optional),
            "original_score": float (optional)
        }

        Returns:
            List of top_k candidate dicts sorted by rerank_score descending,
            with 'rerank_score' added.
        """
        if not candidates:
            return []

        if not self._ranker or not HAS_FLASHRANK or not query.strip():
            # Fallback: return top_k candidates by their original order
            for c in candidates:
                if "rerank_score" not in c:
                    c["rerank_score"] = c.get("original_score", 1.0)
            return candidates[:top_k]

        start_time = time.time()
        try:
            passages = [
                {"id": idx, "text": cand.get("text", "")[:1200]}
                for idx, cand in enumerate(candidates)
            ]

            rerank_request = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(rerank_request)

            reranked_candidates = []
            for item in results:
                idx = item.get("id")
                score = float(item.get("score", 0.0))
                if idx is not None and idx < len(candidates):
                    cand_copy = dict(candidates[idx])
                    cand_copy["rerank_score"] = round(score, 5)
                    reranked_candidates.append(cand_copy)

            # Sort strictly descending by rerank score
            reranked_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

            elapsed_ms = (time.time() - start_time) * 1000
            top_score = reranked_candidates[0]["rerank_score"] if reranked_candidates else 0.0
            print(f"🎯 [FlashRank Rerank]: {len(candidates)} candidates reranked in {elapsed_ms:.2f}ms. Top score: {top_score}")

            # Apply relevance threshold filter if top candidate is sufficiently strong
            if top_score >= 0.05:
                filtered = [c for c in reranked_candidates if c["rerank_score"] >= min_score]
                return filtered[:top_k] if filtered else reranked_candidates[:top_k]

            return reranked_candidates[:top_k]

        except Exception as e:
            print(f"⚠️ [SmartReranker Exec Error]: {e}. Falling back to candidate order.")
            for c in candidates:
                if "rerank_score" not in c:
                    c["rerank_score"] = c.get("original_score", 1.0)
            return candidates[:top_k]

# Global singleton helper
_reranker_instance = None

def get_reranker() -> SmartReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = SmartReranker.get_instance()
    return _reranker_instance

def rerank_chunks(query: str, candidates: List[Dict[str, Any]], top_k: int = 4, min_score: float = 0.0005) -> List[Dict[str, Any]]:
    return get_reranker().rerank_chunks(query, candidates, top_k=top_k, min_score=min_score)
