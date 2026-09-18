import time
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from collections import deque

# 1. Open-Source Arize Phoenix & OpenTelemetry Instrumentation
_PHOENIX_ACTIVE = False
try:
    from openinference.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()
    _PHOENIX_ACTIVE = True
    print("🔭 [Arize Phoenix Tracer]: OpenAI instrumentation active via OpenInference.")
except Exception as px_err:
    print(f"ℹ️ [Arize Phoenix]: Standby / standard tracing mode ({px_err})")

# In-memory circular buffer for ultra-fast dashboard queries (last 200 traces)
_TRACE_BUFFER = deque(maxlen=200)

class RAGPipelineTracer:
    """
    High-precision RAG Pipeline Observability & Tracing Engine.
    Tracks every query end-to-end:
    - Vector embeddings & hybrid retrieval scores
    - FlashRank cross-encoder rerank scores and candidate sorting
    - Token counting (via tiktoken) and context budget usage
    - Post-LLM Guardrail decisions (escape hatch vs citations)
    - Granular millisecond latencies
    """
    def __init__(self, bot_id: str, query: str, conversation_id: Optional[str] = None):
        self.trace_id = str(uuid.uuid4())
        self.bot_id = str(bot_id)
        self.conversation_id = str(conversation_id) if conversation_id else None
        self.query = query
        self.created_at = datetime.now(timezone.utc)
        self.start_time = time.perf_counter()

        # Step data
        self.retrieval_data: Dict[str, Any] = {}
        self.rerank_data: Dict[str, Any] = {}
        self.token_data: Dict[str, Any] = {}
        self.guardrail_data: Dict[str, Any] = {}
        self.timings: Dict[str, float] = {}
        self.llm_engine: str = ""
        self.response_preview: str = ""

    def log_retrieval(
        self,
        total_candidates: int,
        top_vector_score: float,
        best_hybrid_score: float,
        candidates: List[Dict[str, Any]],
        latency_ms: float
    ):
        self.timings["retrieval_ms"] = round(latency_ms, 2)
        self.retrieval_data = {
            "total_candidates": total_candidates,
            "top_vector_score": round(float(top_vector_score), 4),
            "best_hybrid_score": round(float(best_hybrid_score), 4),
            "candidates": [
                {
                    "id": c.get("id"),
                    "title": c.get("title", "Document"),
                    "source_url": c.get("source_url", ""),
                    "document_type": c.get("document_type", "DOC"),
                    "extracted_date": str(c.get("extracted_date", "")),
                    "vector_score": round(float(c.get("vec_sim", 0.0)), 4),
                    "hybrid_score": round(float(c.get("original_score", 0.0)), 4),
                    "text_preview": (c.get("text", "")[:120] + "...") if len(c.get("text", "")) > 120 else c.get("text", "")
                }
                for c in candidates[:10]
            ]
        }

    def log_rerank(
        self,
        model_name: str,
        latency_ms: float,
        reranked_candidates: List[Dict[str, Any]],
        filtered_out_count: int
    ):
        self.timings["rerank_ms"] = round(latency_ms, 2)
        top_score = reranked_candidates[0].get("rerank_score", 0.0) if reranked_candidates else 0.0
        self.rerank_data = {
            "reranker_model": model_name,
            "top_rerank_score": round(float(top_score), 5),
            "selected_count": len(reranked_candidates),
            "filtered_out_count": filtered_out_count,
            "candidates": [
                {
                    "rank": idx + 1,
                    "title": c.get("title", "Document"),
                    "source_url": c.get("source_url", ""),
                    "document_type": c.get("document_type", "DOC"),
                    "rerank_score": round(float(c.get("rerank_score", 0.0)), 5),
                    "selected": True,
                    "text_preview": (c.get("text", "")[:120] + "...") if len(c.get("text", "")) > 120 else c.get("text", "")
                }
                for idx, c in enumerate(reranked_candidates)
            ]
        }

    def log_tokens(self, telemetry: Dict[str, Any]):
        self.token_data = telemetry

    def log_guardrail(
        self,
        status: str,
        escape_hatch_triggered: bool,
        citations_found: List[str]
    ):
        self.guardrail_data = {
            "status": status,
            "escape_hatch_triggered": escape_hatch_triggered,
            "citations_found": citations_found
        }

    def finalize(
        self,
        llm_engine: str,
        response_text: str,
        db_session = None
    ) -> Dict[str, Any]:
        total_latency_ms = (time.perf_counter() - self.start_time) * 1000
        self.timings["total_ms"] = round(total_latency_ms, 2)
        self.llm_engine = llm_engine
        self.response_preview = (response_text[:180] + "...") if len(response_text) > 180 else response_text

        trace_summary = {
            "trace_id": self.trace_id,
            "bot_id": self.bot_id,
            "conversation_id": self.conversation_id,
            "query": self.query,
            "timestamp": self.created_at.isoformat(),
            "llm_engine": self.llm_engine,
            "response_preview": self.response_preview,
            "timings_ms": self.timings,
            "retrieval": self.retrieval_data,
            "rerank": self.rerank_data,
            "tokens": self.token_data,
            "guardrail": self.guardrail_data
        }

        # 1. Add to in-memory fast ring buffer
        _TRACE_BUFFER.appendleft(trace_summary)

        # 2. Persist to PostgreSQL if db_session is provided
        if db_session:
            try:
                from app.db import models
                trace_db = models.RAGTrace(
                    id=self.trace_id,
                    botId=self.bot_id,
                    conversationId=self.conversation_id,
                    query=self.query,
                    retrievedCandidatesCount=self.retrieval_data.get("total_candidates", 0),
                    topVectorScore=float(self.retrieval_data.get("top_vector_score", 0.0)),
                    topRerankScore=float(self.rerank_data.get("top_rerank_score", 0.0)),
                    rerankLatencyMs=float(self.timings.get("rerank_ms", 0.0)),
                    totalLatencyMs=float(self.timings.get("total_ms", 0.0)),
                    promptTokens=int(self.token_data.get("total_prompt_tokens", 0)),
                    contextTokens=int(self.token_data.get("context_tokens", 0)),
                    totalTokens=int(self.token_data.get("total_prompt_tokens", 0)),
                    guardrailStatus=self.guardrail_data.get("status", "PASSED"),
                    candidatesJson=json.dumps({
                        "retrieval": self.retrieval_data.get("candidates", []),
                        "rerank": self.rerank_data.get("candidates", [])
                    }),
                    createdAt=self.created_at
                )
                db_session.add(trace_db)
                db_session.commit()
            except Exception as e:
                print(f"⚠️ [Trace DB Persistence Warning]: {e}")

        # 3. Log clean structured observability entry to terminal
        print(f"📊 [RAG Trace Completed]: ID={self.trace_id[:8]} | Rerank={self.rerank_data.get('top_rerank_score')} ({self.timings.get('rerank_ms')}ms) | Tokens={self.token_data.get('total_prompt_tokens')} | Guardrail={self.guardrail_data.get('status')}")

        return trace_summary

def get_recent_traces(bot_id: Optional[str] = None, limit: int = 25) -> List[Dict[str, Any]]:
    """Fetches recent traces from the fast ring buffer or PostgreSQL."""
    traces = list(_TRACE_BUFFER)
    if bot_id:
        traces = [t for t in traces if t.get("bot_id") == str(bot_id)]
    return traces[:limit]

