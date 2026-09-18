import asyncio
from app.db.database import SessionLocal
from app.services.rag import stream_rag_pipeline
from app.services.tracer import get_recent_traces

async def test():
    db = SessionLocal()
    bot_id = "3a9557b0-e181-4b96-9e62-ed4717a96249"
    q = "What is the resignation policy at Alorica?"
    print(f"=== Testing RAG Query: '{q}' ===")
    async for chunk in stream_rag_pipeline(bot_id=bot_id, question=q, db=db):
        pass
    
    traces = get_recent_traces(bot_id=bot_id, limit=1)
    print(f"=== Trace Count: {len(traces)} ===")
    if traces:
        t = traces[0]
        print(f"Trace ID: {t.get('trace_id')}")
        print(f"Retrieval: {t.get('retrieval', {}).get('total_candidates')} candidates, Top Vec: {t.get('retrieval', {}).get('top_vector_score')}")
        print(f"Rerank: Top Score: {t.get('rerank', {}).get('top_rerank_score')} ({t.get('timings_ms', {}).get('rerank_ms')}ms)")
        print(f"Tokens: {t.get('tokens')}")
        print(f"Guardrail: {t.get('guardrail')}")
    db.close()

if __name__ == "__main__":
    asyncio.run(test())
