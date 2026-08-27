import re
import json
from typing import AsyncGenerator
from sqlalchemy.orm import Session
from groq import Groq
from openai import OpenAI
from config import (
    GROQ_API_KEY, NVIDIA_API_KEY, NVIDIA_BASE_URL,
    RELEVANCE_FLOOR, RESCUE_FLOOR, TOP_K_CHUNKS
)
from services.embedding import get_embedding
import models

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
nvidia_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=18.0) if NVIDIA_API_KEY else None

GROUNDED_SYSTEM_PROMPT = """# THE ONE RULE THAT MATTERS
Everything you say must come strictly from the Knowledge section below.
It is your ONLY source.

You must NOT:
- invent, estimate, extrapolate, or 'ballpark' any figure or fact
- use any external knowledge from your pre-training
- include any thinking tags, internal reasoning notes, or process logs

LEAD MARKER RULES:
1. If the Knowledge section contains enough information: Answer accurately and directly. Do NOT include [[LEAD_MARKER]].
2. ONLY if the Knowledge section does NOT contain the answer: State politely that you do not have this information in your knowledge base, and append [[LEAD_MARKER]] at the end.

Knowledge:
{knowledge}
"""

def clean_llm_text(text: str) -> str:
    """Strips <think> tags from reasoning models"""
    if not text:
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("</think>", "").replace("<think>", "").strip()
    return cleaned

def generate_llm_response(messages: list) -> tuple[str, str]:
    # 1. Try Groq Dynamic Discovery
    if groq_client:
        try:
            m_list = groq_client.models.list().data
            active_models = [m.id for m in m_list if "whisper" not in m.id and "guard" not in m.id]
        except Exception:
            active_models = ["gemma2-9b-it", "mixtral-8x7b-32768"]

        for model_id in active_models:
            try:
                resp = groq_client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=450
                )
                raw = resp.choices[0].message.content or ""
                ans = clean_llm_text(raw)
                if ans:
                    print(f"✅ [LLM Tier 1 Active]: Groq ({model_id}) delivered answer.")
                    return ans, f"Groq ({model_id})"
            except Exception as e:
                continue

    # 2. Try NVIDIA NIM Backup
    if nvidia_client:
        for nv_model in ["nvidia/llama-3.1-nemotron-70b-instruct", "meta/llama-3.3-70b-instruct"]:
            try:
                resp = nvidia_client.chat.completions.create(
                    model=nv_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=450
                )
                raw = resp.choices[0].message.content or ""
                ans = clean_llm_text(raw)
                if ans:
                    print(f"✅ [LLM Tier 2 Active]: NVIDIA ({nv_model}) delivered answer.")
                    return ans, f"NVIDIA ({nv_model})"
            except Exception as e:
                continue

    return "", "None"

async def stream_rag_pipeline(bot_id: str, question: str, db: Session) -> AsyncGenerator[str, None]:
    print(f"\n🔍 [RAG Query]: '{question}'")
    query_vec = get_embedding(question)

    results = (
        db.query(
            models.DocumentChunk,
            models.DocumentChunk.embedding.cosine_distance(query_vec).label("distance")
        )
        .join(models.BotSource)
        .filter(models.BotSource.botId == bot_id)
        .order_by("distance")
        .limit(TOP_K_CHUNKS)
        .all()
    )

    if not results:
        yield f"data: {json.dumps({'type': 'token', 'content': 'I do not have any knowledge loaded yet. '})}\n\n"
        yield f"data: {json.dumps({'type': 'lead_form'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'full_text': 'No knowledge'})}\n\n"
        return

    scored_chunks = [(1.0 - float(distance), chunk.content) for chunk, distance in results]
    best_score = round(scored_chunks[0][0], 4)
    print(f"📊 [RAG Match Score]: {best_score}")

    passed_chunks = [c[1] for c in scored_chunks if c[0] >= RELEVANCE_FLOOR]
    if not passed_chunks and best_score >= RESCUE_FLOOR:
        passed_chunks = [scored_chunks[0][1]]

    nothing_retrieved = len(passed_chunks) == 0
    knowledge_ctx = "\n\n---\n\n".join(passed_chunks) if passed_chunks else "No relevant content found in knowledge base."

    messages = [
        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT.format(knowledge=knowledge_ctx)},
        {"role": "user", "content": question}
    ]

    yield f"data: {json.dumps({'type': 'start', 'confidence': best_score})}\n\n"

    raw_text, engine_used = generate_llm_response(messages)

    if not raw_text:
        yield f"data: {json.dumps({'type': 'token', 'content': 'I am unable to process your request at the moment. '})}\n\n"
        yield f"data: {json.dumps({'type': 'lead_form'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'full_text': 'Error'})}\n\n"
        return

    has_lead_marker = "[[LEAD_MARKER]]" in raw_text
    lead_form_required = nothing_retrieved or (has_lead_marker and best_score < RELEVANCE_FLOOR) or ("don't have" in raw_text.lower() and has_lead_marker)

    clean_text = raw_text.replace("[[LEAD_MARKER]]", "").strip()

    words = clean_text.split(" ")
    for word in words:
        if word:
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

    if lead_form_required:
        yield f"data: {json.dumps({'type': 'lead_form'})}\n\n"

    print(f"📝 [Final Answer Delivered by {engine_used}]: {clean_text}\n")
    yield f"data: {json.dumps({'type': 'done', 'full_text': clean_text})}\n\n"