import re
import json
from typing import AsyncGenerator
from sqlalchemy.orm import Session
from openai import OpenAI
from groq import Groq
from config import (
    NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_LLM_MODEL,
    GROQ_API_KEY, RELEVANCE_FLOOR, RESCUE_FLOOR, TOP_K_CHUNKS
)
from services.embedding import get_embedding
import models

# Primary: NVIDIA NIM Engine
nvidia_client = OpenAI(
    base_url=NVIDIA_BASE_URL, 
    api_key=NVIDIA_API_KEY, 
    timeout=30.0
) if NVIDIA_API_KEY else None

# Fallback: Groq Engine
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

GROUNDED_SYSTEM_PROMPT = """# THE ONE RULE THAT MATTERS
Everything you say must come strictly from the Knowledge section below.
It is your ONLY source.

You must NOT:
- invent, estimate, extrapolate, or 'ballpark' any figure or fact
- use any external knowledge from your pre-training
- include any thinking tags, internal reasoning notes, or process logs

RESPONSE GUIDELINES:
1. Always maintain a clear, professional tone in English (or respond in the language asked if requested).
2. If the Knowledge section contains enough information: Answer accurately, directly, and comprehensively. Do NOT include [[LEAD_MARKER]].
3. ONLY if the Knowledge section does NOT contain the answer: State politely that you do not have this information in your knowledge base, and ALWAYS append [[LEAD_MARKER]] at the very end of your response.

Knowledge:
{knowledge}
"""

def clean_llm_text(text: str) -> str:
    """Strips <think> tags and reasoning tokens from Nemotron / DeepSeek models"""
    if not text:
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("</think>", "").replace("<think>", "").strip()
    return cleaned

def generate_llm_response(messages: list) -> tuple[str, str]:
    # 1. Tier 1: NVIDIA NIM (Nemotron Primary)
    if nvidia_client:
        models_to_try = [
            NVIDIA_LLM_MODEL if NVIDIA_LLM_MODEL else "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3-nano-30b-a3b",
            "nvidia/nemotron-3.5-lightning-30b-a3b",
            "meta/llama-3.2-11b-vision-instruct"
        ]
        for nv_model in models_to_try:
            try:
                resp = nvidia_client.chat.completions.create(
                    model=nv_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=512
                )
                raw = resp.choices[0].message.content or ""
                ans = clean_llm_text(raw)
                if ans:
                    print(f"✅ [LLM Tier 1 Active]: NVIDIA ({nv_model}) delivered response.")
                    return ans, f"NVIDIA ({nv_model})"
            except Exception as e:
                print(f"⚠️ [NVIDIA NIM Warning]: {nv_model} failed: {e}")
                continue

    # 2. Tier 2: Groq Fallback Engine
    if groq_client:
        for groq_model in ["llama-3.3-70b-versatile", "gemma2-9b-it"]:
            try:
                resp = groq_client.chat.completions.create(
                    model=groq_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=450
                )
                raw = resp.choices[0].message.content or ""
                ans = clean_llm_text(raw)
                if ans:
                    print(f"✅ [LLM Tier 2 Fallback]: Groq ({groq_model}) delivered response.")
                    return ans, f"Groq ({groq_model})"
            except Exception:
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
    print(f"📊 [RAG Match Confidence]: {best_score}")

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
    lower_raw = raw_text.lower()
    unanswered_phrases = ["do not have", "don't have", "not found", "not mentioned", "mere knowledge", "jaankari nahi", "nahi hai", "cannot find"]
    has_unanswered_text = any(phrase in lower_raw for phrase in unanswered_phrases)

    lead_form_required = nothing_retrieved or has_lead_marker or (has_unanswered_text and best_score < RELEVANCE_FLOOR) or (has_unanswered_text and has_lead_marker)

    clean_text = raw_text.replace("[[LEAD_MARKER]]", "").strip()

    words = clean_text.split(" ")
    for word in words:
        if word:
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

    if lead_form_required:
        yield f"data: {json.dumps({'type': 'lead_form'})}\n\n"

    print(f"📝 [Delivered by {engine_used}]: {clean_text}\n")
    yield f"data: {json.dumps({'type': 'done', 'full_text': clean_text})}\n\n"