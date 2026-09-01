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

GROUNDED_SYSTEM_PROMPT = """# THE CORE PRINCIPLE
You are a brilliant, highly intelligent, and helpful AI agent.
Everything you say must be grounded strictly in the Knowledge section below.

RULES FOR HIGH-INTELLIGENCE RESPONSES:
1. Direct Customer-Facing Output: Output ONLY the final customer-facing answer directly. NEVER output any internal reasoning, planning steps, or 'Okay, the user is asking...' scratchpad thoughts.
2. Smart Synthesis: When asked overview questions (such as "What is [Company] about?", "About Us", "What do you offer?", "Who are you?", or "Overview"), synthesize a comprehensive, elegant, and structured answer using all relevant facts, capabilities, tools, and offerings described across the Knowledge section.
3. No Meta-Excuses: Never output robotic disclaimers like "the specific about us section has no text" if the organization's description and capabilities are present in the provided knowledge.
4. Links & Social Media: When asked for links, social channels, or contact info (such as LinkedIn, X/Twitter, Instagram, GitHub, email, or phone), ALWAYS output them as clean markdown links: [Platform Name](URL).
5. Diagrams & Figures: If the Knowledge section contains an image tag or diagram (e.g. ![Figure Caption](image_url)), YOU MUST INCLUDE THAT EXACT IMAGE TAG ![Figure Caption](image_url) in your answer so the user can visually see the diagram in the chat.
6. Truly Missing Information: Only if the Knowledge section genuinely contains zero information related to the question, state politely that you do not have that specific detail in your knowledge base and end your response with [[LEAD_MARKER]].

Knowledge:
{knowledge}
"""

def clean_llm_text(text: str) -> str:
    """Strips <think> tags, reasoning tokens, and scratchpads from LLM responses"""
    if not text:
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("</think>", "").replace("<think>", "").strip()

    if "ANSWER:" in cleaned:
        parts = cleaned.split("ANSWER:")
        cleaned = parts[-1].strip()
    elif "Answer:" in cleaned and len(cleaned.split("Answer:")[0]) > 60:
        parts = cleaned.split("Answer:")
        cleaned = parts[-1].strip()

    # Strip conversational reasoning preambles (e.g. "Okay, the user is asking...")
    if re.search(r'^(okay|let me|the user is asking|scanning the knowledge|looking through the provided|i need to)', cleaned, re.IGNORECASE):
        answer_markers = ['Here is', 'Here are', 'To get in touch', 'VedaOne', 'Appdeft', 'You can', 'Contact', 'Reach out', '**', '- ']
        for marker in answer_markers:
            if marker in cleaned:
                pos = cleaned.find(marker)
                cleaned = cleaned[pos:].strip()
                break

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
                    temperature=0.1,
                    max_tokens=1200
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
                    temperature=0.1,
                    max_tokens=1200
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

    # Extract query keywords for lexical boosting (acronyms, names, technical terms)
    stopwords = {
        'what', 'is', 'the', 'a', 'an', 'in', 'of', 'for', 'to', 'and', 'or', 'on', 'with',
        'about', 'how', 'who', 'where', 'when', 'why', 'can', 'you', 'tell', 'me', 'give',
        'does', 'do', 'did', 'are', 'was', 'were', 'which', 'kaun', 'kya', 'hai', 'hain', 'me', 'se', 'ke'
    }
    words = [w.strip('?,.!\"\'()[]{}') for w in question.lower().split()]
    keywords = [w for w in words if len(w) > 2 and w not in stopwords]

    # Expand common intent synonyms for high-confidence retrieval
    intent_expansions = {
        'cost': ['cost', 'price', 'pricing', 'subscription', 'plans', 'free', 'trial', 'tier'],
        'pricing': ['pricing', 'cost', 'plans', 'subscription', 'free', 'trial'],
        'price': ['price', 'cost', 'pricing', 'plans', 'subscription'],
        'offer': ['offer', 'services', 'capabilities', 'features', 'product', 'toolkit', 'solutions'],
        'touch': ['touch', 'contact', 'email', 'support', 'reach', 'social', 'message'],
        'contact': ['contact', 'email', 'phone', 'support', 'reach', 'touch', 'social', 'channels']
    }
    expanded_keywords = set(keywords)
    for kw in keywords:
        if kw in intent_expansions:
            expanded_keywords.update(intent_expansions[kw])

    results = (
        db.query(
            models.DocumentChunk,
            models.DocumentChunk.embedding.cosine_distance(query_vec).label("distance")
        )
        .join(models.BotSource)
        .filter(models.BotSource.botId == bot_id)
        .order_by("distance")
        .limit(TOP_K_CHUNKS * 2)
        .all()
    )

    if not results:
        yield f"data: {json.dumps({'type': 'token', 'content': 'I do not have any knowledge loaded yet. '})}\n\n"
        yield f"data: {json.dumps({'type': 'lead_form'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'full_text': 'No knowledge'})}\n\n"
        return

    # Hybrid Scoring: Dense Vector Cosine Similarity + Keyword Match Bonus
    scored_chunks = []
    for chunk, distance in results:
        vec_sim = 1.0 - float(distance)
        kw_hits = sum(1 for kw in expanded_keywords if kw in chunk.content.lower())
        kw_boost = min(0.40, kw_hits * 0.20) if expanded_keywords else 0.0
        hybrid_score = round(vec_sim + kw_boost, 4)
        scored_chunks.append((hybrid_score, chunk.content, vec_sim))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    best_score = scored_chunks[0][0]
    print(f"📊 [RAG Hybrid Confidence]: {best_score} (Top Vec: {scored_chunks[0][2]:.4f})")

    passed_chunks = [c[1] for c in scored_chunks if c[0] >= RELEVANCE_FLOOR]
    if not passed_chunks and best_score >= RESCUE_FLOOR:
        passed_chunks = [scored_chunks[0][1]]

    nothing_retrieved = len(passed_chunks) == 0
    knowledge_ctx = "\n\n---\n\n".join(passed_chunks[:TOP_K_CHUNKS]) if passed_chunks else "No relevant content found in knowledge base."

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

    # Guaranteed Visual Image & Diagram Injection if present in retrieved knowledge
    if not nothing_retrieved and not lead_form_required:
        retrieved_images = re.findall(r'!\[([^\]]*)\]\((/static/extracted_diagrams/[^)]+)\)', knowledge_ctx)
        if retrieved_images and '![' not in clean_text:
            # Prepend the primary diagram image to the response
            caption, img_url = retrieved_images[0]
            clean_text = f"![{caption}]({img_url})\n\n" + clean_text

    words = clean_text.split(" ")
    for word in words:
        if word:
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

    # Format source citations for evidence verification
    source_citations = []
    seen_titles = set()
    if not nothing_retrieved and not lead_form_required:
        for chunk, _ in results:
            if chunk.content in passed_chunks:
                source = getattr(chunk, "source", None)
                s_title = source.title if source else "Knowledge Document"
                if s_title not in seen_titles:
                    seen_titles.add(s_title)
                    s_clean = chunk.content.replace('\n', ' ')[:280].strip()
                    source_citations.append({
                        "title": s_title,
                        "kind": getattr(source, "kind", "DOC") if source else "DOC",
                        "url": getattr(source, "url", "") if source else "",
                        "snippet": s_clean
                    })

    if source_citations:
        yield f"data: {json.dumps({'type': 'sources', 'sources': source_citations})}\n\n"

    if lead_form_required:
        yield f"data: {json.dumps({'type': 'lead_form'})}\n\n"

    print(f"📝 [Delivered by {engine_used}]: {clean_text}\n")
    yield f"data: {json.dumps({'type': 'done', 'full_text': clean_text})}\n\n"