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

nvidia_client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
    timeout=8.0,
    max_retries=0
) if NVIDIA_API_KEY and not NVIDIA_API_KEY.startswith("your_") else None

# Fallback: Groq Engine
groq_client = Groq(api_key=GROQ_API_KEY, timeout=4.0, max_retries=0) if GROQ_API_KEY and not GROQ_API_KEY.startswith("your_") and not GROQ_API_KEY.startswith("gsk_3aFf12J8") else None

GROUNDED_SYSTEM_PROMPT = """# THE CORE PRINCIPLE
You are a warm, highly articulate, intelligent, and human-like AI specialist.
Communicate like a knowledgeable, polite human consultant: conversational, clear, helpful, and direct.
Everything you say must be grounded strictly in the Knowledge section below.

HUMANIZED & CRISP POINT-TO-POINT FORMAT RULES:
1. Warm & Natural Opening: Start with a brief, friendly human sentence (e.g., "Certainly! Here is a concise overview for you:", "I'd be glad to share the key details with you:", "Here is the breakdown of what we offer:"). Never sound robotic or repetitive.
2. Short Executive Summary: Follow immediately with a 1-to-2 sentence direct answer under "📌 **Summary:**". Make it crisp and high-impact so the user grasps the core answer in 5 seconds.
3. Point-by-Point Details (Key Highlights): Follow with 3 to 5 clean, focused bullet points under "✨ **Key Highlights & Details:**". Always bold the key term at the beginning of each bullet (e.g. • **Conversational AI Bots:** ...). Keep each bullet concise (1 to 2 lines max) so it is effortless to read without overwhelming the user.
4. Natural Contact & Address Answers: When asked for the company's address, location, email, or phone, provide a warm and direct sentence with the verified details. If a physical address is not publicly listed, state that politely and offer the support channels.
5. Strict Domain & Category Isolation: When asked about a specific product or industry (e.g. steel, footwear, AI software, financial valuation), provide facts ONLY from that specific document/topic. Never mix unrelated industries.
6. Diagrams, Tables & Visuals: If an image or diagram tag (![Caption](url)) is in the knowledge, include it so the user can visually see it. If comparing parameters or workflows, draw a clean structured markdown table.
7. Official PDF Catalogue & Specifications Card:
- When answering about a SPECIFIC industry, product, or service (e.g. steel, footwear, cosmetics, software, valuation, pricing), invite the user to review or download the full official PDF document by including:
---
📑 **Official Specifications & Product Catalogue (PDF)**
*Access or download the complete verified documentation:*
[PDF_CARD:{topic_slug}|{Topic Title}]
(Replace {topic_slug} with a hyphenated lowercase name like steel-specifications, ai-software-development, sports-footwear, or cosmetics-beauty, and {Topic Title} with the real title).
- CRITICAL RULE: If the user asks GENERALLY about catalogues without naming a specific product or industry (e.g. "What catalogues do you have?", "Explore product catalogue", "Show me your catalogue"), DO NOT include any [PDF_CARD:...]. Instead, list the available categories in the knowledge base and ask the user which specific one they want to explore first!
8. Highly Intelligent Dynamic Follow-Up Chips:
Append a structured follow-up block at the very end of your response inside <<<FOLLOW_UP>>> and <<<END_FOLLOW_UP>>> tags with valid JSON:
<<<FOLLOW_UP>>>
{
  "prompt": "What would you like to explore next?",
  "options": ["Explore [Specific Topic A]", "Learn about [Specific Topic B]", "Pricing & Details", "Official Address & Contact"]
}
<<<END_FOLLOW_UP>>>

CRITICAL RULES FOR RESPONSE COMPLETENESS:
- You MUST ALWAYS write the comprehensive, humanized answer in the body BEFORE the <<<FOLLOW_UP>>> block. NEVER output only the follow-up block or empty text.
- If specific pricing or details are not explicitly quoted in the knowledge, provide an informative overview of the company's offerings and explain how the customer can request a custom proposal or reach out.
- Provide 3 to 4 distinct, descriptive options based directly on topics present in the knowledge.

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

def generate_llm_response(messages: list, knowledge_chunks: list = None) -> tuple[str, str]:
    # 1. Tier 1: NVIDIA NIM (Fast ~1.5s - 3s response with strict 7s timeout)
    if nvidia_client:
        nv_model = NVIDIA_LLM_MODEL if NVIDIA_LLM_MODEL else "meta/llama-3.2-11b-vision-instruct"
        try:
            kwargs = {
                "model": nv_model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 600,
                "timeout": 8.0
            }
            resp = nvidia_client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            ans = clean_llm_text(raw)
            if ans and len(ans.strip()) > 5:
                print(f"✅ [LLM Tier 1 Active]: NVIDIA ({nv_model}) delivered response.")
                return ans, f"NVIDIA ({nv_model})"
        except Exception as e:
            print(f"⚠️ [NVIDIA NIM Fallback Triggered]: {nv_model} -> {e}")

    # 2. Tier 2: Groq Fallback Engine
    if groq_client:
        for groq_model in ["llama-3.3-70b-versatile", "gemma2-9b-it"]:
            try:
                resp = groq_client.chat.completions.create(
                    model=groq_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=600,
                    timeout=5.0
                )
                raw = resp.choices[0].message.content or ""
                ans = clean_llm_text(raw)
                if ans and len(ans.strip()) > 5:
                    print(f"✅ [LLM Tier 2 Fallback]: Groq ({groq_model}) delivered response.")
                    return ans, f"Groq ({groq_model})"
            except Exception:
                continue

    # 3. Tier 3: Direct Grounded Knowledge Extraction (0.02s Instant Fallback)
    if knowledge_chunks:
        primary_chunks = knowledge_chunks[:3]
        extracted_lines = []
        for ch in primary_chunks:
            # Strip internal title / URL prefix for clean user presentation
            lines = [l.strip() for l in ch.splitlines() if l.strip() and not l.startswith("===") and not l.startswith("http") and not l.startswith("{") and not l.startswith(".pi-")]
            for line in lines:
                sentences = re.split(r'(?<=[.!?])\s+', line)
                for s in sentences:
                    s_clean = s.strip()
                    if len(s_clean) > 20 and not s_clean.startswith("http") and s_clean not in extracted_lines:
                        extracted_lines.append(s_clean)

        if extracted_lines:
            first_raw = extracted_lines[0]
            clean_sum = re.sub(r'https?://\S+', '', first_raw)
            clean_sum = re.sub(r'[\|\(\)]', ' ', clean_sum).strip()
            if len(clean_sum.split()) > 3:
                summary = clean_sum[:180].strip() + ("..." if len(clean_sum) > 180 else "")
            else:
                summary = "Here are the verified highlights and specifications from the official documentation."

            bullets = []
            for item in extracted_lines[1:]:
                clean_item = re.sub(r'[*_#]', '', item).strip()
                clean_item = re.sub(r'https?://\S+', '', clean_item).strip()
                if len(clean_item) < 15:
                    continue
                if len(clean_item) > 160:
                    clean_item = clean_item[:157].rsplit(' ', 1)[0] + "..."

                if ":" in clean_item:
                    k, v = clean_item.split(":", 1)
                    if len(k.strip()) < 35:
                        bullets.append(f"• **{k.strip()}:** {v.strip()}")
                        if len(bullets) >= 4:
                            break
                        continue

                words = clean_item.split()
                if len(words) >= 4:
                    k = " ".join(words[:3]).title()
                    v = " ".join(words[3:])
                    bullets.append(f"• **{k}:** {v}")
                else:
                    bullets.append(f"• **Key Spec:** {clean_item}")

                if len(bullets) >= 4:
                    break

            if not bullets:
                bullets.append("• **Official Documentation:** Detailed specifications and catalogue available below.")

            synthesized = f"Certainly! Here is a concise overview for you:\n\n📌 **Summary:**\n{summary}\n\n✨ **Key Highlights & Details:**\n" + "\n".join(bullets)
            print("✅ [LLM Tier 3 Fallback]: Direct Grounded Knowledge Extraction delivered response.")
            return synthesized, "Grounded Direct Knowledge"

    return "Certainly! Here is a concise overview for you:\n\n📌 **Summary:**\nVerified official documentation is loaded in the knowledge base.\n\n✨ **Key Highlights & Details:**\n• **Official Grounding:** Certified parameters extracted directly from verified sources.\n• **Detailed Documentation:** Full specifications available for review below.", "Default Knowledge Fallback"

async def stream_rag_pipeline(
    bot_id: str,
    question: str,
    db: Session,
    conversation_id: str = None,
    message_id: str = None
) -> AsyncGenerator[str, None]:
    print(f"\n🔍 [RAG Query]: '{question}'")
    query_vec = get_embedding(question)

    # Extract query keywords for lexical boosting (acronyms, names, technical terms)
    stopwords = {
        'what', 'is', 'the', 'a', 'an', 'in', 'of', 'for', 'to', 'and', 'or', 'on', 'with',
        'about', 'how', 'who', 'why', 'can', 'you', 'your', 'yours', 'my', 'our', 'ours',
        'tell', 'me', 'give', 'we', 'us', 'they', 'them', 'their', 'he', 'she', 'it', 'its',
        'this', 'that', 'these', 'those', 'any', 'some', 'all', 'have', 'has', 'had', 'be',
        'been', 'being', 'do', 'does', 'did', 'are', 'was', 'were', 'which', 'kaun', 'kya',
        'hai', 'hain', 'me', 'se', 'ke', 'ko', 'ki', 'ka'
    }
    # Typo correction & normalization
    normalized_q = question.lower()
    typo_map = {
        'adress': 'address',
        'compny': 'company',
        'comapny': 'company',
        'servce': 'service',
        'locaton': 'location',
        'offce': 'office',
        'phne': 'phone',
        'numbr': 'number',
        'prce': 'price',
        'cetalouge': 'catalogue',
        'cataloge': 'catalogue',
        'catlog': 'catalogue'
    }
    for wrong, right in typo_map.items():
        normalized_q = re.sub(r'\b' + wrong + r'\b', right, normalized_q)

    # 1. Expand keywords from normalized query
    words = [w.strip('?,.!\"\'()[]{}') for w in normalized_q.split()]
    keywords = [w for w in words if len(w) > 2 and w not in stopwords]

    intent_expansions = {
        'cost': ['cost', 'price', 'pricing', 'subscription', 'plans', 'free', 'trial', 'tier'],
        'pricing': ['pricing', 'cost', 'plans', 'subscription', 'free', 'trial'],
        'price': ['price', 'cost', 'pricing', 'plans', 'subscription'],
        'offer': ['offer', 'services', 'capabilities', 'features', 'product', 'toolkit', 'solutions'],
        'touch': ['touch', 'contact', 'email', 'support', 'reach', 'social', 'message', 'phone'],
        'contact': ['contact', 'email', 'phone', 'support', 'reach', 'touch', 'social', 'channels', 'address'],
        'address': ['address', 'location', 'office', 'headquarters', 'city', 'state', 'country', 'located', 'floor', 'sector', 'street', 'where'],
        'location': ['location', 'address', 'office', 'headquarters', 'city', 'country', 'located', 'where'],
        'where': ['where', 'location', 'address', 'office', 'headquarters', 'located', 'city'],
        'catalogue': ['catalogue', 'catalog', 'products', 'inventory', 'directory', 'brochure', 'services', 'items', 'offerings'],
        'catalog': ['catalogue', 'catalog', 'products', 'inventory', 'directory', 'brochure', 'services', 'items', 'offerings']
    }
    expanded_keywords = set(keywords)
    for kw in keywords:
        if kw in intent_expansions:
            expanded_keywords.update(intent_expansions[kw])

    # 2. Scope sources: include bot-specific sources + universal knowledge sources for this organization
    from sqlalchemy import or_, and_
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first()
    bot_org_id = bot.orgId if bot else None

    if bot_org_id:
        source_scope = or_(
            models.BotSource.botId == bot_id,
            and_(models.BotSource.isUniversal == True, models.BotSource.orgId == bot_org_id),
            and_(models.BotSource.isUniversal == True, models.BotSource.orgId == None)
        )
    else:
        source_scope = or_(
            models.BotSource.botId == bot_id,
            models.BotSource.isUniversal == True
        )

    available_sources = db.query(models.BotSource).filter(source_scope).all()

    # ----------------------------------------------------
    # 0. BOT SELF-IDENTITY & COMPANY RESOLVER
    # ----------------------------------------------------
    b_name = (bot.name if bot else "Our Company").strip()
    b_domain = (bot.domain if bot else "").strip()
    b_name_clean = re.sub(r'[^a-z0-9]', '', b_name.lower())
    q_clean = re.sub(r'[^a-z0-9]', '', normalized_q)

    b_name_words = [w for w in re.split(r'[^a-z0-9]+', b_name.lower()) if len(w) >= 3]
    b_domain_clean = re.sub(r'^(https?://)?(www\.)?', '', b_domain.lower()).split('/')[0].split('.')[0]

    is_name_match = (
        (len(b_name_clean) >= 3 and b_name_clean in q_clean) or
        any(w in normalized_q for w in b_name_words) or
        (len(b_domain_clean) >= 3 and b_domain_clean in normalized_q)
    )

    identity_triggers = [
        'who are you', 'who r u', 'what are you', 'what do you do', 'tell me about yourself',
        'about yourself', 'your company', 'about company', 'tell me about your company',
        'what is this bot', 'who is this bot', 'kaun ho', 'kya ho', 'kya karte ho', 'kon ho',
        'tum kaun ho', 'aap kaun ho'
    ]
    is_identity_query = is_name_match or any(trig in normalized_q for trig in identity_triggers)

    if is_identity_query:
        print(f"🤖 [Self-Identity Query Detected]: '{question}' for bot '{b_name}' ({b_domain})")
        # Check if bot has proprietary sources
        prop_sources = [s for s in available_sources if str(s.botId or '') == str(bot_id) and not s.isUniversal]
        has_prop_match = False
        if prop_sources:
            prop_results = (
                db.query(
                    models.DocumentChunk,
                    models.DocumentChunk.embedding.cosine_distance(query_vec).label("distance")
                )
                .join(models.BotSource)
                .filter(models.BotSource.botId == bot_id, models.BotSource.isUniversal == False)
                .order_by("distance")
                .limit(5)
                .all()
            )
            if prop_results:
                best_prop_sim = 1.0 - float(prop_results[0][1])
                if best_prop_sim >= 0.25:
                    has_prop_match = True
                    # Let the pipeline search solely inside proprietary chunks!
                    source_scope = and_(models.BotSource.botId == bot_id, models.BotSource.isUniversal == False)

        if not has_prop_match:
            # Generate verified, branded identity card without leaking universal Nykaa/Steel/Sports
            summary = f"I am **{b_name}**, the official AI assistant for **{b_domain or 'our enterprise'}**. I am designed to assist visitors with instant product guidance, verified documentation, enterprise solutions, and direct support."
            bullets = [
                f"• **Conversational Intelligence:** Delivering instant, verified answers across official documentation 24/7.",
                f"• **Domain Grounding:** Dedicated exclusively to {b_name}'s services, products, and technical specifications.",
                f"• **Direct Support Access:** Assisting with technical documentation, product catalogues, and direct agent connection."
            ]
            body = (
                f"Certainly! Here is a concise overview for you:\n\n"
                f"📌 **Summary:**\n{summary}\n\n"
                f"✨ **Key Highlights & Details:**\n" + "\n".join(bullets) + "\n\n"
                f"How can I assist you with {b_name} today?"
            )
            followup_data = {
                "prompt": f"What would you like to explore regarding {b_name}?",
                "options": ["Explore Product Catalogue 📁", "Pricing & Specifications", "Official Contact & Support", "Schedule a Consultation"]
            }

            if conversation_id:
                try:
                    bot_msg_db = models.Message(
                        conversationId=conversation_id,
                        role="BOT",
                        content=body,
                        unanswered=False
                    )
                    db.add(bot_msg_db)
                    db.commit()
                except Exception as e:
                    print(f"Error saving bot response: {e}")

            yield f"data: {json.dumps({'type': 'start', 'confidence': 1.0})}\n\n"
            for word in body.split(" "):
                if word:
                    yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

            yield f"data: {json.dumps({'type': 'sources', 'sources': [{'title': f'{b_name} Official Profile', 'kind': 'PAGE', 'url': f'https://{b_domain}' if b_domain else '', 'snippet': f'Verified AI assistant profile for {b_name}.'}]})}\n\n"
            yield f"data: {json.dumps({'type': 'followup', 'prompt': followup_data['prompt'], 'options': followup_data['options']})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'full_text': body, 'followup': followup_data})}\n\n"
            return

    # 1. Industry / Category Intent Detection
    industry_keywords = {
        'steel': ['steel', 'metal', 'iron', 'pipe', 'sheet', 'alloy', 'cross-section', 'astm', '7304'],
        'sports': ['sport', 'sports', 'shoe', 'shoes', 'footwear', 'sneaker', 'badminton', 'racket', 'running', 'athletic', 'fitness'],
        'cosmetics': ['cosmetic', 'cosmetics', 'beauty', 'skincare', 'makeup', 'nykaa', 'lipstick', 'serum', 'lotion', 'cream'],
        'software': ['software', 'chatbot', 'chatbots', 'nlp', 'conversational', 'development', 'ai bot', 'consulting'],
        'financial': ['financial', 'valuation', 'projections', 'vedaone', 'dcf']
    }

    matched_industry = None
    for ind, kws in industry_keywords.items():
        if any(kw in normalized_q for kw in kws):
            matched_industry = ind
            break

    is_catalogue_mention = any(k in normalized_q for k in ['catalogue', 'catalog', 'cetalouge', 'cataloge', 'products', 'inventory', 'brochure'])
    is_general_catalogue_query = is_catalogue_mention and (matched_industry is None)

    # If the user asks broadly about the catalogue without selecting a specific category:
    # Present the verified Catalogue Directory Menu with all available industries and prompt them to choose!
    if is_general_catalogue_query:
        b_name = (bot.name if bot else "Our Company").strip()
        has_steel = any('steel' in s.title.lower() or 'metal' in s.title.lower() for s in available_sources)
        has_sports = any('sport' in s.title.lower() or 'shoe' in s.title.lower() or 'archive' in s.title.lower() for s in available_sources)
        has_beauty = any('cosmetic' in s.title.lower() or 'beauty' in s.title.lower() or 'nykaa' in s.title.lower() for s in available_sources)

        bullets = []
        options = []

        bullets.append(f"• **{b_name} Core Solutions:** Custom conversational AI bots, intelligent virtual assistants, enterprise software platforms, and workflow automation.")
        options.append(f"{b_name} AI Solutions")

        if has_steel:
            bullets.append("• **Steel & Metal Specifications:** High-tensile structural steel grades, carbon steel pipes, ASTM/EN standards, and industrial metal classifications.")
            options.append("Explore Steel & Metal Specs")

        if has_sports:
            bullets.append("• **Sports & Footwear Items:** Athletic footwear, performance sports equipment, fitness gear, and active lifestyle apparel.")
            options.append("Explore Sports & Footwear")

        if has_beauty:
            bullets.append("• **Cosmetics & Beauty Store:** Skincare collections, beauty essentials, cosmetics, and personal wellness products.")
            options.append("Explore Cosmetics & Beauty")

        summary = f"Our central database maintains certified technical catalogues for both our {b_name} enterprise solutions and universal industry categories. Please select which catalogue or industry you would like to explore below."

        body = (
            f"Certainly! Here is an overview of the verified product catalogues available in our central database:\n\n"
            f"📌 **Summary:**\n{summary}\n\n"
            f"✨ **Available Product & Industry Catalogues:**\n" + "\n".join(bullets) + "\n\n"
            f"👉 **Which catalogue or industry would you like to explore?**\n"
            f"*Click any option below to view complete specifications, parameters, and access the official certified PDF to view or download:*"
        )

        followup_data = {
            "prompt": "Which catalogue would you like to explore?",
            "options": options[:4]
        }

        yield f"data: {json.dumps({'type': 'start', 'confidence': 1.0})}\n\n"
        words = body.split(" ")
        for word in words:
            if word:
                yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

        yield f"data: {json.dumps({'type': 'sources', 'sources': [{'title': 'Central Knowledge Base', 'kind': 'DOC', 'url': '', 'snippet': 'Certified central catalogue directory.'}]})}\n\n"
        yield f"data: {json.dumps({'type': 'followup', 'prompt': followup_data['prompt'], 'options': followup_data['options']})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'full_text': body, 'followup': followup_data})}\n\n"
        return

    # 2. Dense Vector Search with wide top-40 candidate pool
    results = (
        db.query(
            models.DocumentChunk,
            models.DocumentChunk.embedding.cosine_distance(query_vec).label("distance"),
            models.BotSource.botId.label("src_bot_id"),
            models.BotSource.isUniversal.label("src_is_universal")
        )
        .join(models.BotSource, models.DocumentChunk.sourceId == models.BotSource.id)
        .filter(source_scope)
        .order_by("distance")
        .limit(40)
        .all()
    )

    # 3. Direct SQL Lexical Search to ensure high-intent keywords (address, phone, contact) are never missed
    high_intent_terms = [kw for kw in expanded_keywords if kw in ['address', 'location', 'phone', 'email', 'contact', 'headquarters', 'office']]
    lexical_results = []
    if high_intent_terms:
        lexical_results = (
            db.query(
                models.DocumentChunk,
                models.DocumentChunk.embedding.cosine_distance(query_vec).label("distance"),
                models.BotSource.botId.label("src_bot_id"),
                models.BotSource.isUniversal.label("src_is_universal")
            )
            .join(models.BotSource, models.DocumentChunk.sourceId == models.BotSource.id)
            .filter(source_scope)
            .filter(or_(*[models.DocumentChunk.content.ilike(f"%{term}%") for term in high_intent_terms]))
            .limit(10)
            .all()
        )

    # Merge vector results and lexical candidates (deduplicated by chunk id)
    seen_ids = set()
    candidate_chunks = []
    for chunk, distance, src_bot_id, src_is_universal in results:
        seen_ids.add(chunk.id)
        is_prop = (str(src_bot_id or '') == str(bot_id) and not src_is_universal)
        candidate_chunks.append((chunk, float(distance), is_prop, bool(src_is_universal)))

    for chunk, distance, src_bot_id, src_is_universal in lexical_results:
        if chunk.id not in seen_ids:
            seen_ids.add(chunk.id)
            is_prop = (str(src_bot_id or '') == str(bot_id) and not src_is_universal)
            candidate_chunks.append((chunk, float(distance), is_prop, bool(src_is_universal)))

    if not candidate_chunks:
        yield f"data: {json.dumps({'type': 'token', 'content': 'I do not have any knowledge loaded yet. '})}\n\n"
        yield f"data: {json.dumps({'type': 'lead_form'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'full_text': 'No knowledge'})}\n\n"
        return

    # Hybrid Scoring: Dense Vector Cosine Similarity + Keyword Match Bonus + Intent Boost + Proprietary Boost + Domain Isolation
    scored_chunks = []
    for chunk, distance, is_bot_proprietary, is_universal in candidate_chunks:
        vec_sim = 1.0 - float(distance)
        content_lower = chunk.content.lower()
        kw_hits = sum(1 for kw in expanded_keywords if kw in content_lower)
        intent_boost = 0.35 if any(k in content_lower for k in ['address', 'location', 'phone', 'email', 'contact', 'headquarters']) and any(k in normalized_q for k in ['address', 'where', 'location', 'contact', 'reach', 'phone', 'email']) else 0.0
        kw_boost = min(0.40, kw_hits * 0.15) + intent_boost

        # Priority boost for bot's own proprietary sources
        bot_priority_boost = 0.30 if is_bot_proprietary else 0.0

        # Targeted Industry Boost & Cross-Industry Strict Isolation
        industry_boost = 0.0
        if matched_industry == 'steel':
            if any(k in content_lower for k in ['steel', 'metal', 'iron', '7304', 'alloy', 'cross-section']):
                industry_boost = 0.60
            else:
                industry_boost = -0.60
        elif matched_industry == 'sports':
            if any(k in content_lower for k in ['sport', 'shoe', 'footwear', 'athletic', 'sneaker', 'badminton']):
                industry_boost = 0.60
            else:
                industry_boost = -0.60
        elif matched_industry == 'cosmetics':
            if any(k in content_lower for k in ['cosmetic', 'beauty', 'nykaa', 'skincare', 'makeup']):
                industry_boost = 0.60
            else:
                industry_boost = -0.60
        elif matched_industry == 'software':
            if is_bot_proprietary or any(k in content_lower for k in ['software', 'chatbot', 'ai ', 'nlp', 'development', 'machine learning', 'appdeft', 'vinnisoft']):
                industry_boost = 0.60
            else:
                industry_boost = -0.60
        elif matched_industry is None and not is_catalogue_mention:
            # When the user is NOT asking about a specific catalogue or exploring catalogues,
            # universal shared catalogue chunks (steel, footwear, cosmetics) must NOT leak into general company inquiries!
            if is_universal and any(k in content_lower for k in ['steel', 'metal', 'nykaa', 'cosmetic', 'lipstick', 'shoe', 'footwear', 'badminton']):
                industry_boost = -0.60

        hybrid_score = round(vec_sim + kw_boost + bot_priority_boost + industry_boost, 4)
        scored_chunks.append((hybrid_score, chunk.content, vec_sim))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    best_score = scored_chunks[0][0] if scored_chunks else 0.0
    print(f"📊 [RAG Hybrid Confidence]: {best_score} (Top Vec: {scored_chunks[0][2]:.4f})")

    passed_chunks = [c[1] for c in scored_chunks if c[0] >= 0.28]

    if not passed_chunks or best_score < 0.28:
        print(f"⚠️ [Unanswered Query]: Best score {best_score} < RELEVANCE_FLOOR (0.28). Flagging as content gap.")
        fallback_text = (
            f"Certainly! I want to provide you with the most accurate details, but I don't have verified documentation for '{question}' in my knowledge base yet.\n\n"
            f"📌 **Summary:**\nNo verified records were found in the knowledge base for this inquiry.\n\n"
            f"✨ **Next Steps:**\n"
            f"• **Leave Your Details:** Share your contact details below, and our team will follow up directly with the exact answer.\n"
            f"• **Explore Verified Topics:** You can also ask about our available catalogues, company overview, or official contact channels."
        )

        if message_id:
            try:
                user_msg_db = db.query(models.Message).filter(models.Message.id == message_id).first()
                if user_msg_db:
                    user_msg_db.unanswered = True
                    db.commit()
            except Exception as e:
                print(f"Error marking message unanswered: {e}")

        if conversation_id:
            try:
                bot_msg_db = models.Message(
                    conversationId=conversation_id,
                    role="BOT",
                    content=fallback_text,
                    unanswered=False
                )
                db.add(bot_msg_db)
                db.commit()
            except Exception as e:
                print(f"Error saving bot response: {e}")

        yield f"data: {json.dumps({'type': 'start', 'confidence': best_score})}\n\n"
        for word in fallback_text.split(" "):
            if word:
                yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

        yield f"data: {json.dumps({'type': 'lead_form'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'full_text': fallback_text})}\n\n"
        return

    nothing_retrieved = len(passed_chunks) == 0
    truncated_passed = [c[:1200] for c in passed_chunks[:TOP_K_CHUNKS]]
    knowledge_ctx = "\n\n---\n\n".join(truncated_passed) if truncated_passed else "No relevant content found in knowledge base."

    # Catalogue & Domain Awareness: Inject real active source titles so broad overview questions reflect exact database content
    available_sources = db.query(models.BotSource).filter(source_scope).all()
    if available_sources:
        cat_summary_lines = ["=== Officially Active Knowledge Sources & Catalogues in Database ==="]
        for src in available_sources:
            src_type = "Universal Shared Catalogue" if src.isUniversal else "Bot Proprietary Source"
            cat_summary_lines.append(f"- {src.title} [{src_type}]")
        cat_header = "\n".join(cat_summary_lines)
        knowledge_ctx = f"{cat_header}\n\n---\n\n{knowledge_ctx}"

    messages = [
        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT.replace("{knowledge}", knowledge_ctx)},
        {"role": "user", "content": question}
    ]

    yield f"data: {json.dumps({'type': 'start', 'confidence': best_score})}\n\n"

    raw_text, engine_used = generate_llm_response(messages, passed_chunks)

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

    # 1. Parse follow-up block if present (handling markdown bolding or missing end tags)
    followup_data = None
    followup_match = re.search(r'<<<FOLLOW_UP>>>(.*?)(?:<<<END_FOLLOW_UP>>>|$)', raw_text, re.DOTALL)
    if not followup_match:
        followup_match = re.search(r'(\{\s*"prompt"\s*:\s*"[^"]+",\s*"options"\s*:\s*\[.*?\]\s*\})', raw_text, re.DOTALL)
    if followup_match:
        raw_json = followup_match.group(1).strip()
        raw_json = re.sub(r'^```(?:json)?', '', raw_json).rstrip('`').strip()
        j_start = raw_json.find('{')
        j_end = raw_json.rfind('}')
        if j_start != -1 and j_end != -1:
            try:
                parsed = json.loads(raw_json[j_start:j_end+1])
                if isinstance(parsed, dict) and "prompt" in parsed and "options" in parsed and isinstance(parsed["options"], list):
                    cleaned_opts = [str(o).strip() for o in parsed["options"] if str(o).strip()][:4]
                    if cleaned_opts:
                        followup_data = {
                            "prompt": str(parsed["prompt"]).strip(),
                            "options": cleaned_opts
                        }
            except Exception as f_err:
                print(f"⚠️ [Followup Parse Error]: {f_err}")

    # Remove follow-up block from clean_text so it never leaks into user-facing text
    clean_text = re.sub(r'[*_]*<{2,4}\s*FOLLOW_UP\s*>+.*?((?:<{2,4}\s*(?:END_)?FOLLOW_UP\s*>+)|$)', '', raw_text, flags=re.DOTALL).strip()
    clean_text = re.sub(r'<{2,4}\s*(?:END_)?FOLLOW_UP\s*>+', '', clean_text).strip()
    clean_text = re.sub(r'<{2,4}\s*$', '', clean_text).strip()
    if followup_data:
        clean_text = re.sub(r'\{\s*"prompt"\s*:\s*"[^"]+",\s*"options"\s*:\s*\[.*?\]\s*\}', '', clean_text, flags=re.DOTALL).strip()
    clean_text = clean_text.replace("[[LEAD_MARKER]]", "").strip()

    # Ironclad Fallback: If clean_text was empty or completely stripped, synthesize a grounded answer directly from passed chunks
    if not clean_text or len(clean_text.strip()) < 15:
        if passed_chunks:
            extracted_lines = []
            for ch in passed_chunks[:2]:
                for line in ch.splitlines():
                    line_s = line.strip()
                    if line_s and not line_s.startswith("===") and not line_s.startswith("http") and not line_s.startswith("{") and not line_s.startswith(".pi-"):
                        extracted_lines.append(line_s)
            if extracted_lines:
                sum_line = extracted_lines[0]
                bullets = [f"• **{l[:25]}:** {l}" for l in extracted_lines[1:4]]
                clean_text = f"Certainly! Here is a concise overview for you:\n\n📌 **Summary:**\n{sum_line}\n\n✨ **Key Highlights & Details:**\n" + "\n".join(bullets)
        if not clean_text or len(clean_text.strip()) < 15:
            clean_text = "Certainly! Here is a concise overview for you:\n\n📌 **Summary:**\nVerified official documentation is loaded in the knowledge base.\n\n✨ **Key Highlights & Details:**\n• **Official Grounding:** Certified parameters extracted directly from verified sources.\n• **Detailed Documentation:** Full specifications available for review below."

    # Dynamic Multi-Topic Follow-Up Generator (Fallback when LLM omitted structured tags)
    if not followup_data and not nothing_retrieved and not lead_form_required:
        bullet_matches = re.findall(r'(?:^|\n)\s*[-*•]\s*(?:(?:Explore|Learn about|Discover|Browse)\s+)?([A-Z][^\n:]{3,40})', raw_text)
        if len(bullet_matches) >= 2:
            cleaned_bullets = [b.strip() for b in bullet_matches if len(b.strip()) < 35][:4]
            followup_data = {
                "prompt": "Which department or category would you like to explore?",
                "options": cleaned_bullets
            }

    if matched_industry and not nothing_retrieved and not lead_form_required:
        b_name = (bot.name if bot else "Our Company").strip()
        other_options = []
        if matched_industry != 'steel' and any('steel' in s.title.lower() or 'metal' in s.title.lower() for s in available_sources):
            other_options.append("Explore Steel & Metal Specs")
        if matched_industry != 'sports' and any('sport' in s.title.lower() or 'shoe' in s.title.lower() or 'archive' in s.title.lower() for s in available_sources):
            other_options.append("Explore Sports & Footwear")
        if matched_industry != 'cosmetics' and any('cosmetic' in s.title.lower() or 'beauty' in s.title.lower() or 'nykaa' in s.title.lower() for s in available_sources):
            other_options.append("Explore Cosmetics & Beauty")
        if matched_industry != 'software':
            other_options.append(f"{b_name} AI Solutions")
        other_options.append("Pricing & Specifications")
        other_options.append("Official Contact & Support")

        followup_data = {
            "prompt": "What would you like to explore next?",
            "options": other_options[:4]
        }

    elif not followup_data and not nothing_retrieved and not lead_form_required:
        candidate_options = []
        # Derive directly from available database sources (offering other departments/topics)
        if available_sources:
            for s in available_sources:
                st = (s.title or '').lower()
                if ('steel' in st or 'metal' in st or 'iron' in st) and not any(k in question.lower() for k in ['steel', 'metal', 'iron']):
                    candidate_options.append("Explore Steel & Metal Specs")
                elif ('sports' in st or 'footwear' in st or 'shoe' in st or 'archive' in st) and not any(k in question.lower() for k in ['sports', 'footwear', 'shoe', 'badminton']):
                    candidate_options.append("Explore Sports & Footwear")
                elif ('vasudev' in st or 'software' in st or 'development' in st) and not any(k in question.lower() for k in ['software', 'development', 'vasudev']):
                    candidate_options.append("AI Software Development")
                elif ('vedaone' in st or 'valuation' in st) and not any(k in question.lower() for k in ['valuation', 'financial', 'vedaone']):
                    candidate_options.append("Financial Valuation AI")
                elif ('nykaa' in st or 'beauty' in st or 'cosmetics' in st) and not any(k in question.lower() for k in ['nykaa', 'beauty', 'cosmetics']):
                    candidate_options.append("Explore Cosmetics & Beauty")
                elif ('youtube' in st) and 'youtube' not in question.lower():
                    candidate_options.append("YouTube Guidelines")

        if not any(k in question.lower() for k in ['catalogue', 'catalog']):
            candidate_options.append("Explore Product Catalogue 📁")
        if not any(k in question.lower() for k in ['price', 'pricing', 'cost']):
            candidate_options.append("Pricing & Specifications")
        if not any(k in question.lower() for k in ['contact', 'address', 'office', 'phone', 'reach']):
            candidate_options.append("Official Contact & Support")

        filtered_options = [opt for opt in candidate_options if opt.lower() not in question.lower()]
        if filtered_options:
            selected = list(dict.fromkeys(filtered_options))[:4]
            followup_data = {
                "prompt": "What would you like to explore next?",
                "options": selected
            }

    # Guaranteed Visual Image & Diagram Injection if present in retrieved knowledge
    if not nothing_retrieved and not lead_form_required:
        retrieved_images = re.findall(r'!\[([^\]]*)\]\((/static/extracted_diagrams/[^)]+)\)', knowledge_ctx)
        if retrieved_images and '![' not in clean_text:
            # Prepend the primary diagram image to the response
            caption, img_url = retrieved_images[0]
            clean_text = f"![{caption}]({img_url})\n\n" + clean_text

    # Inject Official PDF Catalogue Action Card ONLY when a specific product/industry/topic has been selected or queried
    if matched_industry and not nothing_retrieved and not lead_form_required:
        if '[PDF_CARD:' not in clean_text and '[VIEW_PDF:' not in clean_text:
            b_name = (bot.name if bot else "Our Company").strip()
            topic_clean = None
            topic_slug = None

            if matched_industry == 'steel':
                topic_clean = "Steel & Metal Specifications"
                topic_slug = "steel-specifications"
            elif matched_industry == 'sports':
                topic_clean = "Sports & Footwear Catalogue"
                topic_slug = "sports-footwear"
            elif matched_industry == 'cosmetics':
                topic_clean = "Cosmetics & Beauty Product Catalogue"
                topic_slug = "cosmetics-beauty"
            elif matched_industry == 'software':
                topic_clean = f"{b_name} AI Software & Solutions"
                topic_slug = "ai-software-development"
            elif matched_industry == 'financial':
                topic_clean = "VedaOne Financial AI Specifications"
                topic_slug = "financial-valuation"

            if topic_clean and topic_slug:
                clean_text += f"\n\n---\n📑 **Official Specifications & Product Catalogue (PDF)**\n*Would you like to review or download the complete certified documentation for this?*\n\n[PDF_CARD:{topic_slug}|{topic_clean}]"

    words = clean_text.split(" ")
    for word in words:
        if word:
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

    # Format source citations for evidence verification
    source_citations = []
    seen_titles = set()
    if not nothing_retrieved and not lead_form_required:
        for row in results:
            chunk = row[0]
            if chunk.content in passed_chunks:
                source = db.query(models.BotSource).filter(models.BotSource.id == chunk.sourceId).first()
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

    if followup_data and not lead_form_required:
        yield f"data: {json.dumps({'type': 'followup', 'prompt': followup_data['prompt'], 'options': followup_data['options']})}\n\n"

    if lead_form_required:
        yield f"data: {json.dumps({'type': 'lead_form'})}\n\n"
        if message_id:
            try:
                user_msg_db = db.query(models.Message).filter(models.Message.id == message_id).first()
                if user_msg_db:
                    user_msg_db.unanswered = True
                    db.commit()
            except Exception as e:
                print(f"Error marking message unanswered: {e}")

    if conversation_id:
        try:
            bot_msg_db = models.Message(
                conversationId=conversation_id,
                role="BOT",
                content=clean_text,
                unanswered=False
            )
            db.add(bot_msg_db)
            db.commit()
        except Exception as e:
            print(f"Error saving bot response: {e}")

    print(f"📝 [Delivered by {engine_used}]: {clean_text}\n")
    done_payload = {'type': 'done', 'full_text': clean_text}
    if followup_data and not lead_form_required:
        done_payload['followup'] = followup_data
    yield f"data: {json.dumps(done_payload)}\n\n"
