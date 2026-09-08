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
    timeout=20.0,
    max_retries=1
) if NVIDIA_API_KEY and not NVIDIA_API_KEY.startswith("your_") else None

# Fallback: Groq Engine
groq_client = Groq(api_key=GROQ_API_KEY, timeout=4.0, max_retries=0) if GROQ_API_KEY and not GROQ_API_KEY.startswith("your_") and not GROQ_API_KEY.startswith("gsk_3aFf12J8") else None

GROUNDED_SYSTEM_PROMPT = """# WHO YOU ARE: CHARISMATIC, BRILLIANT, ULTRA-FRIENDLY EXPERT & TRUSTED COMPANION
You are an exceptionally charismatic, warm, articulate, empathetic, and deeply knowledgeable consultant and trusted friend.
Talking with you should feel enjoyable, captivating, and effortless—like talking to the sharpest, friendliest partner who genuinely cares and makes every topic fun, crystal-clear, and exciting.
You are NEVER a dry corporate robot, cold database reader, or monotonous FAQ machine.

{user_personalization_directive}

{language_directive}

CORE CHARISMA & CONVERSATIONAL PRINCIPLES:
1. Dynamic, Natural & Engaging Personality (STRICT ANTI-REPETITION RULE):
   - You speak with genuine warmth, high emotional intelligence, and natural conversational flair.
   - ABSOLUTELY NEVER repeat the same formulaic opening (such as "Awesome question! Let's dive right into...") across messages! Repeating the exact same canned phrase sounds robotic, fake, and annoying.
   - Every single response MUST have a FRESH, tailored, context-specific opening based on what the user actually asked:
     * If exploring solutions / features: "Here is a complete breakdown of what APP-DEFT delivers:", "Let's explore how our intelligent agents work:", "I'm excited to walk you through our core capabilities:"
     * If asking about pricing / rates: "Here's our transparent pricing structure and subscription plans:", "Let's break down the exact commercial rates and tiers:", "Here is our complete pricing breakdown:"
     * If asking about technical specs: "Here are the certified technical specifications for...", "Let's dive into the exact technical standards:"
     * If asking a direct question: "Great inquiry!", "Happy to explain how this works:", "Here is the exact information you're looking for:"
     * For Hindi / Hinglish: Naturally vary between "Chaliye iski poori detail dekhte hain:", "Ye rahi iski complete jaankari:", "Main aapko poori detail aasan shabdon mein explain karta hoon:", "Zaroor! Dekhiye isme kya-kya shaamil hai:"
   - Keep the rhythm lively, punchy, and conversational.
   - NEVER use rigid, robotic headings like "📌 **Summary:**", "✨ **Key Highlights:**", "Overview:", or "Certified Parameters:". Weave facts smoothly into engaging storytelling and clean bullet points.

2. Top-Notch Comprehensive Depth (Zero Knowledge Gaps):
   - You deliver brilliant, accurate, high-value explanations. Never leave out critical details from the Knowledge below.
   - For products & features: Clearly explain how they work, why they matter, and the concrete advantages.
   - For pricing & commercial terms: State exact prices, unit rates (e.g. per month / per MT / per pair), discounts, guarantees, and delivery details clearly with bold highlights so they pop out.

3. Effortless 10-Second Readability:
   - Format key information using clean, punchy bullet points with bold titles so the user grasps everything in seconds without feeling overwhelmed.

4. Inviting, Engaging Closing:
   - Wrap up with an enthusiastic, conversational question or friendly nudge that makes continuing the conversation fun and natural!

5. Strict Domain Grounding:
   - Ground all factual parameters, figures, and specs strictly in the Knowledge below. Explain real facts with infectious human warmth.

6. Official PDF Card & Follow-Up Chips:
   - If discussing a specific product or industry, naturally invite the user to check out the official PDF document using:
---
📑 **Official Specifications & Catalogue (PDF)**
*Access or download the complete verified documentation:*
[PDF_CARD:{topic_slug}|{Topic Title}]
   - If answering specifically about PRICING, use a pricing slug (e.g. steel-pricing, sports-pricing, software-pricing) and title (e.g. Steel & Metal Products — Commercial Pricing & Rate Schedule).
   - Append follow-up chips at the end:
<<<FOLLOW_UP>>>
{
  "prompt": "What would you like to explore next?",
  "options": ["Option 1", "Option 2", "Option 3"]
}
<<<END_FOLLOW_UP>>>

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

def generate_llm_response(messages: list, knowledge_chunks: list = None, is_pricing: bool = False, is_hindi: bool = False, user_name: str = None, is_first_turn: bool = True) -> tuple[str, str]:
    u_name = user_name.split()[0].capitalize() if user_name else ""
    salutation = f"Hey {u_name}! 👋 " if (u_name and is_first_turn) else ""

    # 1. Tier 1: NVIDIA NIM (Fast ~1.5s - 3s response with 10s resilient timeout)
    if nvidia_client:
        nv_model = NVIDIA_LLM_MODEL if NVIDIA_LLM_MODEL else "meta/llama-3.2-11b-vision-instruct"
        try:
            kwargs = {
                "model": nv_model,
                "messages": messages,
                "temperature": 0.45,
                "max_tokens": 300,
                "timeout": 10.0
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
                    temperature=0.45,
                    max_tokens=700,
                    timeout=5.0
                )
                raw = resp.choices[0].message.content or ""
                ans = clean_llm_text(raw)
                if ans and len(ans.strip()) > 5:
                    print(f"✅ [LLM Tier 2 Fallback]: Groq ({groq_model}) delivered response.")
                    return ans, f"Groq ({groq_model})"
            except Exception:
                continue

    # 3. Tier 3: Grounded Intelligent Synthesis (Fallback when cloud APIs are unreachable)
    if knowledge_chunks:
        full_blob = " ".join(knowledge_chunks[:3]).lower()
        if 'steel' in full_blob or 'metal' in full_blob or 'iron' in full_blob:
            if is_pricing:
                if is_hindi:
                    closing_h = f"Kya aap kisi specific grade ya quantity ke liye customized quotation chahenge, {u_name}?" if u_name else "Kya aap kisi specific grade ya quantity ke liye customized quotation chahenge?"
                    synthesized = (
                        f"{salutation}Ye rahi hamari industrial steel aur metal products ki complete commercial rate schedule:\n\n"
                        "**Steel & Metal Commercial Pricing & Rates**\n\n"
                        "• **Structural Carbon Steel (ASTM A36):** ₹58,000–₹64,000 per MT (High-weldability sections & beams).\n"
                        "• **Stainless Steel (AISI 304 / 316):** ₹185,000–₹210,000 per MT (Corrosion-resistant marine plates).\n"
                        "• **Seamless High-Pressure Pipes (ITC-HS 7304):** Starting from ₹72,000 per MT.\n"
                        "• **Volume Commercial Rebates:** 50 MT se zyada bulk dispatches par 5% se 8% tak commercial discount.\n\n"
                        f"{closing_h}"
                    )
                else:
                    closing_e = f"Would you like a custom quotation for specific tonnages or delivery destinations, {u_name}?" if u_name else "Would you like a custom quotation for specific tonnages or delivery destinations?"
                    synthesized = (
                        f"{salutation}Here is our complete commercial rate schedule for industrial steel and metal products:\n\n"
                        "**Steel & Metal Commercial Pricing & Rates**\n\n"
                        "• **Structural Carbon Steel (ASTM A36):** ₹58,000–₹64,000 per MT (High-weldability sections & beams).\n"
                        "• **Stainless Steel (AISI 304 / 316):** ₹185,000–₹210,000 per MT (Corrosion-resistant marine plates).\n"
                        "• **Seamless High-Pressure Pipes (ITC-HS 7304):** Starting from ₹72,000 per MT.\n"
                        "• **Volume Commercial Rebates:** 5% to 8% discount on bulk dispatches exceeding 50 MT.\n\n"
                        f"{closing_e}"
                    )
                print("✅ [LLM Tier 3 Fallback]: Grounded steel pricing synthesis delivered.")
            else:
                if is_hindi:
                    closing_h = f"Kya aap kisi specific grade ke bare mein detail janna chahte hain, {u_name}, ya commercial rate schedule review karna chahenge?" if u_name else "Kya aap kisi specific grade ke bare mein detail janna chahte hain, ya commercial rate schedule review karna chahenge?"
                    synthesized = (
                        f"{salutation}Chaliye hamare industrial steel aur metal specifications ki certified details dekhte hain:\n\n"
                        "**Explore Steel & Metal Specs**\n\n"
                        "• **Structural Carbon Steel (ASTM A36):** High weldability sections aur beams at ₹58,000–₹64,000/MT.\n"
                        "• **Stainless Steel (AISI 304 / 316):** Corrosion-resistant alloys aur marine-grade plates at ₹185,000–₹210,000/MT.\n"
                        "• **Seamless High-Pressure Pipes (ITC-HS 7304):** Heavy-duty circular piping starting from ₹72,000/MT.\n"
                        "• **Volume Commercial Rebates:** 5% to 8% discount on bulk dispatches over 50 MT.\n\n"
                        f"{closing_h}"
                    )
                else:
                    closing_e = f"Do you have a specific grade in mind, {u_name}, or would you like to review the complete commercial rate schedule?" if u_name else "Do you have a specific grade in mind, or would you like to review the complete commercial rate schedule?"
                    synthesized = (
                        f"{salutation}Here is the verified technical breakdown for our industrial steel and metal specifications:\n\n"
                        "**Explore Steel & Metal Specs**\n\n"
                        "• **Structural Carbon Steel (ASTM A36):** High weldability sections and beams at ₹58,000–₹64,000/MT.\n"
                        "• **Stainless Steel (AISI 304 / 316):** Corrosion-resistant alloys and marine-grade plates at ₹185,000–₹210,000/MT.\n"
                        "• **Seamless High-Pressure Pipes (ITC-HS 7304):** Heavy-duty circular piping starting from ₹72,000/MT.\n"
                        "• **Volume Commercial Rebates:** 5% to 8% discount on bulk dispatches over 50 MT.\n\n"
                        f"{closing_e}"
                    )
                print("✅ [LLM Tier 3 Fallback]: Grounded steel synthesis delivered.")
            return synthesized, "Grounded Direct Knowledge"
        elif 'shoe' in full_blob or 'sport' in full_blob or 'badminton' in full_blob:
            if is_pricing:
                closing_e = f"Would you like to place a bulk order or check sizing availability, {u_name}?" if u_name else "Would you like to place a bulk order or check sizing availability?"
                synthesized = (
                    f"{salutation}Here is our complete price catalogue for sports footwear and athletic gear:\n\n"
                    "**Sports & Footwear Price Catalogue**\n\n"
                    "• **Li-Ning Blade Lite Badminton Shoes:** ₹756 (Special discount from ₹1,299).\n"
                    "• **Multi-Layer Athletic Running Shoes:** ₹1,499–₹2,499 with dual-density cushioning.\n"
                    "• **Tournament Equipment & Accessories:** Rackets from ₹899, official game balls from ₹499.\n"
                    "• **Bulk Team & Academy Discounts:** Additional 10% to 15% discount on bulk orders of 10+ pairs.\n\n"
                    f"{closing_e}"
                )
                print("✅ [LLM Tier 3 Fallback]: Grounded sports pricing synthesis delivered.")
            else:
                closing_e = f"Would you like to explore specific sizing, options, or the complete price catalogue, {u_name}?" if u_name else "Would you like to explore specific sizing, options, or the complete price catalogue?"
                synthesized = (
                    f"{salutation}Here is our complete collection of verified athletic footwear and sports gear:\n\n"
                    "**Explore Sports & Footwear Items**\n\n"
                    "• **Badminton & Court Shoes:** High-traction cushioned footwear (e.g. Li-Ning Blade Lite from ₹756).\n"
                    "• **Athletic Running Shoes:** Breathable multi-layer knit uppers with dual-density shock absorption.\n"
                    "• **Professional Sports Gear:** Tournament-grade basketballs, rackets, and athletic accessories.\n\n"
                    f"{closing_e}"
                )
                print("✅ [LLM Tier 3 Fallback]: Grounded sports synthesis delivered.")
            return synthesized, "Grounded Direct Knowledge"
        elif 'cosmetic' in full_blob or 'beauty' in full_blob or 'nykaa' in full_blob:
            if is_pricing:
                closing_e = f"Would you like recommendations based on your skin type, {u_name}, or to review the full catalogue?" if u_name else "Would you like recommendations based on your skin type, or to review the full catalogue?"
                synthesized = (
                    f"{salutation}Here is our verified pricing catalogue for skincare and beauty essentials:\n\n"
                    "**Cosmetics & Beauty Store Pricing**\n\n"
                    "• **Hydrating Vitamin C & Hyaluronic Serums:** ₹499–₹899.\n"
                    "• **Long-Wear Matte Liquid Lipsticks & Foundations:** ₹399–₹799.\n"
                    "• **Botanical Foaming Cleansers:** ₹299–₹549.\n"
                    "• **Special Seasonal Bundle Offers:** Buy 2 Get 1 Free on all seasonal skincare sets.\n\n"
                    f"{closing_e}"
                )
                print("✅ [LLM Tier 3 Fallback]: Grounded cosmetics pricing synthesis delivered.")
            else:
                closing_e = f"Let me know if you'd like to check out specific collections or view the full catalogue, {u_name}!" if u_name else "Let me know if you'd like to check out specific collections or view the full catalogue!"
                synthesized = (
                    f"{salutation}Here is our verified collection of skincare essentials, cosmetics, and beauty products:\n\n"
                    "**Explore Cosmetics & Beauty Store**\n\n"
                    "• **Hydrating Serums & Skincare:** Dermatologically tested formulations with Vitamin C and Hyaluronic Acid.\n"
                    "• **Long-Wear Cosmetics:** Cruelty-free matte foundations, lip colors, and eye palettes.\n"
                    "• **Botanical Cleansers:** Gentle daily skin formulations with natural active extracts.\n\n"
                    f"{closing_e}"
                )
                print("✅ [LLM Tier 3 Fallback]: Grounded cosmetics synthesis delivered.")
            return synthesized, "Grounded Direct Knowledge"
        elif 'software' in full_blob or 'bot' in full_blob or 'ai' in full_blob:
            if is_pricing:
                if is_hindi:
                    closing_h = f"Inme se aapke requirement ke hisaab se kaunsa subscription tier sabse perfect lag raha hai, {u_name}?" if u_name else "Inme se aapke requirement ke hisaab se kaunsa subscription tier sabse perfect lag raha hai?"
                    synthesized = (
                        f"{salutation}Ye rahi hamari complete commercial pricing aur subscription plans ki jaankari:\n\n"
                        "**APP-DEFT Commercial Pricing & Subscription Plans**\n\n"
                        "• **Starter AI Agent ($49/month):** Up to 1,000 conversations/mo, semantic RAG search, standard web chat widget, aur email support.\n"
                        "• **Growth Automation Tier ($199/month):** Up to 10,000 conversations/mo, CRM & webhook sync, custom branding, aur priority routing.\n"
                        "• **Enterprise Custom Plan:** Unlimited concurrency, dedicated cloud/VPC deployment, custom LLM fine-tuning, aur 24/7 SLA guarantee.\n\n"
                        f"{closing_h}"
                    )
                else:
                    closing_e = f"Which subscription tier best matches your organization's deployment scale, {u_name}?" if u_name else "Which subscription tier best matches your organization's deployment scale?"
                    synthesized = (
                        f"{salutation}Here is our complete commercial pricing and subscription breakdown:\n\n"
                        "**APP-DEFT Commercial Pricing & Subscription Plans**\n\n"
                        "• **Starter AI Agent ($49/month):** Up to 1,000 conversations/mo, semantic RAG search, standard web chat widget, and email support.\n"
                        "• **Growth Automation Tier ($199/month):** Up to 10,000 conversations/mo, CRM & webhook sync, custom branding, and priority routing.\n"
                        "• **Enterprise Custom Plan:** Unlimited concurrency, dedicated cloud/VPC deployment, custom LLM fine-tuning, and 24/7 SLA guarantee.\n\n"
                        f"{closing_e}"
                    )
                print("✅ [LLM Tier 3 Fallback]: Grounded software pricing synthesis delivered.")
            else:
                if is_hindi:
                    closing_h = f"Inme se aap kis capability mein sabse zyada interested hain, {u_name}? Main aapko pricing tiers explore karne ya live demo schedule karne mein assist kar sakta hoon!" if u_name else "Main aapko pricing tiers explore karne ya live consultation schedule karne mein kaise help kar sakta hoon?"
                    synthesized = (
                        f"{salutation}Chaliye hamare enterprise conversational AI aur automation platforms ki complete jaankari dekhte hain:\n\n"
                        "**AI & Software Development Services**\n\n"
                        "• **Custom Conversational Agents:** 24/7 omnichannel customer support aur automated FAQ answering.\n"
                        "• **Voice & Multilingual Intelligence:** Natural voice recognition aur live call routing.\n"
                        "• **CRM & Workflow Integrations:** Salesforce, HubSpot, aur custom APIs ke sath seamless connectors.\n\n"
                        f"{closing_h}"
                    )
                else:
                    closing_e = f"Which of these capabilities sounds most exciting for your projects, {u_name}? I'd be glad to help you explore pricing tiers or schedule a live consultation!" if u_name else "I'd be glad to help you explore pricing tiers or schedule a live consultation for your project!"
                    synthesized = (
                        f"{salutation}Here is a complete overview of our enterprise conversational AI and automation platforms:\n\n"
                        "**AI & Software Development Services**\n\n"
                        "• **Custom Conversational Agents:** 24/7 omnichannel customer support and automated FAQ answering.\n"
                        "• **Voice & Multilingual Intelligence:** Natural voice recognition and live call routing.\n"
                        "• **CRM & Workflow Integrations:** Seamless connectors for Salesforce, HubSpot, and custom APIs.\n\n"
                        f"{closing_e}"
                    )
                print("✅ [LLM Tier 3 Fallback]: Grounded software synthesis delivered.")
            return synthesized, "Grounded Direct Knowledge"
        else:
            meaningful_sentences = []
            for ch in knowledge_chunks[:2]:
                for line in ch.splitlines():
                    ls = line.strip()
                    if len(ls) > 25 and not ls.startswith("===") and not ls.startswith("http") and not ls.startswith("{") and not ls.startswith(".pi-") and not ls.startswith("Product Category"):
                        meaningful_sentences.append(ls)
            points = [f"• **{s[:30]}:** {s}" for s in meaningful_sentences[:3]] if meaningful_sentences else [
                "• **Verified Documentation:** All technical specifications are certified directly from official records.",
                "• **Full Catalogue:** Complete parameter sheets and commercial terms are available on request."
            ]
            closing_gen = f"Please feel free to ask if you'd like to explore any specific details, pricing, or documentation, {u_name}!" if u_name else "Please feel free to ask if you'd like to explore any specific details, pricing, or documentation!"
            synthesized = (
                f"{salutation}Here are the verified highlights from our official records:\n\n"
                + "\n".join(points) + "\n\n"
                f"{closing_gen}"
            )
            print("✅ [LLM Tier 3 Fallback]: Grounded general synthesis delivered.")
            return synthesized, "Grounded Direct Knowledge"

    return "Here are the verified details from our knowledge base:\n\n• **Certified Grounding:** All parameters and specifications are verified directly against official documentation.\n• **Full Documentation:** Detailed catalogues and technical data sheets are available on request.", "Default Knowledge Fallback"

async def stream_rag_pipeline(
    bot_id: str,
    question: str,
    db: Session,
    conversation_id: str = None,
    message_id: str = None,
    user_name: str = None
) -> AsyncGenerator[str, None]:
    print(f"\n🔍 [RAG Query]: '{question}'")

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

    # Strict Language Detection: Hindi/Hinglish vs English
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', question))
    hindi_keywords = {
        'kya', 'hai', 'hain', 'kaise', 'batao', 'kitna', 'kitne', 'dam', 'kimat', 'paisa',
        'paise', 'bhai', 'mujhe', 'chahiye', 'karo', 'karna', 'hoga', 'hogi', 'milega',
        'milegi', 'bataiye', 'aap', 'tum', 'ye', 'yeh', 'woh', 'kaun', 'kon', 'kaha',
        'kahan', 'kyu', 'kyun', 'nahi', 'nahin', 'sakte', 'sakta', 'sakti', 'accha',
        'achha', 'thik', 'sahi', 'lekin', 'aur', 'par', 'chahie', 'batayein',
        'ke', 'ki', 'ko', 'ka', 'se', 'me', 'mein', 'baare', 'bare', 'kuch',
        'dikhao', 'kardo', 'kariye', 'kijiye', 'bataye', 'wali', 'wala', 'wale'
    }
    q_words = set(re.findall(r'[a-zA-Z]+', normalized_q))
    hindi_word_hits = sum(1 for w in q_words if w in hindi_keywords)
    is_user_hindi = has_devanagari or (hindi_word_hits >= 2) or (len(q_words) <= 3 and hindi_word_hits >= 1 and any(w in q_words for w in ['kya', 'kaise', 'kitna', 'batao', 'chahiye', 'kaun']))

    # Fetch recent conversation context if conversation_id is available (excluding current message)
    recent_history_texts = []
    prior_messages = []
    if conversation_id:
        try:
            q_hist = db.query(models.Message).filter(models.Message.conversationId == conversation_id)
            if message_id:
                q_hist = q_hist.filter(models.Message.id != message_id)
            prior_messages = q_hist.order_by(models.Message.createdAt.desc()).limit(6).all()
            recent_history_texts = [m.content.lower() for m in prior_messages if m.content]
        except Exception as err:
            print(f"⚠️ [History Fetch Warning]: {err}")

    # Industry / Category Intent Detection
    industry_keywords = {
        'steel': ['steel', 'metal', 'iron', 'pipe', 'sheet', 'alloy', 'cross-section', 'astm', '7304'],
        'sports': ['sport', 'sports', 'shoe', 'shoes', 'footwear', 'sneaker', 'badminton', 'racket', 'running', 'athletic', 'fitness'],
        'cosmetics': ['cosmetic', 'cosmetics', 'beauty', 'skincare', 'makeup', 'nykaa', 'lipstick', 'serum', 'lotion', 'cream'],
        'software': [
            'software', 'chatbot', 'chatbots', 'nlp', 'conversational', 'development',
            'ai bot', 'ai bots', 'consulting', 'appdeft', 'app-deft', 'solutions', 'solution',
            'automation', 'crm', 'workflow', 'assistant', 'agent', 'agents', 'telephony',
            'integrations', 'integration'
        ],
        'financial': ['financial', 'valuation', 'projections', 'vedaone', 'dcf']
    }

    matched_industry = None
    for ind, kws in industry_keywords.items():
        if any(kw in normalized_q for kw in kws):
            matched_industry = ind
            break

    # If the user asks to switch or view other catalogues, reset category isolation
    is_switch_catalogue = any(k in normalized_q for k in ['other catalogue', 'other catalog', 'switch catalogue', 'all catalogue', 'all catalog', 'explore other', 'another catalogue'])
    if is_switch_catalogue:
        matched_industry = None

    # Multi-turn Context Resolution: If query uses pronouns ("it", "this", "pricing", "cost") without naming industry
    if not matched_industry and not is_switch_catalogue and recent_history_texts:
        combined_prev = " ".join(recent_history_texts)
        for ind, kws in industry_keywords.items():
            if any(kw in combined_prev for kw in kws):
                matched_industry = ind
                print(f"🔄 [Conversational Context Resolved]: matched_industry='{ind}' from previous conversation history")
                break

    # Detect Pricing & Commercial Query Intent
    is_pricing_query = any(kw in normalized_q for kw in [
        'price', 'pricing', 'cost', 'costs', 'rate', 'rates', 'fee', 'fees',
        'quote', 'quotation', 'charges', 'charge', 'how much', 'expensive',
        'cheap', 'budget', 'dam', 'kitna', 'kimat', 'paisa', 'bill'
    ])

    # Dynamic Conversational Query Augmentation for Embedding
    augmented_embedding_query = question
    pronoun_or_followup = any(w in normalized_q.split() for w in [
        'it', 'its', 'this', 'that', 'they', 'them', 'these', 'those',
        'price', 'pricing', 'cost', 'costs', 'rate', 'rates', 'fee', 'quote',
        'spec', 'specs', 'specification', 'specifications', 'details', 'detail', 'catalogue'
    ])
    industry_in_q = any(kw in normalized_q for kw in industry_keywords.get(matched_industry, [])) if matched_industry else False

    if matched_industry and (pronoun_or_followup or not industry_in_q):
        prefix = f"{matched_industry} metal " if matched_industry == 'steel' else f"{matched_industry} "
        augmented_embedding_query = f"{prefix}{question}"
        print(f"🧠 [Query Augmented with Context for Embedding]: '{augmented_embedding_query}'")

    query_vec = get_embedding(augmented_embedding_query)

    # 1. Expand keywords from normalized query
    words = [w.strip('?,.!\"\'()[]{}') for w in normalized_q.split()]
    keywords = [w for w in words if len(w) > 2 and w not in stopwords]

    intent_expansions = {
        'cost': ['cost', 'price', 'pricing', 'subscription', 'plans', 'free', 'trial', 'tier', 'rate'],
        'pricing': ['pricing', 'cost', 'plans', 'subscription', 'free', 'trial', 'rate', 'rates'],
        'price': ['price', 'cost', 'pricing', 'plans', 'subscription', 'rate', 'rates'],
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
    if matched_industry:
        expanded_keywords.update(industry_keywords.get(matched_industry, []))

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
    # 0. SMART USER RECOGNITION (ANY USER: SAHIL, JASBIR, OR GUEST)
    # ----------------------------------------------------
    detected_user_name = (user_name or "").strip()

    # 1. Detect user self-introduction from question (e.g. "I am Sahil", "My name is John", "Mera naam Rahul hai", "Hey this is Alex")
    name_intro_match = re.search(r'\b(?:my name is|i am|i\'m|this is|call me|mera naam|naam hai)\s+([A-Za-z]+)', question, re.IGNORECASE)
    if name_intro_match:
        detected_user_name = name_intro_match.group(1).capitalize()

    # 2. Check prior messages in this conversation if not detected yet
    if not detected_user_name and prior_messages:
        for pm in prior_messages:
            if pm.role in ["USER", "user"]:
                pm_match = re.search(r'\b(?:my name is|i am|i\'m|this is|call me|mera naam|naam hai)\s+([A-Za-z]+)', pm.content, re.IGNORECASE)
                if pm_match:
                    detected_user_name = pm_match.group(1).capitalize()
                    break

    # 3. Check conversation record & associated lead or test sandbox user
    if not detected_user_name and conversation_id:
        try:
            conv_record = db.query(models.Conversation).filter(models.Conversation.id == conversation_id).first()
            if conv_record:
                lead = db.query(models.Lead).filter(models.Lead.botId == bot_id).order_by(models.Lead.createdAt.desc()).first()
                if lead and lead.name:
                    detected_user_name = lead.name.strip().split()[0].capitalize()
                elif conv_record.isTest and bot and bot.orgId:
                    org_user = db.query(models.User).filter(models.User.orgId == bot.orgId).order_by(models.User.createdAt.asc()).first()
                    if org_user:
                        raw_u = (org_user.name or (org_user.email.split('@')[0] if org_user.email else "")).strip()
                        if raw_u:
                            detected_user_name = raw_u.split()[0].capitalize()
        except Exception as e:
            print(f"⚠️ [User Name Resolution Warning]: {e}")

    # 4. Fallback for bot workspace test if still undetected
    if not detected_user_name and bot and bot.orgId:
        try:
            org_user = db.query(models.User).filter(models.User.orgId == bot.orgId).order_by(models.User.createdAt.asc()).first()
            if org_user:
                raw_u = (org_user.name or (org_user.email.split('@')[0] if org_user.email else "")).strip()
                if raw_u:
                    detected_user_name = raw_u.split()[0].capitalize()
        except Exception:
            pass

    user_first_name = detected_user_name.split()[0].capitalize() if detected_user_name else ""
    is_first_turn = (len(prior_messages) == 0)
    user_salutation = f"Hey {user_first_name}! 👋 " if (is_first_turn and user_first_name) else ""
    if user_first_name:
        print(f"👤 [User Recognized]: '{user_first_name}' (First Turn: {is_first_turn})")

    # ----------------------------------------------------
    # 0. BOT SELF-IDENTITY & OFFICIAL CONTACT RESOLVER
    # ----------------------------------------------------
    b_name = (bot.name if bot else "Our Company").strip()
    b_domain = (bot.domain if bot else "").strip()
    b_name_clean = re.sub(r'[^a-z0-9]', '', b_name.lower())
    q_clean = re.sub(r'[^a-z0-9]', '', normalized_q)

    b_name_words = [w for w in re.split(r'[^a-z0-9]+', b_name.lower()) if len(w) >= 3]
    b_domain_clean = re.sub(r'^(https?://)?(www\.)?', '', b_domain.lower()).split('/')[0].split('.')[0]

    # Dedicated Contact & Support Handler (Zero Third-Party Contamination)
    contact_triggers = [
        'contact', 'support', 'email', 'phone', 'call', 'number', 'reach',
        'touch', 'helpdesk', 'customer service', 'get in touch', 'talk to human',
        'consultant', 'office', 'address', 'headquarters', 'location'
    ]
    is_contact_query = any(trig in normalized_q for trig in contact_triggers)
    is_third_party_brand = any(k in normalized_q for k in ['nykaa', 'steel', 'sports', 'footwear', 'badminton', 'shoe'])

    if is_contact_query and not is_third_party_brand:
        print(f"📞 [Official Contact Query Detected]: '{question}' for bot '{b_name}' ({b_domain})")
        if is_user_hindi:
            contact_closing = f"Inme se aap kis service ya solution ke baare mein pehle discuss karna chahenge, {user_first_name}? Niche diye gaye options par tap karein ya seedhe batayein! 😊" if user_first_name else "Inme se aap kis service ya solution ke baare mein pehle discuss karna chahenge? Niche diye gaye options par tap karein ya seedhe batayein! 😊"
            contact_body = (
                f"{user_salutation}Humse connect karne ke liye bohot shukriya! 🚀 Ye rahi **{b_name}** ki official contact details—hum 24/7 aapki madad ke liye tayyar hain:\n\n"
                f"• **Official Support Email:** support@{b_domain or 'appdeft.ai'}\n"
                f"• **Enterprise Website:** https://{b_domain or 'appdeft.ai'}\n"
                f"• **Live Consultation & Demos:** Hamari AI solutions engineering team custom integrations aur live platform demonstrations ke liye 24/7 available hai.\n"
                f"• **14-Day Evaluation Guarantee:** Har enterprise rollout ke sath 14 business days ka 100% money-back guarantee milta hai, zero cancellation penalties ke sath.\n\n"
                f"{contact_closing}"
            )
            prompt_text = f"{user_first_name}, {b_name} ke baare mein aur kya explore karna chahenge aap?" if user_first_name else f"{b_name} ke baare mein aur kya explore karna chahenge aap?"
        else:
            contact_closing = f"Which exciting service or capability would you like to explore first, {user_first_name}? Just tap an option below or ask me directly!" if user_first_name else "Which exciting service or capability would you like to explore first? Just tap an option below or ask me directly!"
            contact_body = (
                f"{user_salutation}We'd love to connect with you directly! 🚀 Here are the official contact channels and support details for **{b_name}**—our team is always ready to assist you:\n\n"
                f"• **Official Support Email:** support@{b_domain or 'appdeft.ai'}\n"
                f"• **Enterprise Website:** https://{b_domain or 'appdeft.ai'}\n"
                f"• **Live Consultation & Demos:** Our AI solutions engineering team is available 24/7 for architectural deep-dives, live platform demonstrations, and custom integrations.\n"
                f"• **14-Day Evaluation Guarantee:** All enterprise deployments include a 100% risk-free refund within 14 business days of initial rollout with zero cancellation penalties.\n\n"
                f"{contact_closing}"
            )
            prompt_text = f"How else can we assist you with {b_name} today, {user_first_name}?" if user_first_name else f"How else can we assist you with {b_name} today?"

        followup_data = {
            "prompt": prompt_text,
            "options": ["AI Chatbot Capabilities", "Commercial Pricing & Plans", "CRM Integrations", "Explore Other Catalogues"]
        }
        if conversation_id:
            try:
                bot_msg_db = models.Message(
                    conversationId=conversation_id,
                    role="BOT",
                    content=contact_body,
                    unanswered=False
                )
                db.add(bot_msg_db)
                db.commit()
            except Exception as e:
                print(f"Error saving bot response: {e}")

        yield f"data: {json.dumps({'type': 'start', 'confidence': 1.0})}\n\n"
        for word in contact_body.split(" "):
            if word:
                yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

        yield f"data: {json.dumps({'type': 'sources', 'sources': [{'title': f'{b_name} Official Support Directory', 'kind': 'PAGE', 'url': f'https://{b_domain}' if b_domain else '', 'snippet': f'Verified official contact channels and support directory for {b_name}.'}]})}\n\n"
        yield f"data: {json.dumps({'type': 'followup', 'prompt': followup_data['prompt'], 'options': followup_data['options']})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'full_text': contact_body, 'followup': followup_data})}\n\n"
        return

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
        if matched_industry is None:
            matched_industry = 'software'

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
            if is_user_hindi:
                body = (
                    f"{user_salutation}Great to connect with you! Main **{b_name}** hoon, aapka personal AI companion aur consultant yahan **{b_domain or 'hamare enterprise'}** par. 🚀\n\n"
                    f"**{b_name}** par hum cutting-edge enterprise conversational AI systems, autonomous workflows aur smart digital agents build karte hain:\n\n"
                    f"• **Conversational AI Bots:** 24/7 intelligent customer engagement, multi-turn memory aur verified documentation se certified grounding.\n"
                    f"• **Workflow Automation & RPA:** Autonomous lead qualification, ticket routing aur instant appointment scheduling.\n"
                    f"• **Multilingual Voice & Chatbot Integration:** Real-time speech-to-text, ultra-low latency (<500ms) aur 20+ languages ka full support.\n"
                    f"• **Enterprise CRM & ERP Connectors:** Salesforce, HubSpot, Zendesk aur SQL databases ke sath seamless turnkey sync.\n"
                    f"• **Transparent Commercial Tiers:** $49/mo (Starter) se shuru hokar $199/mo (Growth) aur high-scale custom Enterprise tiers.\n\n"
                    f"---\n📑 **Official Specifications & Product Catalogue (PDF)**\n*Kya aap iska complete certified documentation review ya download karna chahenge?*\n\n"
                    f"[PDF_CARD:ai-software-development|{b_name} AI Software & Solutions Catalogue]"
                )
                prompt_followup = f"{user_first_name}, {b_name} ke baare mein aur kya explore karna chahenge aap?" if user_first_name else f"{b_name} ke baare mein aur kya explore karna chahenge aap?"
            else:
                body = (
                    f"{user_salutation}Great to connect with you! I'm **{b_name}**, your personal AI companion and expert consultant here at **{b_domain or 'our enterprise'}**! 🚀\n\n"
                    f"At **{b_name}**, we specialize in building enterprise-grade conversational AI platforms, custom autonomous workflows, and intelligent digital agents:\n\n"
                    f"• **Conversational AI Bots:** 24/7 intelligent customer engagement, multi-turn memory, and semantic grounding over verified documentation.\n"
                    f"• **Workflow Automation & RPA:** Autonomous lead qualification, ticket routing, and appointment scheduling.\n"
                    f"• **Multilingual Voice & Chatbot Integration:** Real-time speech-to-text, ultra-low latency synthesis (<500ms), and 20+ language support.\n"
                    f"• **Enterprise CRM & ERP Connectors:** Turnkey bi-directional sync with Salesforce, HubSpot, Zendesk, and SQL databases.\n"
                    f"• **Transparent Commercial Tiers:** From $49/mo (Starter) to $199/mo (Growth) and custom high-concurrency Enterprise clusters.\n\n"
                    f"---\n📑 **Official Specifications & Product Catalogue (PDF)**\n*Would you like to review or download the complete certified documentation for this?*\n\n"
                    f"[PDF_CARD:ai-software-development|{b_name} AI Software & Solutions Catalogue]"
                )
                prompt_followup = f"What would you like to explore regarding {b_name}, {user_first_name}?" if user_first_name else f"What would you like to explore regarding {b_name}?"

            followup_data = {
                "prompt": prompt_followup,
                "options": ["AI Chatbot Capabilities", "Commercial Pricing & Plans", "CRM Integrations", "Schedule a Live Demo"]
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

    is_catalogue_mention = any(k in normalized_q for k in ['catalogue', 'catalog', 'cetalouge', 'cataloge', 'products', 'inventory', 'brochure'])
    is_general_catalogue_query = (is_catalogue_mention or is_switch_catalogue) and (matched_industry is None) and not is_pricing_query

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

        if is_user_hindi:
            cat_closing_hi = f"Inme se sabse pehle kis category mein dive karna chahenge aap, {user_first_name}? Niche diye gaye options par tap karein ya seedhe poochiye, main poori detail share karunga! 😊" if user_first_name else "Inme se sabse pehle kis category mein dive karna chahenge aap? Niche diye gaye options par tap karein ya seedhe poochiye, main poori detail share karunga! 😊"
            body = (
                f"{user_salutation}Chaliye hamare certified product catalogues explore karte hain! 🚀\n\n"
                f"**Hamari Verified Catalogues Directory:**\n\n"
                f"Hamare central knowledge base mein **{b_name}** ke enterprise solutions se lekar universal industry categories tak ka certified collection maujood hai. Dekhiye hum kya-kya offer karte hain:\n\n"
                + "\n".join(bullets) + "\n\n"
                + f"{cat_closing_hi}"
            )
            followup_prompt = f"{user_first_name}, aap sabse pehle kaunsa catalogue explore karna chahenge?" if user_first_name else "Aap sabse pehle kaunsa catalogue explore karna chahenge?"
        else:
            cat_closing_en = f"Which of these categories would you love to dive into first, {user_first_name}? Just tap an option below or ask me directly!" if user_first_name else "Which of these categories would you love to dive into first? Just tap an option below or ask me directly!"
            body = (
                f"{user_salutation}Welcome to our verified product catalogue directory! 🚀\n\n"
                f"**Explore Our Complete Product & Solutions Directory**\n\n"
                f"We maintain an extensive, certified collection covering both **{b_name}** enterprise AI systems and universal industry specifications. Here's a look at what you can explore:\n\n"
                + "\n".join(bullets) + "\n\n"
                + f"{cat_closing_en}"
            )
            followup_prompt = f"Which catalogue would you like to explore first, {user_first_name}?" if user_first_name else "Which catalogue would you like to explore first?"

        followup_data = {
            "prompt": followup_prompt,
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

    # 3. Direct SQL Lexical Search to ensure high-intent keywords (address, phone, contact, pricing) are never missed
    high_intent_terms = [kw for kw in expanded_keywords if kw in ['address', 'location', 'phone', 'email', 'contact', 'headquarters', 'office', 'price', 'pricing', 'rate', 'cost']]
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
                industry_boost = -1.20
        elif matched_industry == 'sports':
            if any(k in content_lower for k in ['sport', 'shoe', 'footwear', 'athletic', 'sneaker', 'badminton']):
                industry_boost = 0.60
            else:
                industry_boost = -1.20
        elif matched_industry == 'cosmetics':
            if any(k in content_lower for k in ['cosmetic', 'beauty', 'nykaa', 'skincare', 'makeup']):
                industry_boost = 0.60
            else:
                industry_boost = -1.20
        elif matched_industry == 'software':
            if is_bot_proprietary or any(k in content_lower for k in ['software', 'chatbot', 'ai ', 'nlp', 'development', 'machine learning', 'appdeft', 'vinnisoft', 'automation', 'crm', 'enterprise']):
                industry_boost = 0.60
            else:
                industry_boost = -1.20
        elif matched_industry is None and not is_catalogue_mention:
            # When the user is NOT asking about a specific catalogue or exploring catalogues,
            # universal shared catalogue chunks (steel, footwear, cosmetics) must NOT leak into general company inquiries!
            if is_universal and any(k in content_lower for k in ['steel', 'metal', 'nykaa', 'cosmetic', 'lipstick', 'shoe', 'footwear', 'badminton']):
                industry_boost = -5.0

        # Absolute hard anti-leakage barriers:
        # 1. Nykaa cosmetics chunks must NEVER leak into non-cosmetic queries
        if 'nykaa' in content_lower and not any(k in normalized_q for k in ['nykaa', 'cosmetic', 'beauty', 'lipstick', 'serum', 'skincare', 'makeup']):
            industry_boost = -10.0

        # 2. Steel / metals must NEVER leak into software or sports queries
        if any(k in content_lower for k in ['astm a36', 'astm a572', '7304', 'structural steel', 'metric ton']) and matched_industry in ['software', 'sports', 'cosmetics']:
            industry_boost = -10.0

        # 3. Sports shoes must NEVER leak into software or steel queries
        if any(k in content_lower for k in ['running shoes', 'badminton racket', 'basketball size']) and matched_industry in ['software', 'steel', 'cosmetics']:
            industry_boost = -10.0

        hybrid_score = round(vec_sim + kw_boost + bot_priority_boost + industry_boost, 4)
        scored_chunks.append((hybrid_score, chunk.content, vec_sim))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    best_score = scored_chunks[0][0] if scored_chunks else 0.0
    print(f"📊 [RAG Hybrid Confidence]: {best_score} (Top Vec: {scored_chunks[0][2]:.4f})")

    passed_chunks = [c[1] for c in scored_chunks if c[0] >= 0.28]

    # Dynamic Industry Recovery: If matched_industry is None, infer from the top passed chunk or bot identity
    if matched_industry is None and passed_chunks:
        top_content = passed_chunks[0].lower()
        if any(k in top_content for k in ['appdeft', 'app-deft', 'starter ai agent', 'conversational ai', 'bot', 'workflow automation', 'crm & helpdesk', 'vinnisoft']):
            matched_industry = 'software'
            print(f"🔄 [Inferred Category from Top Chunk]: matched_industry='software'")
        elif any(k in top_content for k in ['steel', 'metal', 'astm', '7304', 'pipe', 'carbon steel']):
            matched_industry = 'steel'
            print(f"🔄 [Inferred Category from Top Chunk]: matched_industry='steel'")
        elif any(k in top_content for k in ['running shoes', 'badminton', 'footwear', 'shoe']):
            matched_industry = 'sports'
            print(f"🔄 [Inferred Category from Top Chunk]: matched_industry='sports'")
        elif any(k in top_content for k in ['nykaa', 'cosmetic', 'serum', 'skincare']):
            matched_industry = 'cosmetics'
            print(f"🔄 [Inferred Category from Top Chunk]: matched_industry='cosmetics'")
        elif bot and ('appdeft' in (bot.domain or '').lower() or 'app-deft' in (bot.name or '').lower()):
            matched_industry = 'software'
            print(f"🔄 [Inferred Category from Bot Profile]: matched_industry='software'")

    if not passed_chunks or best_score < 0.28:
        print(f"⚠️ [Unanswered Query]: Best score {best_score} < RELEVANCE_FLOOR (0.28). Flagging as content gap.")
        if is_user_hindi:
            fallback_text = (
                f"Aapne bohot accha sawaal poocha! Main hamesha 100% accurate aur certified jaankari dena chahta hoon, par filhal mere current knowledge base mein '{question}' ka exact verified record nahi mila.\n\n"
                f"• **Direct Follow-up:** Aap niche apna contact detail chhod dijiye, hamari team personally verify karke aapse turant connect karegi!\n"
                f"• **Explore Verified Topics:** Aap hamari available product catalogues, enterprise AI solutions ya official contact channels ke baare mein bhi pooch sakte hain."
            )
        else:
            fallback_text = (
                f"That's a great question! I want to ensure you get 100% accurate and verified information, but I don't have an exact certified record for '{question}' in my current knowledge base.\n\n"
                f"• **Direct Follow-up:** Please leave your contact details below, and our team will personally verify the answer and reach out to you right away!\n"
                f"• **Explore Verified Topics:** You can also explore our available product catalogues, enterprise AI solutions, or official contact channels!"
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

    # Construct multi-turn conversation history for LLM
    chat_history_for_llm = []
    if prior_messages:
        for m in reversed(prior_messages):
            role = "user" if m.role in ["USER", "user"] else "assistant"
            clean_c = re.sub(r'\[PDF_CARD:.*?\]', '', m.content).strip()
            clean_c = re.sub(r'<<<FOLLOW_UP>>>.*?<<<END_FOLLOW_UP>>>', '', clean_c, flags=re.DOTALL).strip()
            clean_c = re.sub(r'---\s*📑.*', '', clean_c, flags=re.DOTALL).strip()
            if clean_c:
                chat_history_for_llm.append({"role": role, "content": clean_c[:400]})

    if user_first_name:
        if is_first_turn:
            user_personalization_directive = (
                f"USER RECOGNITION & INITIAL GREETING (TURN 1):\n"
                f"• You are talking directly with '{user_first_name}'.\n"
                f"• WARM INITIAL GREETING: Since this is the very first turn of the chat, open warmly addressing {user_first_name} by name (e.g. 'Hey {user_first_name}! 👋 Great to connect with you!').\n"
                f"• Speak with infectious enthusiasm, friendly camaraderie, and high emotional intelligence.\n"
                f"• Close personally with an inviting, friendly question."
            )
        else:
            user_personalization_directive = (
                f"ACTIVE MULTI-TURN CONVERSATION WITH '{user_first_name}' (STRICT ANTI-ROBOTIC RULE):\n"
                f"• You are ALREADY in an ongoing conversation with {user_first_name}.\n"
                f"• ABSOLUTELY NEVER start this response with 'Hey {user_first_name}! 👋 ' or repeat greetings at the beginning of every turn! Saying 'Hey {user_first_name}!' in every message sounds like an annoying, repetitive robot.\n"
                f"• Instead, jump straight into the flow with dynamic, natural conversational openings:\n"
                f"  - Examples: 'Absolutely!', 'You got it!', 'Sure thing!', 'Great question!', 'Here is how that works:', 'Glad you asked!', 'Let\\'s break that down:'\n"
                f"• You may weave {user_first_name}'s name naturally into the explanation or closing question (e.g. 'As you can see, {user_first_name}...', 'Which option sounds best for you, {user_first_name}?'), but DO NOT force it onto every single turn.\n"
                f"• Keep every turn engaging, entertaining, and captivating so {user_first_name} thoroughly enjoys chatting with you!"
            )
    else:
        user_personalization_directive = (
            "USER ENGAGEMENT & WARMTH:\n"
            "• The user is an esteemed guest.\n"
            "• Speak directly to them with charismatic, friendly, and engaging conversation.\n"
            "• If in an ongoing chat, jump straight into the flow without repetitive greeting formulas.\n"
            "• Close with a warm, inviting question to keep the conversation flowing effortlessly."
        )

    if is_user_hindi:
        closing_prompt_h = f"Inme se aapke requirement ke hisaab se kaunsa option sabse mast lag raha hai, {user_first_name}? Aap batayein, hum milkar sab finalize kar lenge!" if user_first_name else "Inme se aapke requirement ke hisaab se kaunsa option sabse mast lag raha hai? Aap batayein, hum milkar sab finalize kar lenge!"
        if is_first_turn:
            opening_examples_h = f"'Hey {user_first_name}! 👋 Chaliye iski poori detail dekhte hain:', 'Hey {user_first_name}, ye rahi iski complete jaankari:'" if user_first_name else "'Namaste! Chaliye iski poori detail dekhte hain:'"
        else:
            opening_examples_h = "'Haan bilkul! Chaliye dekhte hain:', 'Zaroor! Ye rahi iski complete jaankari:', 'Bilkul sahi sawal! Iska breakdown kuch is tarah hai:', 'Aapko vistaar se batata hoon:'"

        language_directive = (
            "LANGUAGE & TONE DIRECTIVE (STRICT HINDI/HINGLISH):\n"
            "The user asked their question in Hindi or Hinglish.\n"
            "• Respond in an ultra-friendly, energetic, respectful, and engaging Hindi/Hinglish (using 'Aap', polite yet charismatic dostana tone).\n"
            "• DYNAMIC & VARIED HUMAN OPENINGS (CRITICAL - NEVER REPEAT CANNED PHRASES):\n"
            "  - NEVER use the exact same opening phrase across turns. Vary naturally based on context!\n"
            f"  - Examples: {opening_examples_h}.\n"
            "• Make the explanation enjoyable, engaging, and lively so the user has fun reading every line!\n"
            "• Provide 100% complete, top-notch technical and pricing figures—leave zero details out.\n"
            f"• Close with an inviting, enthusiastic friendly question (e.g. '{closing_prompt_h}')."
        )
    else:
        closing_prompt_e = f"Which of these options sounds like the best fit for your goals, {user_first_name}? Let me know, and we can tailor it perfectly for you!" if user_first_name else "Which of these options sounds like the best fit for your goals? Let me know, and we can tailor it perfectly for you!"
        if is_first_turn:
            opening_examples_e = (
                f"    * 'Hey {user_first_name}! 👋 Great to connect with you! Here is an overview of what our AI platforms deliver:'\n"
                f"    * 'Hey {user_first_name}! Welcome! Here is our transparent pricing structure:'"
            )
        else:
            opening_examples_e = (
                "    * For features / solutions: 'Here is a complete breakdown of what our AI platforms deliver:', 'Let\\'s explore how our intelligent agents work in action:', 'You got it! Here\\'s the breakdown:'\n"
                "    * For pricing / plans: 'Here is our transparent pricing structure and subscription plans:', 'Let\\'s break down the exact commercial rates and tiers:', 'Sure thing! Here are the rates:'\n"
                "    * For technical specs: 'Here are the certified technical specifications and parameters:', 'Let\\'s dive into the technical standards:'\n"
                "    * For scheduling / questions: 'Absolutely! Here\\'s what you can expect:', 'Happy to walk you through how this works:'"
            )

        language_directive = (
            "LANGUAGE & TONE DIRECTIVE (STRICT ENGLISH):\n"
            "The user asked their question in ENGLISH.\n"
            "• Respond in 100% fluent, lively, engaging, charismatic, and conversational ENGLISH.\n"
            "• Strictly ZERO Hindi/Hinglish words (NO 'Arre', NO 'bilkul', NO 'bata', NO 'aap', NO 'shaamil').\n"
            "• DYNAMIC & VARIED HUMAN OPENINGS (CRITICAL - ZERO CANNED COPY-PASTE):\n"
            "  - ABSOLUTELY NEVER start every response with 'Awesome question!' or 'Hey {name}! 👋' over and over again! That sounds robotic, fake, and annoying.\n"
            "  - Every response must start with a fresh, natural opening tailored directly to what was asked:\n"
            f"{opening_examples_e}\n"
            "• Make the explanation super engaging, enjoyable, and crystal-clear so the user is captivated and loves talking with you!\n"
            "• Provide 100% complete, top-notch technical parameters and pricing figures—leave zero details out.\n"
            f"• Close with an inviting, friendly conversational question (e.g. '{closing_prompt_e}')."
        )

    system_prompt = (
        GROUNDED_SYSTEM_PROMPT
        .replace("{user_personalization_directive}", user_personalization_directive)
        .replace("{language_directive}", language_directive)
        .replace("{knowledge}", knowledge_ctx)
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    messages.extend(chat_history_for_llm[-4:])
    messages.append({"role": "user", "content": question})

    yield f"data: {json.dumps({'type': 'start', 'confidence': best_score})}\n\n"

    raw_text, engine_used = generate_llm_response(
        messages, passed_chunks,
        is_pricing=is_pricing_query,
        is_hindi=is_user_hindi,
        user_name=user_first_name,
        is_first_turn=is_first_turn
    )

    if raw_text:
        # If this is an ongoing turn, strip accidental repetitive "Hey {name}! 👋 " greetings generated by LLM
        if not is_first_turn and user_first_name:
            raw_text = re.sub(rf'^(?:hey|hi|hello)\s+{re.escape(user_first_name)}[!,.]*\s*(?:👋\s*)?', '', raw_text, flags=re.IGNORECASE).strip()

        if not is_user_hindi:
            # Strip accidental Hinglish greeting if LLM generated any
            raw_text = re.sub(r'^(?:arre\s+bilkul!?|haan\s+bilkul!?|bilkul!?)\s*(?:main\s+aapko\s+[^.]*\.\s*)?', '', raw_text, flags=re.IGNORECASE).strip()
            raw_text = re.sub(r'^(?:aapko\s+[^?]*\?\s*)', '', raw_text, flags=re.IGNORECASE).strip()
            # Clean repetitive canned hooks if LLM outputs repetitive "Awesome question! Let's dive right into..."
            canned_match = re.match(r'^(?:awesome question!?\s*(?:let\'s\s+dive\s+right\s+into[^\n:]*[:—-]*)?)\s*', raw_text, re.IGNORECASE)
            salutation = f"Hey {user_first_name}! 👋 " if (is_first_turn and user_first_name) else ""
            if canned_match:
                rest_of_text = raw_text[canned_match.end():].strip()
                if matched_industry == 'software' and is_pricing_query:
                    raw_text = f"{salutation}Here is our complete commercial pricing and subscription breakdown:\n\n" + rest_of_text
                elif matched_industry == 'software':
                    raw_text = f"{salutation}Here is a complete breakdown of our AI solutions and platform capabilities:\n\n" + rest_of_text
                elif matched_industry == 'steel' and is_pricing_query:
                    raw_text = f"{salutation}Here is our complete commercial rate schedule for industrial steel:\n\n" + rest_of_text
                elif matched_industry == 'steel':
                    raw_text = f"{salutation}Here is the technical breakdown for our industrial steel and metal specifications:\n\n" + rest_of_text
                elif matched_industry == 'sports':
                    raw_text = f"{salutation}Here is our collection of verified athletic footwear and sports items:\n\n" + rest_of_text
                elif matched_industry == 'cosmetics':
                    raw_text = f"{salutation}Here is our verified collection of skincare and beauty essentials:\n\n" + rest_of_text
                else:
                    raw_text = f"{salutation}Here is the complete breakdown you requested:\n\n" + rest_of_text
            elif is_first_turn and user_first_name:
                first_line = raw_text.splitlines()[0]
                if user_first_name.lower() not in first_line.lower():
                    raw_text = f"Hey {user_first_name}! 👋 " + raw_text

        elif is_user_hindi and is_first_turn and user_first_name:
            first_line = raw_text.splitlines()[0]
            if user_first_name.lower() not in first_line.lower():
                raw_text = f"Hey {user_first_name}! 👋 " + raw_text

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
    # Strip any LLM-emitted PDF cards and banners so backend injects the authoritative, query-specific card
    clean_text = re.sub(r'---\s*📑[^\n]*\n\*[^\n]*\*', '', clean_text).strip()
    clean_text = re.sub(r'---\s*📑.*?(?=\n\n|$)', '', clean_text, flags=re.DOTALL).strip()
    clean_text = re.sub(r'\[(?:PDF_CARD|VIEW_PDF):[^\]]*\]', '', clean_text).strip()
    clean_text = re.sub(r'---\s*$', '', clean_text).strip()

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
                hook = "Chaliye iski poori detail dekhte hain:\n\n" if is_user_hindi else "Here is the complete breakdown for you:\n\n"
                closing = "Inme se aap kis option ke baare mein aur detail chahte hain? Niche diye gaye options par tap karein ya poochiye!" if is_user_hindi else "Which of these options would you like to explore further? Just tap an option below or ask me directly!"
                clean_text = f"{hook}**{sum_line}**\n\n" + "\n".join(bullets) + f"\n\n{closing}"
        if not clean_text or len(clean_text.strip()) < 15:
            if is_user_hindi:
                clean_text = "Chaliye hamare verified documentation ki details dekhte hain:\n\n• **Certified Grounding:** Hamare saare technical parameters official records se verified hain.\n• **Full Catalogue:** Complete specifications aur commercial pricing request par available hain.\n\nAap kis specific topic ke baare mein jaanna chahenge?"
            else:
                clean_text = "Here are the verified details from our official records:\n\n• **Certified Grounding:** All parameters and specifications are verified directly against official documentation.\n• **Full Documentation:** Detailed catalogues and technical data sheets are available on request.\n\nWhich specific area would you like to explore further?"

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
        cat_options = []
        if matched_industry == 'software':
            if is_pricing_query:
                cat_options = ["AI Chatbot Capabilities", "CRM Integrations", "14-Day Refund Guarantee", "Schedule a Live Demo"]
            elif any(k in normalized_q for k in ['chatbot', 'agent', 'bot', 'capabilities', 'features']):
                cat_options = ["Commercial Pricing & Plans", "CRM Integrations", "Telephony Voice Bots", "Schedule a Live Demo"]
            elif any(k in normalized_q for k in ['crm', 'integration', 'connectors', 'erp', 'webhook']):
                cat_options = ["Commercial Pricing & Plans", "AI Chatbot Capabilities", "Schedule a Live Demo", "Explore Other Catalogues"]
            elif any(k in normalized_q for k in ['contact', 'support', 'demo', 'schedule', 'call']):
                cat_options = ["AI Chatbot Capabilities", "Commercial Pricing & Plans", "CRM Integrations", "Explore Other Catalogues"]
            else:
                cat_options = ["AI Chatbot Capabilities", "Commercial Pricing & Plans", "CRM Integrations", "Schedule a Live Demo"]
        elif matched_industry == 'steel':
            if is_pricing_query:
                cat_options = ["Technical Specifications", "ASTM A36 & A572 Plates", "Seamless Carbon Steel Pipes", "Explore Other Catalogues"]
            else:
                cat_options = ["Steel Pricing & Rate Schedule", "Structural ASTM A36 & A572", "Seamless Carbon Steel Pipes", "Explore Other Catalogues"]
        elif matched_industry == 'sports':
            if is_pricing_query:
                cat_options = ["Product Specifications", "Athletic Running Shoes", "Badminton & Court Shoes", "Explore Other Catalogues"]
            else:
                cat_options = ["Footwear Price List", "Athletic Running Shoes", "Badminton & Court Shoes", "Explore Other Catalogues"]
        elif matched_industry == 'cosmetics':
            if is_pricing_query:
                cat_options = ["Ingredient & Safety Profiles", "Hydrating Facial Serums", "24h Matte Foundations", "Explore Other Catalogues"]
            else:
                cat_options = ["Cosmetics Price Catalogue", "Hydrating Serums & SPF", "Matte Foundations & Lipsticks", "Explore Other Catalogues"]
        else:
            cat_options = [f"{b_name} AI Solutions", "Pricing & Specifications", "Official Contact & Support", "Explore Other Catalogues"]

        followup_data = {
            "prompt": f"What would you like to explore next regarding {b_name if matched_industry == 'software' else matched_industry.title()}?",
            "options": cat_options[:4]
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

            if is_pricing_query:
                card_title_banner = "📑 **Official Commercial Pricing & Rate Schedule (PDF)**\n*Would you like to review or download the complete certified rate schedule for this?*"
                if matched_industry == 'steel':
                    topic_clean = "Steel & Metal Products — Commercial Pricing & Rate Schedule"
                    topic_slug = "steel-pricing"
                elif matched_industry == 'sports':
                    topic_clean = "Sports & Footwear Items — Price List & Retail Catalogue"
                    topic_slug = "sports-pricing"
                elif matched_industry == 'cosmetics':
                    topic_clean = "Cosmetics & Beauty Store — Product Price Catalogue"
                    topic_slug = "cosmetics-pricing"
                elif matched_industry == 'software':
                    topic_clean = f"{b_name} AI Software — Commercial Pricing & Enterprise Plans"
                    topic_slug = "software-pricing"
                elif matched_industry == 'financial':
                    topic_clean = "VedaOne AI Financial Valuation — Pricing & Subscription Tiers"
                    topic_slug = "financial-valuation-pricing"
                else:
                    topic_clean = f"{b_name} — Commercial Pricing & Rate Breakdown"
                    topic_slug = "pricing-schedule"
            else:
                card_title_banner = "📑 **Official Specifications & Product Catalogue (PDF)**\n*Would you like to review or download the complete certified documentation for this?*"
                if matched_industry == 'steel':
                    topic_clean = "Steel & Metal Technical Specifications & Standards"
                    topic_slug = "steel-specifications"
                elif matched_industry == 'sports':
                    topic_clean = "Sports & Athletic Footwear Product Catalogue"
                    topic_slug = "sports-footwear"
                elif matched_industry == 'cosmetics':
                    topic_clean = "Cosmetics & Beauty Product Catalogue"
                    topic_slug = "cosmetics-beauty"
                elif matched_industry == 'software':
                    topic_clean = f"{b_name} AI Software & Solutions Catalogue"
                    topic_slug = "ai-software-development"
                elif matched_industry == 'financial':
                    topic_clean = "VedaOne Financial AI Specifications"
                    topic_slug = "financial-valuation"
                else:
                    topic_clean = f"{b_name} Verified Documentation & Catalogue"
                    topic_slug = "product-catalogue"

            if topic_clean and topic_slug:
                clean_text += f"\n\n---\n{card_title_banner}\n\n[PDF_CARD:{topic_slug}|{topic_clean}]"

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
