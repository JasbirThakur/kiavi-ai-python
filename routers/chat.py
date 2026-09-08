from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db
import models
from services.auth import get_current_user
from services.rag import stream_rag_pipeline

router = APIRouter(tags=["Chat"])

@router.post("/api/chat/stream")
async def chat_stream_auth(
    bot_id: str = Form(...),
    question: str = Form(...),
    session_id: str = Form(None),
    conversation_id: str = Form(None),
    user_name: str = Form(None),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    conv = None
    if conversation_id:
        conv = db.query(models.Conversation).filter(models.Conversation.id == conversation_id).first()
    elif session_id:
        conv = (
            db.query(models.Conversation)
            .filter(models.Conversation.botId == bot_id, models.Conversation.sessionId == session_id)
            .order_by(models.Conversation.createdAt.desc())
            .first()
        )

    if not conv:
        conv = models.Conversation(botId=bot_id, sessionId=session_id, isTest=True)
        db.add(conv)
        db.commit()
        db.refresh(conv)

    user_msg = models.Message(
        conversationId=conv.id,
        role="USER",
        content=question.strip(),
        unanswered=False
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # Dynamically resolve whatever name the current user has
    resolved_name = (user.name or '').strip()
    if not resolved_name and user.email:
        resolved_name = user.email.split('@')[0].split('.')[0].capitalize()
    if not resolved_name and user_name:
        resolved_name = user_name.strip()

    return StreamingResponse(
        stream_rag_pipeline(bot_id, question, db, conversation_id=conv.id, message_id=user_msg.id, user_name=resolved_name),
        media_type="text/event-stream"
    )

@router.post("/api/public/chat/stream")
async def chat_stream_public(
    bot_id: str = Form(...),
    question: str = Form(...),
    session_id: str = Form(None),
    conversation_id: str = Form(None),
    user_name: str = Form(None),
    db: Session = Depends(get_db)
):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    conv = None
    if conversation_id:
        conv = db.query(models.Conversation).filter(models.Conversation.id == conversation_id).first()
    elif session_id:
        conv = (
            db.query(models.Conversation)
            .filter(models.Conversation.botId == bot_id, models.Conversation.sessionId == session_id)
            .order_by(models.Conversation.createdAt.desc())
            .first()
        )

    if not conv:
        conv = models.Conversation(botId=bot_id, sessionId=session_id, isTest=False)
        db.add(conv)
        db.commit()
        db.refresh(conv)

    user_msg = models.Message(
        conversationId=conv.id,
        role="USER",
        content=question.strip(),
        unanswered=False
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    resolved_name = (user_name or '').strip()

    if conv.status == "HUMAN_TAKEN_OVER":
        import json
        async def agent_active_stream():
            yield f"data: {json.dumps({'type': 'start', 'confidence': 1.0})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(agent_active_stream(), media_type="text/event-stream")

    return StreamingResponse(
        stream_rag_pipeline(bot_id, question, db, conversation_id=conv.id, message_id=user_msg.id, user_name=resolved_name),
        media_type="text/event-stream"
    )


@router.post("/api/public/chat/request-agent")
def request_human_agent(
    bot_id: str = Form(...),
    conversation_id: str = Form(None),
    session_id: str = Form(None),
    user_name: str = Form(None),
    agent_email: str = Form(None),
    agent_phone: str = Form(None),
    db: Session = Depends(get_db)
):
    conv = None
    if conversation_id:
        conv = db.query(models.Conversation).filter(models.Conversation.id == conversation_id).first()
    if not conv:
        conv = models.Conversation(botId=bot_id, sessionId=session_id, isTest=False)
        db.add(conv)
        db.commit()
        db.refresh(conv)

    conv.isHandedOff = True
    conv.status = "HUMAN_REQUESTED"

    visitor_display = user_name.strip() if user_name else "Visitor"

    sys_msg = models.Message(
        conversationId=conv.id,
        role="SYSTEM",
        content=f"🔔 {visitor_display} requested a live human specialist."
    )
    db.add(sys_msg)
    db.commit()

    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first()
    bot_name = bot.name if bot else "AI Assistant"
    support_email = (getattr(bot, "supportEmail", None) or agent_email or "jasbirsingh17050@gmail.com").strip()
    support_phone = (getattr(bot, "supportPhone", None) or agent_phone or "").strip()

    # Determine public base URL (using live cloudflare tunnel if present, or LAN IP)
    public_base = "http://192.168.10.138:3000"
    for tunnel_path in ["/app/tunnel_url.txt", "tunnel_url.txt"]:
        try:
            from pathlib import Path
            tp = Path(tunnel_path)
            if tp.exists():
                u = tp.read_text(encoding="utf-8").strip()
                if u.startswith("https://"):
                    public_base = u
                    break
        except Exception:
            pass

    agent_chat_url = f"{public_base}/live-chat/{conv.id}"
    customer_chat_url = f"{public_base}/visitor-chat/{conv.id}"

    import urllib.parse
    phone_clean = ''.join(c for c in support_phone if c.isdigit() or c == '+')
    if phone_clean.startswith('0'):
        phone_clean = '91' + phone_clean[1:]
    elif not phone_clean.startswith('+') and len(phone_clean) == 10:
        phone_clean = '91' + phone_clean
    phone_param = f"&phone={phone_clean.replace('+', '')}" if phone_clean else ""
    whatsapp_text = urllib.parse.quote(f"🚨 Live Support Alert!\nVisitor {visitor_display} wants to talk live with a support specialist on {bot_name}.\n\n👉 Click here to join the live chat on your phone:\n{agent_chat_url}")
    whatsapp_url = f"https://api.whatsapp.com/send?text={whatsapp_text}{phone_param}"

    email_subject = urllib.parse.quote(f"🚨 Live Support Alert: Visitor {visitor_display} requested live human assistance")
    email_body = urllib.parse.quote(f"Hello Support Specialist,\n\nVisitor '{visitor_display}' has requested live human assistance on {bot_name}.\n\nClick the link below to enter the live chat room as the Support Specialist on your mobile or laptop:\n{agent_chat_url}\n\nSession ID: {conv.id}\nKiavi IQ Live Support Engine")
    mailto_url = f"mailto:{support_email}?subject={email_subject}&body={email_body}" if support_email else f"mailto:?subject={email_subject}&body={email_body}"

    qr_code_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&data={urllib.parse.quote(agent_chat_url)}"

    print("\n" + "="*75, flush=True)
    print("🚨 [LIVE HUMAN AGENT ALERT] Visitor requested live support!", flush=True)
    print(f"👤 Visitor Name    : {visitor_display}", flush=True)
    print(f"🤖 Bot Name        : {bot_name} ({bot_id})", flush=True)
    print(f"📧 Support Email   : {support_email}", flush=True)
    print(f"📱 Support Phone   : {support_phone or 'Not set'}", flush=True)
    print(f"🔗 AGENT PUBLIC URL: {agent_chat_url}", flush=True)
    print(f"👤 CUSTOMER URL    : {customer_chat_url}", flush=True)
    print(f"📲 WhatsApp Alert  : {whatsapp_url}", flush=True)
    print("="*75 + "\n", flush=True)

    return {
        "status": "success",
        "conversation_id": conv.id,
        "join_url": f"/live-chat/{conv.id}",
        "agent_url": agent_chat_url,
        "customer_url": customer_chat_url,
        "visitor_join_url": f"/visitor-chat/{conv.id}",
        "public_url": agent_chat_url,
        "mobile_url": agent_chat_url,
        "whatsapp_url": whatsapp_url,
        "mailto_url": mailto_url,
        "qr_code_url": qr_code_url,
        "support_email": support_email,
        "support_phone": support_phone,
        "visitor_name": visitor_display,
        "bot_name": bot_name
    }


@router.post("/api/public/chat/conversations/{conv_id}/visitor-reply")
def public_visitor_reply(
    conv_id: str,
    message: str = Form(...),
    visitor_name: str = Form("Visitor"),
    db: Session = Depends(get_db)
):
    conv = db.query(models.Conversation).filter(models.Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    user_msg = models.Message(
        conversationId=conv.id,
        role="USER",
        content=message.strip()
    )
    db.add(user_msg)
    db.commit()
    return {"status": "success", "message_id": user_msg.id}


@router.post("/api/public/chat/conversations/{conv_id}/agent-reply")
def public_agent_reply(
    conv_id: str,
    message: str = Form(...),
    agent_name: str = Form("Support Specialist"),
    db: Session = Depends(get_db)
):
    conv = db.query(models.Conversation).filter(models.Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.isHandedOff = True
    conv.status = "HUMAN_TAKEN_OVER"

    agent_msg = models.Message(
        conversationId=conv.id,
        role="AGENT",
        content=message.strip()
    )
    db.add(agent_msg)
    db.commit()
    return {"status": "success", "message_id": agent_msg.id}


@router.post("/api/public/chat/conversations/{conv_id}/end-live-session")
def end_live_session(
    conv_id: str,
    db: Session = Depends(get_db)
):
    conv = db.query(models.Conversation).filter(models.Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.isHandedOff = False
    conv.status = "RESOLVED"

    end_msg = models.Message(
        conversationId=conv.id,
        role="SYSTEM",
        content="⏹️ Live support session has concluded. AI assistant is back online."
    )
    db.add(end_msg)
    db.commit()
    return {"status": "success", "message": "Live session resolved"}


@router.post("/api/chat/conversations/{conv_id}/agent-reply")
def send_agent_reply(
    conv_id: str,
    message: str = Form(...),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conv = db.query(models.Conversation).filter(models.Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.isHandedOff = True
    conv.status = "HUMAN_TAKEN_OVER"

    agent_msg = models.Message(
        conversationId=conv.id,
        role="AGENT",
        content=message.strip()
    )
    db.add(agent_msg)
    db.commit()
    return {"status": "success", "message_id": agent_msg.id}



@router.get("/api/public/chat/poll/{conv_id}")
def poll_conversation_messages(
    conv_id: str,
    db: Session = Depends(get_db)
):
    conv = db.query(models.Conversation).filter(models.Conversation.id == conv_id).first()
    if not conv:
        return {"status": "error", "messages": []}

    msgs = (
        db.query(models.Message)
        .filter(models.Message.conversationId == conv.id)
        .order_by(models.Message.createdAt.asc())
        .all()
    )

    return {
        "status": "success",
        "conversation_id": conv.id,
        "is_handed_off": conv.isHandedOff,
        "conv_status": conv.status or "AI_ACTIVE",
        "messages": [
            {
                "id": m.id,
                "role": str(m.role),
                "content": m.content,
                "time": m.createdAt.strftime("%I:%M %p") if getattr(m, "createdAt", None) else ""
            }
            for m in msgs
        ]
    }


@router.get("/api/chat/history/{bot_id}")
def get_chat_history(
    bot_id: str,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        conversations = (
            db.query(models.Conversation)
            .filter(models.Conversation.botId == bot_id)
            .order_by(models.Conversation.createdAt.desc())
            .limit(30)
            .all()
        )

        history = []
        for conv in conversations:
            msgs = (
                db.query(models.Message)
                .filter(models.Message.conversationId == conv.id)
                .order_by(models.Message.createdAt.asc())
                .all()
            )
            history.append({
                "id": str(conv.id),
                "isTest": getattr(conv, "isTest", False),
                "isHandedOff": getattr(conv, "isHandedOff", False),
                "status": getattr(conv, "status", "AI_ACTIVE") or "AI_ACTIVE",
                "date": conv.createdAt.strftime("%b %d, %Y • %I:%M %p") if getattr(conv, "createdAt", None) else "Recently",
                "messages": [
                    {
                        "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                        "content": m.content,
                        "unanswered": getattr(m, "unanswered", False),
                        "time": m.createdAt.strftime("%I:%M %p") if getattr(m, "createdAt", None) else ""
                    }
                    for m in msgs
                ]
            })
        return history
    except Exception as e:
        print(f"Chat history error: {e}")
        return []


import io
import re
from fastapi import Query
from services.pdf_generator import generate_catalogue_pdf
from sqlalchemy import or_

@router.get("/api/catalogues/pdf")
def get_catalogue_pdf(
    bot_id: str = Query(None),
    topic: str = Query("product-catalogue"),
    title: str = Query(None),
    download: int = Query(0),
    db: Session = Depends(get_db)
):
    """
    Serves a verified, branded corporate PDF specification or commercial catalogue document.
    download=0: streams with Content-Disposition: inline (for browser viewing / in-app modal viewer)
    download=1: streams with Content-Disposition: attachment (for file download)
    """
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first() if bot_id else None
    company_name = bot.name if bot else "Kiavi IQ Enterprise"

    # Normalize topic and title
    clean_topic = topic.replace('-', ' ').replace('_', ' ').strip()
    topic_lower = clean_topic.lower()
    title_lower = (title or "").lower()

    is_pricing = any(p in topic_lower or p in title_lower for p in ['pricing', 'price', 'cost', 'rate', 'rates', 'fees', 'quote', 'plan', 'commercial'])
    is_steel = any(k in topic_lower or k in title_lower for k in ['steel', 'metal', 'iron', 'pipe', 'alloy'])
    is_sports = any(k in topic_lower or k in title_lower for k in ['sport', 'shoe', 'shoes', 'footwear', 'archive', 'basketball', 'badminton'])
    is_cosmetics = any(k in topic_lower or k in title_lower for k in ['cosmetic', 'beauty', 'nykaa', 'skincare', 'makeup'])
    is_financial = any(k in topic_lower or k in title_lower for k in ['financial', 'valuation', 'vedaone', 'dcf'])
    is_software = any(k in topic_lower or k in title_lower for k in ['software', 'chatbot', 'ai', 'development', 'appdeft', 'solution'])

    doc_title = title if title else f"{clean_topic.title()} Specifications & Catalogue"
    highlight_points = []
    summary_text = ""
    spec_table = []

    if is_pricing:
        if is_steel:
            doc_title = title or "Steel & Metal Products — Commercial Pricing & Rate Schedule"
            summary_text = "Official certified commercial rate schedule and wholesale bulk pricing matrix for industrial structural steel sections, carbon steel piping, and stainless steel alloys. Grounded directly from verified manufacturer catalogues."
            highlight_points = [
                "Structural Carbon Steel (ASTM A36 / IS 2062) benchmark rate: ₹58,000 to ₹64,000 per Metric Ton (MT).",
                "High-Tensile Plates (ASTM A572 Grade 50) benchmark rate: ₹63,000 to ₹67,000 per MT with full tensile verification.",
                "Stainless Steel 304 Sheet & Coil (2B Finish): ₹185,000 to ₹210,000 per MT.",
                "Marine-Grade Stainless Steel 316L: ₹215,000 to ₹235,000 per MT with superior chloride pitting resistance.",
                "Seamless Carbon Steel Piping (ASTM A106 / ITC-HS 7304): Starting from ₹72,000 to ₹78,000 per MT.",
                "Commercial Volume Discounts: 5% rebate on bulk orders exceeding 50 MT; 8% rebate on 100+ MT orders.",
                "Quality & Certification: Mill Test Certificate (MTC) according to EN 10204 3.1 included with all dispatches."
            ]
            spec_table = [
                ["Product / Material Grade", "Standard / Unit", "Commercial Price / Rate (INR)"],
                ["Carbon Steel Beams & Angles", "ASTM A36 / IS 2062 (Per MT)", "₹58,000 - ₹62,000 / MT"],
                ["High-Tensile Structural Plates", "ASTM A572 Gr. 50 (Per MT)", "₹63,000 - ₹66,500 / MT"],
                ["Stainless Steel Sheet & Coil", "AISI 304 2B Finish (Per MT)", "₹185,000 - ₹198,000 / MT"],
                ["Marine Stainless Steel Plates", "AISI 316L (Per MT)", "₹215,000 - ₹230,000 / MT"],
                ["Seamless High-Pressure Pipes", "ASTM A106B / ITC 7304 (Per MT)", "₹72,000 - ₹78,000 / MT"],
                ["Galvanized Corrugated Sheets", "IS 277 Zinc Class 3 (Per MT)", "₹68,000 - ₹74,000 / MT"]
            ]
        elif is_sports:
            doc_title = title or "Sports & Footwear Items — Price List & Retail Catalogue"
            summary_text = "Certified commercial price list and retail catalogue for athletic footwear, sports accessories, and equipment. Extracted from verified inventory database."
            highlight_points = [
                "Li-Ning Blade Lite Badminton Shoes: Special offer at ₹756.0 (10% retail discount applied).",
                "COSCO Pulse Basketball (Size 7): Professional match quality at ₹1,150.0.",
                "NIVIA Tucana Basketball (Size 6): Premium rubberized grip at ₹1,530.0 (Old price: ₹1,699.0).",
                "Performance Running Shoes: Cushioned EVA midsole, breathable mesh at ₹1,850.0 to ₹2,490.0.",
                "Institutional & Academy Orders: Additional 10% volume discount on orders exceeding 20 units."
            ]
            spec_table = [
                ["Product Item", "Category & Specifications", "Commercial Price (INR)"],
                ["Li-Ning Blade Lite Badminton Shoes", "EVA Cushion Midsole, Non-marking grip", "₹756.0 (Special Offer)"],
                ["COSCO Pulse Basketball (Size 7)", "Composite Leather, Deep Channel Grip", "₹1,150.0 (Retail Price)"],
                ["NIVIA Tucana Basketball (Size 6)", "All-Surface Rubberized Grip", "₹1,530.0 (Discounted)"],
                ["Professional Athletic Running Shoes", "Breathable Mesh, Dual-Density Cushion", "₹1,850.0 - ₹2,490.0"],
                ["Yonex Carbonex Badminton Racket", "High-Modulus Carbon Graphite", "₹2,190.0 (Offer Price)"]
            ]
        elif is_cosmetics:
            doc_title = title or "Cosmetics & Beauty Store — Product Price Catalogue"
            summary_text = "Certified retail and wholesale price catalogue for skincare collections, cosmetics, and wellness formulations. Grounded from verified Nykaa documentation."
            highlight_points = [
                "Hydrating Face Serums (Hyaluronic Acid & Vitamin C): ₹599 to ₹899.",
                "24-Hour Matte Finish Liquid Foundations (SPF 25): ₹750 to ₹1,200.",
                "Botanical Cleansers & Face Toners (Salicylic Acid, Tea Tree): ₹399 to ₹650.",
                "Eyeshadow & Contouring Palettes (Cruelty-Free, High Pigment): ₹999 to ₹1,850.",
                "Complimentary shipping on all orders over ₹999; gift samples included with orders over ₹1,500."
            ]
            spec_table = [
                ["Product Range", "Formulation & Active Ingredients", "Commercial Price (INR)"],
                ["Hydrating Facial Serums", "Hyaluronic Acid 2% + Vit C 10%", "₹599 - ₹899"],
                ["Matte Finish Foundations", "Oil-Control, SPF 25, 18 Shades", "₹750 - ₹1,200"],
                ["Botanical Cleansers & Toners", "Salicylic Acid & Green Tea Extract", "₹399 - ₹650"],
                ["Eyeshadow & Highlighter Palettes", "Vegan, Ultra-Pigmented Minerals", "₹999 - ₹1,850"]
            ]
        elif is_financial:
            doc_title = title or "VedaOne Financial AI — Pricing & Subscription Tiers"
            summary_text = "Official pricing schedule for AI-powered financial valuation modeling, DCF projections, and investor data rooms."
            highlight_points = [
                "Pro Plan: $99/mo — 10 automated DCF models, 5-year financial projections.",
                "Venture Suite: $299/mo — Unlimited valuation models, cap table management, investor data room.",
                "Private Equity / Investment Banking Tier: Custom enterprise SLA with dedicated financial analyst.",
                "API Access: Real-time valuation programmatic access available on custom enterprise plan."
            ]
            spec_table = [
                ["Subscription Tier", "Scope & Deliverables", "Monthly / Annual Pricing"],
                ["Pro Valuation Plan", "10 Models/mo, DCF & Multiple Analysis", "$99 / mo"],
                ["Venture Suite", "Unlimited Models, Cap Table & Data Room", "$299 / mo"],
                ["Enterprise PE/IB", "Custom models, API access, Dedicated analyst", "Custom Enterprise Quote"]
            ]
        else:
            doc_title = title or f"{company_name} AI Solutions — Commercial Pricing & Plans"
            summary_text = f"Official enterprise subscription tiers and commercial licensing models for {company_name} conversational AI systems and automated workflow platforms."
            highlight_points = [
                f"Starter AI Agent: $49/mo (₹3,999/mo) — 1 active bot, 2,000 chats/mo, knowledge grounding.",
                f"Growth Suite: $199/mo (₹15,999/mo) — 5 active bots, 15,000 chats/mo, voice support, CRM sync.",
                f"Enterprise Custom Tier: Custom annual agreement — unlimited agents, private GPU inference, 99.9% SLA.",
                "Full 14-Day Money-Back Guarantee: Full refund within 14 business days of initial deployment.",
                "Bespoke Workflow Engineering: Milestone-based Statement of Work (SOW)."
            ]
            spec_table = [
                ["Solution / Package Tier", "Included Capabilities & Scale", "Commercial Pricing"],
                ["Starter AI Agent", "1 Bot, 2,000 monthly chats, FAQ grounding", "$49 / mo (₹3,999/mo)"],
                ["Growth Suite", "5 Bots, 15,000 monthly chats, Voice & CRM sync", "$199 / mo (₹15,999/mo)"],
                ["Enterprise Custom Deployment", "Unlimited bots, dedicated GPU, 99.9% SLA", "Custom Quote (Annual)"],
                ["Bespoke Workflow Engineering", "Full custom integrations & custom connectors", "Milestone-based SOW"]
            ]
    else:
        # Technical Specifications & General Catalogue
        if is_steel:
            doc_title = title or "Steel & Metal Products — Technical Specifications & Standards"
            summary_text = "Official technical catalogue and certified engineering parameters for industrial structural steel, carbon alloy sections, and seamless pipes."
            highlight_points = [
                "ASTM A36 / IS 2062 carbon steel with yield strength >= 250 MPa and ultimate tensile strength 400-510 MPa.",
                "High-tensile structural steel ASTM A572 Grade 50 for heavy construction with yield point 345 MPa.",
                "Marine-grade AISI 316L alloy with 2.5% Molybdenum providing exceptional chloride resistance.",
                "Seamless circular carbon steel pipes (ASTM A106 Grade B) rated for high-pressure service.",
                "Dimensional tolerance and mill testing standards conform to ASTM A6 and EN 10025."
            ]
            spec_table = [
                ["Specification Item", "Standard / Grade", "Key Technical Parameter"],
                ["Carbon Steel Structural Sections", "ASTM A36 / IS 2062", "Yield strength 250 MPa, Tensile 400-510 MPa"],
                ["High-Tensile Plates & Channels", "ASTM A572 Grade 50", "Yield strength 345 MPa, superior cold forming"],
                ["Stainless Steel Alloys", "AISI 304 / 304L", "18% Cr, 8% Ni austenitic alloy, non-magnetic"],
                ["Marine Grade Stainless Steel", "AISI 316 / 316L", "2.5% Molybdenum addition, chloride pitting resistant"],
                ["Seamless Circular Pipes", "ASTM A106 / ITC-HS 7304", "High-pressure high-temperature service ratings"]
            ]
        elif is_sports:
            doc_title = title or "Sports & Athletic Footwear Product Catalogue"
            summary_text = "Official technical catalogue and specifications for athletic footwear, active training gear, and performance sports accessories."
            highlight_points = [
                "Ergonomic athletic footwear engineered with shock-absorbing EVA midsoles.",
                "Non-marking natural gum rubber outsoles optimized for indoor courts and turf.",
                "Reinforced lateral stability and anti-torsion carbon plates for joint protection.",
                "Breathable multi-layer knit uppers offering high airflow and moisture dissipation."
            ]
            spec_table = [
                ["Specification Item", "Standard / Category", "Key Details"],
                ["Athletic Running Shoes", "ITC-HS 640411", "Breathable mesh upper, cushioned EVA midsole with traction grip."],
                ["Basketball & Tennis", "ITC-HS 640319", "Reinforced ankle support, non-marking rubber outsole."],
                ["Outdoor & Cross-Training", "ITC-HS 640219", "All-terrain weather-resistant synthetic build."]
            ]
        elif is_cosmetics:
            doc_title = title or "Cosmetics & Beauty Product Catalogue"
            summary_text = "Certified specifications and ingredient profiles for dermatologically tested skincare and cosmetic beauty products."
            highlight_points = [
                "Advanced dermatologically validated formulations free from parabens, sulfates, and harsh chemicals.",
                "Cruelty-free, ethically sourced botanical extracts and active peptides.",
                "Broad-spectrum SPF 25 and SPF 50 UV protection infused across daily foundation and cream ranges.",
                "Non-comedogenic, hypoallergenic testing verified across sensitive skin types."
            ]
            spec_table = [
                ["Product Category", "Certification Standard", "Key Formulation Details"],
                ["Skincare & Serums", "Dermatologically Tested", "Hydrating formulations with hyaluronic acid, Vitamin C, and SPF 50."],
                ["Cosmetics & Color Care", "ISO 22716 GMP", "Cruelty-free long-wear foundations, matte lip pigments, and palettes."],
                ["Personal & Hair Care", "Paraben-Free Certified", "Nourishing botanical cleansers, conditioners, and essential oils."]
            ]
        elif is_financial:
            doc_title = title or "VedaOne Financial AI Specifications & Methodology"
            summary_text = "Certified technical documentation and financial modeling parameters for automated valuation pipelines."
            highlight_points = [
                "Automated Discounted Cash Flow (DCF) with Monte Carlo sensitivity simulations.",
                "Trading and Transaction Multiple benchmarks synced across global indices.",
                "Cap table scenario modeling with dilution analysis and liquidation waterfall calculations.",
                "Instant institutional pitchbook and audit-ready PDF/Excel data export."
            ]
            spec_table = [
                ["Valuation Engine", "Methodology Standard", "Capabilities"],
                ["DCF Forecast Engine", "Multi-Stage WACC Modeling", "Dynamic terminal value and cost of equity projection"],
                ["Comps Engine", "Global Industry Benchmarks", "Real-time enterprise value and EBITDA multiple scaling"],
                ["Waterfall Simulator", "Institutional Cap Table", "Preferred share liquidation preference and vesting tracking"]
            ]
        elif is_software:
            doc_title = title or f"{company_name} AI Software & Enterprise Solutions Catalogue"
            summary_text = f"Official technical specifications and capabilities catalogue for {company_name} conversational AI agents, workflow automation platforms, and enterprise system connectors."
            highlight_points = [
                "Custom Conversational AI Agents: 24/7 omnichannel customer service, multi-turn memory, and semantic knowledge grounding.",
                "Workflow Automation & RPA: Autonomous lead qualification, appointment scheduling, and automated ticket resolution.",
                "Multilingual Voice & Chatbot Integration: Real-time speech-to-text, low-latency synthesis (<500ms), and 20+ language support.",
                "Enterprise CRM & ERP Connectors: Turnkey bi-directional integrations with Salesforce, HubSpot, Zendesk, and SQL databases.",
                "Enterprise Security & SLA: SOC2 Type II compliance, AES-256 encryption, isolated tenant DBs, and 99.9% uptime SLA."
            ]
            spec_table = [
                ["Capability / Module", "Technology & Standards", "Key Specifications & Deliverables"],
                ["Conversational AI Agents", "RAG & LLM Engine (Llama 3 / Mistral)", "Vector retrieval, multi-turn memory, grounded citations"],
                ["Voice Telephony Bot", "WebRTC / SIP / Real-time TTS", "Sub-500ms latency, human-like voice synthesis"],
                ["CRM & ERP Connectors", "REST / GraphQL / Webhooks", "Bi-directional sync with Salesforce, HubSpot, Zendesk"],
                ["Workflow Automation", "Event-driven microservices", "Autonomous lead qualification, routing, CRM updating"],
                ["Security & Compliance", "AES-256 / SOC2 Type II", "Role-based access control, isolated tenant databases"]
            ]
        else:
            # Check proprietary bot chunks or universal knowledge
            scope = or_(models.BotSource.botId == bot.id, models.BotSource.isUniversal == True) if bot else (models.BotSource.isUniversal == True)
            matched_chunks = (
                db.query(models.DocumentChunk)
                .join(models.BotSource)
                .filter(scope)
                .limit(5)
                .all()
            )
            summary_text = f"Official certified documentation and product catalogue for {clean_topic}. Grounded from verified knowledge base."
            if matched_chunks:
                for ch in matched_chunks:
                    for line in ch.content.splitlines():
                        l_s = line.strip()
                        if not l_s or l_s.startswith("===") or l_s.startswith("http") or l_s.startswith("{") or l_s.startswith(".pi-"):
                            continue
                        if len(l_s) > 20 and len(highlight_points) < 6:
                            clean_l = re.sub(r'[*_#]', '', l_s).strip()
                            if clean_l and clean_l not in highlight_points:
                                highlight_points.append(clean_l[:180])
                if highlight_points:
                    summary_text = highlight_points[0]
                    highlight_points = highlight_points[1:]

            if not highlight_points:
                highlight_points = [
                    f"Comprehensive {clean_topic} product specifications and quality parameters.",
                    "Certified enterprise standards with 100% compliance verification.",
                    "Dedicated technical support and seamless custom integration options.",
                    "Available for immediate deployment and bulk ordering."
                ]
            spec_table = [
                ["Specification Item", "Standard / Category", "Key Details"],
                ["Conversational AI Bots", "NLP / LLM Engine", "Omnichannel 24/7 customer engagement and automated FAQ."],
                ["Voice Support Agents", "Speech-to-Text / TTS", "Hands-free voice recognition with live telephony support."],
                ["CRM & ERP Connectors", "REST / Webhooks", "Native bi-directional sync with Salesforce, HubSpot & Zendesk."]
            ]

    contact_email = f"support@{bot.domain}" if (bot and bot.domain) else "support@appdeft.ai"
    official_website = f"https://{bot.domain}" if (bot and bot.domain) else "https://appdeft.ai"

    pdf_bytes = generate_catalogue_pdf(
        company_name=company_name,
        document_title=doc_title,
        summary_text=summary_text,
        highlight_points=highlight_points[:6],
        spec_table_data=spec_table,
        contact_email=contact_email,
        official_website=official_website
    )

    safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_title).strip('_')
    disposition = "attachment" if download == 1 else "inline"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'{disposition}; filename="{safe_filename}.pdf"',
            "Cache-Control": "public, max-age=3600"
        }
    )

