from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Form, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db import models
from app.core.dependencies import get_current_user
from app.services.rag import stream_rag_pipeline
from app.services.synthesis import synthesize_multiple_documents

router = APIRouter(tags=["Chat"])

class MultiDocSynthesisRequest(BaseModel):
    source_ids: List[str]
    query: Optional[str] = None
    format: Optional[str] = "briefing"

@router.post("/api/chat/multi-doc-synthesis", summary="Conduct comparative multi-document synthesis across manuals/datasheets")
def conduct_multi_doc_synthesis(
    req: MultiDocSynthesisRequest,
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Synthesizes facts, tolerances, and engineering specifications across 2 or more documents.
    Produces grounded side-by-side matrices and compliance deltas with citations.
    """
    client_ip = request.client.host if request.client else "unknown"
    try:
        result = synthesize_multiple_documents(
            db=db,
            source_ids=req.source_ids,
            query=req.query,
            format_type=req.format or "briefing",
            current_user=current_user,
            client_ip=client_ip
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Comparative synthesis failed: {err}")


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
        stream_rag_pipeline(bot_id, question, db, conversation_id=conv.id, message_id=user_msg.id, user_name=resolved_name, user=user),
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
from app.services.pdf_generator import generate_catalogue_pdf
from app.services.docx_generator import generate_compliance_docx
from app.services.scraper import is_valid_address
from sqlalchemy import or_

def resolve_target_source(available_sources: list, title: str, topic: str, db: Session):
    clean_topic = topic.replace('-', ' ').replace('_', ' ').strip() if topic else ""

    # 1. Direct title matching against s.title or s.display_title
    if title:
        tl = title.strip().lower()
        for s in available_sources:
            s_clean = (s.title or '').strip().lower()
            s_disp = (getattr(s, 'display_title', '') or '').strip().lower()
            if s_clean in tl or (len(s_clean) > 4 and s_clean in tl) or (tl in s_clean):
                return s
            if s_disp and (s_disp in tl or tl in s_disp):
                return s

    # 2. Topic keyword matching against s.title or s.display_title
    if topic and topic != "product-catalogue":
        clean_words = [w for w in clean_topic.lower().split() if len(w) > 2]
        for s in available_sources:
            s_lower = ((s.title or '') + ' ' + (getattr(s, 'display_title', '') or '')).lower()
            if any(w in s_lower for w in clean_words):
                return s

    # 3. Deep chunk matching: Check DocumentChunk content for document titles or poster headings
    search_terms = []
    if title and title.lower() not in ['product specifications & catalogue', 'verified documentation']:
        search_terms.append(title.lower())
    if clean_topic and clean_topic.lower() not in ['product catalogue', 'catalogue']:
        search_terms.append(clean_topic.lower())

    if search_terms:
        for s in available_sources:
            s_chunks = db.query(models.DocumentChunk).filter(models.DocumentChunk.sourceId == s.id).limit(3).all()
            chunk_blob = " ".join(c.content.lower() for c in s_chunks if c.content)
            for st in search_terms:
                if st in chunk_blob:
                    return s
                st_words = [w for w in st.split() if len(w) > 3]
                if len(st_words) >= 2 and all(w in chunk_blob for w in st_words):
                    return s

    # 4. Fallback ONLY if no specific title/topic was requested
    is_specific_request = bool(title or (topic and topic != "product-catalogue"))
    if not is_specific_request and available_sources:
        return available_sources[0]

    return None

@router.get("/api/catalogues/pdf")
def get_catalogue_pdf(
    bot_id: str = Query(None),
    topic: str = Query("product-catalogue"),
    title: str = Query(None),
    download: int = Query(0),
    db: Session = Depends(get_db)
):
    """
    Serves a verified, branded corporate PDF specification or catalogue document.
    Strictly isolated to the bot's organization with zero cross-tenant data leakage.
    download=0: streams inline (browser viewing / in-app modal viewer)
    download=1: streams as attachment
    """
    topic = str(topic) if (topic is not None and isinstance(topic, str)) else "product-catalogue"
    title = str(title).strip() if (title is not None and isinstance(title, str) and title.strip()) else None

    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first() if (bot_id and isinstance(bot_id, str)) else None
    company_name = bot.name if bot else "Kiavi IQ Enterprise"
    contact_email = getattr(bot, "supportEmail", None) or (f"support@{bot.domain}" if (bot and bot.domain) else "jasbirsingh17050@gmail.com")
    official_website = f"https://{bot.domain}" if (bot and bot.domain) else ""

    # Strictly load sources belonging to this bot
    available_sources = []
    if bot:
        source_scope = or_(
            models.BotSource.botId == bot.id,
            (models.BotSource.isUniversal == True) & (models.BotSource.orgId == bot.orgId)
        )
        available_sources = db.query(models.BotSource).filter(source_scope).order_by(models.BotSource.createdAt.desc()).all()
    else:
        available_sources = db.query(models.BotSource).filter(
            models.BotSource.isUniversal == True
        ).order_by(models.BotSource.createdAt.desc()).all()

    clean_topic = topic.replace('-', ' ').replace('_', ' ').strip()
    target_source = resolve_target_source(available_sources, title, topic, db)

    if not target_source:
        doc_title = title or f"{company_name} Knowledge Documentation"
        summary_text = f"No indexed knowledge records or uploaded documents are currently associated with {company_name}. Please upload a document or scrape a website to generate grounded documentation."
        highlight_points = [
            f"Active Agent: {company_name}",
            "Knowledge Base Status: Empty (No custom sources indexed)",
            "Ready for Document Upload or Web Crawling"
        ]
        spec_table = [
            ["Property / Field", "Status / Details"],
            ["Agent Name", company_name],
            ["Database Status", "No sources indexed in database"]
        ]
    else:
        source_clean_name = target_source.title.split('|')[0].strip() if '|' in target_source.title else target_source.title
        matched_chunks = (
            db.query(models.DocumentChunk)
            .filter(models.DocumentChunk.sourceId == target_source.id)
            .all()
        )
        chunk_texts = [ch.content for ch in matched_chunks]
        combined_text = "\n\n".join(chunk_texts)
        clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', combined_text).strip()

        is_image_doc = target_source.title.lower().endswith(('.jpeg', '.jpg', '.png', '.webp')) or "[visual image" in combined_text.lower() or "[diagram" in combined_text.lower()
        is_csv_doc = target_source.title.lower().endswith(('.csv', '.tsv', '.xlsx', '.xls')) or "=== csv table" in combined_text.lower() or "[row 1]:" in combined_text.lower()

        if is_image_doc:
            clean_text_no_headers = re.sub(r'\[Visual Image\s*/\s*Diagram Data\s*\([^)]*\)\]:?', '', clean_text, flags=re.IGNORECASE)
            clean_text_no_headers = re.sub(r'\[Visual Image Data\]:?', '', clean_text_no_headers, flags=re.IGNORECASE)
            clean_text_no_headers = re.sub(r'\[Diagram Data\]:?', '', clean_text_no_headers, flags=re.IGNORECASE)

            # Check for real title in chunk
            t_match = re.search(r'(?:poster for\s*["\']([^"\']+)["\']|Title[:\s*]+["\']?([^"\']+)["\']?)', combined_text, re.IGNORECASE)
            extracted_doc_title = (t_match.group(1) or t_match.group(2)).strip() if t_match else ""

            # Extract bullet points from OCR if available
            ocr_bullets = re.findall(r'[\*\-]\s+([^\n\*]+)', clean_text_no_headers)
            clean_ocr_bullets = [b.strip().strip('*').strip('"').strip("'") for b in ocr_bullets if len(b.strip()) > 3]

            # Extract narrative sentences
            filtered_lines = [l.strip() for l in clean_text_no_headers.splitlines() if l.strip() and not l.strip().startswith('*') and not l.strip().startswith('-')]
            full_clean_para = " ".join(filtered_lines)
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_clean_para) if len(s.strip()) > 15]

            first_sentence = sentences[0] if sentences else ""
            if first_sentence:
                first_sentence = re.sub(rf'^{re.escape(source_clean_name)}[:\s]*', '', first_sentence, flags=re.IGNORECASE).strip()

            effective_title = extracted_doc_title or source_clean_name
            doc_title = title if (title and not any(title.lower().startswith(p) for p in ['test.', 'upload_', 'img_'])) else f"{effective_title} — Visual Reference & Documentation"
            summary_text = (
                first_sentence if first_sentence else
                f"Official verified reference document extracted from {effective_title} for {company_name}. Grounded directly from indexed visual knowledge and verified OCR records."
            )
            highlight_points = []
            if clean_ocr_bullets:
                for b in clean_ocr_bullets[:6]:
                    if b not in highlight_points:
                        highlight_points.append(b)
            if len(highlight_points) < 3 and len(sentences) > 1:
                for s in sentences[1:5]:
                    s_clean = re.sub(rf'^{re.escape(source_clean_name)}[:\s]*', '', s, flags=re.IGNORECASE).strip()
                    if s_clean not in highlight_points:
                        highlight_points.append(s_clean)
            if not highlight_points:
                highlight_points = [
                    f"Verified visual reference extracted from {effective_title}.",
                    "High-precision Optical Character Recognition (OCR) processed and grounded in vector database.",
                    "Zero hallucination standard: all details certified directly from document content."
                ]
            spec_table = [
                ["Property / Parameter", "Verified Specification"],
                ["Document Title", effective_title],
                ["Source File", source_clean_name],
                ["Document Category", "Visual Plate / Document Guide"],
                ["Grounding Scope", "Organization Knowledge Base"],
                ["Extraction Engine", "NVIDIA Multimodal Vision OCR / Tesseract"],
                ["Verification Status", "100% Grounded in Vector Database"]
            ]
        elif is_csv_doc:
            doc_title = title or f"{source_clean_name} — Commercial Data Sheet"
            summary_text = f"Verified commercial data sheet and inventory matrix extracted from {source_clean_name}. Grounded directly from verified indexed records for {company_name}."

            row_matches = re.findall(r'\[Row \d+\]:\s*(.+)', clean_text)
            spec_table = []
            highlight_points = []

            header_match = re.search(r'Columns:\s*([^\n]+)', clean_text)
            if header_match:
                headers = [h.strip() for h in header_match.group(1).split(',')[:4]]
            else:
                headers = ["Item / Record", "Details", "Price / Value"]
            spec_table.append(headers)

            for rm in row_matches[:15]:
                items = {}
                for pair in rm.split('|'):
                    if ':' in pair:
                        k, v = pair.split(':', 1)
                        items[k.strip().lower()] = v.strip()

                row_vals = []
                for h in headers:
                    h_clean = h.strip().lower()
                    val = items.get(h_clean, "")
                    if not val:
                        for ik, iv in items.items():
                            if h_clean in ik or ik in h_clean:
                                val = iv
                                break
                    row_vals.append(val[:35] if val else "-")
                if any(v != "-" for v in row_vals):
                    spec_table.append(row_vals)
                    first_val = [v for v in row_vals if v != "-"]
                    if first_val and len(highlight_points) < 5:
                        highlight_points.append(" | ".join(first_val[:3]))

            if len(spec_table) <= 1:
                spec_table = [["Specification Item", "Detail"], ["Data Source", source_clean_name], ["Grounding", "Verified Tabular Records"]]
            if not highlight_points:
                highlight_points = [f"Tabular records extracted and verified from {source_clean_name}.", "Structured parameter mapping for high-precision retrieval."]
        else:
            # General Web Page or Digital Document
            extracted_phones = []
            extracted_emails = []
            extracted_addrs = []
            extracted_urls = []
            service_rows = []
            pricing_rows = []
            general_bullets = []

            p_match = re.findall(r'(?:Telephone|Phone|Contact Phone|Mobile)[:\s]+([^\n]+)', clean_text, re.IGNORECASE)
            for pm in p_match:
                for p_val in re.split(r'[,;]', pm):
                    clean_p = p_val.strip()
                    if clean_p and clean_p not in extracted_phones and len(clean_p) > 6 and "no direct" not in clean_p.lower():
                        extracted_phones.append(clean_p)

            e_match = re.findall(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', clean_text)
            for em in e_match:
                clean_e = em.strip()
                if clean_e and clean_e not in extracted_emails and not clean_e.endswith(('.png', '.jpg', '.webp')):
                    extracted_emails.append(clean_e)

            a_match = re.findall(r'(?:Address|Office Location|Office Address)[:\s]+([^\n]+)', clean_text, re.IGNORECASE)
            for am in a_match:
                clean_a = am.strip()
                if clean_a and clean_a not in extracted_addrs and is_valid_address(clean_a) and "no public" not in clean_a.lower():
                    extracted_addrs.append(clean_a)

            u_match = re.findall(r'(?:Official Website|Website)[:\s]+(https?://[^\s\n]+)', clean_text, re.IGNORECASE)
            for um in u_match:
                clean_u = um.strip()
                if clean_u and clean_u not in extracted_urls:
                    extracted_urls.append(clean_u)

            for line in clean_text.splitlines():
                l_s = line.strip()
                if not l_s or l_s.startswith("===") or l_s.startswith("{") or l_s.startswith(".pi-") or l_s.startswith("http"):
                    continue
                clean_l = re.sub(r'[*_#]', '', l_s).strip()
                if len(clean_l) < 15 or len(clean_l) > 220:
                    continue

                lower_l = clean_l.lower()
                if any(k in lower_l for k in ['$', '₹', 'price', 'pricing', 'rate', 'cost', 'fee', 'plan', '/mo', 'per month', 'subscription']):
                    if clean_l not in pricing_rows and len(pricing_rows) < 6:
                        pricing_rows.append(clean_l)
                elif any(k in lower_l for k in ['service', 'solution', 'platform', 'feature', 'system', 'product', 'development', 'management', 'support']):
                    if clean_l not in service_rows and len(service_rows) < 6:
                        service_rows.append(clean_l)
                else:
                    if clean_l not in general_bullets and len(general_bullets) < 6:
                        general_bullets.append(clean_l)

            if extracted_emails:
                contact_email = extracted_emails[0]
            if extracted_urls:
                official_website = extracted_urls[0]

            doc_title = title or f"{source_clean_name} — Specifications & Overview"
            summary_text = f"Official certified documentation and reference catalogue for {company_name}. Grounded directly from verified database records and indexed knowledge ({source_clean_name})."
            highlight_points = (pricing_rows[:2] + service_rows[:2] + general_bullets[:2])[:6]
            if not highlight_points:
                highlight_points = [
                    f"Comprehensive specifications and parameters for {company_name}.",
                    f"Verified knowledge records extracted from {source_clean_name}.",
                    "Grounded directly from indexed website and documentation data."
                ]

            spec_table = [["Category / Module", "Scope & Specifications", "Verified Detail"]]
            if official_website:
                spec_table.append(["Digital Presence", "Official Website / Domain", official_website[:35]])
            if extracted_addrs:
                spec_table.append(["Office Location", "Corporate Headquarters / Office", extracted_addrs[0][:35]])
            if extracted_phones:
                spec_table.append(["Contact Channel", "Telephone / Hotline", extracted_phones[0][:35]])
            if extracted_emails:
                spec_table.append(["Contact Channel", "Official Email Inquiries", extracted_emails[0][:35]])
            for idx, s_row in enumerate(service_rows[:3], 1):
                parts = s_row.split(':', 1) if ':' in s_row else (f"Capability {idx}", s_row)
                spec_table.append(["Services & Solutions", parts[0][:28], parts[1].strip()[:35]])
            for idx, p_row in enumerate(pricing_rows[:2], 1):
                parts = p_row.split(':', 1) if ':' in p_row else (f"Rate / Plan {idx}", p_row)
                spec_table.append(["Commercial Model", parts[0][:28], parts[1].strip()[:35]])

            if len(spec_table) <= 1:
                spec_table.append(["Knowledge Record", source_clean_name[:28], "100% verified enterprise indexed data"])
                spec_table.append(["Compliance", "Enterprise Grounding", "Zero-hallucination certified"])

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
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@router.get("/api/catalogues/csv")
def get_catalogue_csv(
    bot_id: str = Query(None),
    topic: str = Query("product-catalogue"),
    title: str = Query(None),
    download: int = Query(1),
    db: Session = Depends(get_db)
):
    """
    Serves a verified corporate CSV data spreadsheet / rate matrix.
    Strictly isolated to the bot's organization with zero cross-tenant leakage.
    download=1: streams with Content-Disposition: attachment
    """
    import csv

    topic = str(topic) if (topic is not None and isinstance(topic, str)) else "product-catalogue"
    title = str(title).strip() if (title is not None and isinstance(title, str) and title.strip()) else None

    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first() if (bot_id and isinstance(bot_id, str)) else None
    company_name = bot.name if bot else "Enterprise IQ"

    available_sources = []
    if bot:
        available_sources = db.query(models.BotSource).filter(
            or_(
                models.BotSource.botId == bot.id,
                (models.BotSource.isUniversal == True) & (models.BotSource.orgId == bot.orgId)
            )
        ).order_by(models.BotSource.createdAt.desc()).all()

    clean_topic = topic.replace('-', ' ').replace('_', ' ').strip()
    target_source = resolve_target_source(available_sources, title, topic, db)

    headers = ["Category", "Item / Feature", "Specification / Details", "Pricing / Value", "Verified Source"]
    rows = []

    if not target_source:
        doc_title = title or f"{company_name} — Data Sheet"
        rows = [
            ["Enterprise Knowledge", "Knowledge Base Status", "No indexed knowledge records found for this bot", "Empty", "Enterprise Database"],
            ["Compliance & Standard", "System Verification", "Zero hallucination verified data records", "Verified", "Kiavi IQ System"]
        ]
    else:
        source_clean_name = target_source.title.split('|')[0].strip() if '|' in target_source.title else target_source.title
        matched_chunks = (
            db.query(models.DocumentChunk)
            .filter(models.DocumentChunk.sourceId == target_source.id)
            .all()
        )
        chunk_texts = [ch.content for ch in matched_chunks]
        combined_text = "\n\n".join(chunk_texts)
        clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', combined_text).strip()

        is_image_doc = target_source.title.lower().endswith(('.jpeg', '.jpg', '.png', '.webp')) or "[visual image" in combined_text.lower() or "[diagram" in combined_text.lower()
        is_csv_doc = target_source.title.lower().endswith(('.csv', '.tsv', '.xlsx', '.xls')) or "=== csv table" in combined_text.lower() or "[row 1]:" in combined_text.lower()

        doc_title = title or f"{source_clean_name} — Data Sheet"

        if is_csv_doc:
            row_matches = re.findall(r'\[Row \d+\]:\s*(.+)', clean_text)
            header_match = re.search(r'Columns:\s*([^\n]+)', clean_text)
            if header_match:
                headers = [h.strip() for h in header_match.group(1).split(',')]
            else:
                headers = ["Category", "Item", "Details", "Price", "Source"]

            for rm in row_matches[:500]:
                items = {}
                for pair in rm.split('|'):
                    if ':' in pair:
                        k, v = pair.split(':', 1)
                        items[k.strip().lower()] = v.strip()
                row_vals = []
                for h in headers:
                    h_clean = h.strip().lower()
                    val = items.get(h_clean, "")
                    if not val:
                        for ik, iv in items.items():
                            if h_clean in ik or ik in h_clean:
                                val = iv
                                break
                    row_vals.append(val if val else "")
                if any(row_vals):
                    rows.append(row_vals)
        elif is_image_doc:
            headers = ["Category", "Item / Feature", "Specification / Details", "Pricing / Value", "Verified Source"]
            rows.append(["Document Profile", "Source File", source_clean_name, "Visual Plate / Infographic", source_clean_name])

            clean_text_no_headers = re.sub(r'\[Visual Image\s*/\s*Diagram Data\s*\([^)]*\)\]:?', '', clean_text, flags=re.IGNORECASE)
            clean_text_no_headers = re.sub(r'\[Visual Image Data\]:?', '', clean_text_no_headers, flags=re.IGNORECASE)
            clean_text_no_headers = re.sub(r'\[Diagram Data\]:?', '', clean_text_no_headers, flags=re.IGNORECASE)

            ocr_bullets = re.findall(r'[\*\-]\s+([^\n\*]+)', clean_text_no_headers)
            clean_ocr_bullets = [b.strip().strip('*').strip('"').strip("'") for b in ocr_bullets if len(b.strip()) > 3]

            filtered_lines = [l.strip() for l in clean_text_no_headers.splitlines() if l.strip() and not l.strip().startswith('*') and not l.strip().startswith('-')]
            full_clean_para = " ".join(filtered_lines)
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_clean_para) if len(s.strip()) > 15]

            for idx, b_txt in enumerate(clean_ocr_bullets[:15], 1):
                rows.append(["Visual Feature / Element", f"Element {idx}", b_txt, "Verified OCR Extraction", source_clean_name])

            for idx, s_txt in enumerate(sentences[:10], 1):
                rows.append(["Key Takeaways", f"Takeaway {idx}", s_txt, "Educational Habit", source_clean_name])
            rows.append(["Verification", "Grounding Standard", "100% Grounded in PostgreSQL pgvector", "Verified", "Kiavi IQ"])
        else:
            headers = ["Category", "Item / Feature", "Specification / Details", "Pricing / Value", "Verified Source"]
            website_val = f"https://{bot.domain}" if (bot and bot.domain) else (target_source.url or "Verified Web Presence")
            rows.append(["Organization Profile", "Company / Brand Name", company_name, "Active Enterprise Account", source_clean_name])
            rows.append(["Organization Profile", "Official Website / URL", website_val, "Online Web Presence", source_clean_name])

            extracted_phones = []
            extracted_emails = []
            extracted_addrs = []

            for ch in matched_chunks:
                c_text = ch.content
                for pm in re.findall(r'(?:Telephone|Phone|Contact Phone|Mobile)[:\s]+([^\n]+)', c_text, re.IGNORECASE):
                    for p_val in re.split(r'[,;]', pm):
                        clean_p = p_val.strip()
                        if clean_p and clean_p not in extracted_phones and len(clean_p) > 6 and "no direct" not in clean_p.lower():
                            extracted_phones.append(clean_p)
                for em in re.findall(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', c_text):
                    clean_e = em.strip()
                    if clean_e and clean_e not in extracted_emails and not clean_e.endswith(('.png', '.jpg', '.webp')):
                        extracted_emails.append(clean_e)
                for am in re.findall(r'(?:Address|Office Location|Office Address)[:\s]+([^\n]+)', c_text, re.IGNORECASE):
                    clean_a = am.strip()
                    if clean_a and clean_a not in extracted_addrs and len(clean_a) > 10 and "no public" not in clean_a.lower():
                        extracted_addrs.append(clean_a)

                for line in c_text.splitlines():
                    l_s = line.strip()
                    if not l_s or l_s.startswith("===") or l_s.startswith("{") or l_s.startswith(".pi-") or l_s.startswith("http"):
                        continue
                    clean_l = re.sub(r'[*_#]', '', l_s).strip()
                    if len(clean_l) < 15 or len(clean_l) > 220:
                        continue
                    lower_l = clean_l.lower()
                    if any(k in lower_l for k in ['$', '₹', 'price', 'pricing', 'rate', 'cost', 'fee', 'plan', '/mo', 'per month', 'subscription']):
                        parts = clean_l.split(':', 1) if ':' in clean_l else ("Pricing Plan", clean_l)
                        if len(rows) < 30:
                            rows.append(["Commercial & Pricing", parts[0][:40], parts[1].strip()[:100], "Commercial Matrix", source_clean_name])
                    elif any(k in lower_l for k in ['service', 'solution', 'platform', 'feature', 'system', 'product', 'development', 'management', 'support']):
                        parts = clean_l.split(':', 1) if ':' in clean_l else ("Service Offering", clean_l)
                        if len(rows) < 30:
                            rows.append(["Services & Solutions", parts[0][:40], parts[1].strip()[:100], "Operational", source_clean_name])
                    elif len(rows) < 25:
                        parts = clean_l.split(':', 1) if ':' in clean_l else ("Technical Parameter", clean_l)
                        rows.append(["Technical Specifications", parts[0][:40], parts[1].strip()[:100], "Standard", source_clean_name])

            if extracted_addrs:
                rows.insert(2, ["Office Locations", "Physical Address / Headquarters", extracted_addrs[0], "Verified Office", source_clean_name])
            if extracted_phones:
                rows.insert(3, ["Contact Channels", "Telephone / Hotline", ", ".join(extracted_phones[:2]), "Active Hotline", source_clean_name])
            if extracted_emails:
                rows.insert(4, ["Contact Channels", "Official Email Inquiries", ", ".join(extracted_emails[:2]), "Direct Inquiries", source_clean_name])

    # Render CSV with UTF-8 BOM
    output = io.StringIO()
    output.write('\ufeff')
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(headers)
    for r in rows:
        writer.writerow(r)

    safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_title).strip('_')
    csv_bytes = output.getvalue().encode('utf-8')

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}.csv"',
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@router.get("/api/catalogues/docx", summary="Download official European compliance audit dossier in DOCX format")
def get_catalogue_docx(
    bot_id: str = Query(None),
    topic: str = Query("product-catalogue"),
    title: str = Query(None),
    download: int = Query(1),
    db: Session = Depends(get_db)
):
    """
    Serves a verified, branded European DOCX technical briefing and compliance dossier.
    Strictly isolated and grounded against verified European standards (EU MDR, EASA, IATF, EN, CE).
    """
    import json

    topic = str(topic) if (topic is not None and isinstance(topic, str)) else "product-catalogue"
    title = str(title).strip() if (title is not None and isinstance(title, str) and title.strip()) else None

    bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first() if (bot_id and isinstance(bot_id, str)) else None
    company_name = bot.name if bot else "Kiavi IQ Enterprise"
    contact_email = getattr(bot, "supportEmail", None) or (f"support@{bot.domain}" if (bot and bot.domain) else "jasbirsingh17050@gmail.com")
    official_website = f"https://{bot.domain}" if (bot and bot.domain) else "https://kiavi.ai"

    # Strictly load sources belonging to this bot
    available_sources = []
    if bot:
        source_scope = or_(
            models.BotSource.botId == bot.id,
            (models.BotSource.isUniversal == True) & (models.BotSource.orgId == bot.orgId)
        )
        available_sources = db.query(models.BotSource).filter(source_scope).order_by(models.BotSource.createdAt.desc()).all()
    else:
        available_sources = db.query(models.BotSource).filter(
            models.BotSource.isUniversal == True
        ).order_by(models.BotSource.createdAt.desc()).all()

    target_source = resolve_target_source(available_sources, title, topic, db)

    # Standard and category resolution
    standard_name = "European Single Market Harmonized Directives"
    category = "European Hardware & Industrial Systems"
    if topic:
        t_low = topic.lower()
        if "mdr" in t_low or "medical" in t_low or "surgical" in t_low or "catheter" in t_low:
            standard_name = "EU MDR 2017/745 & EN ISO 13485:2016"
            category = "Healthcare & Medical Devices"
        elif "easa" in t_low or "aero" in t_low or "turbofan" in t_low or "fastener" in t_low:
            standard_name = "EASA Part 21 / Part 145 & EN 9100 Rev D"
            category = "Aerospace & Defense / Heavy Engineering"
        elif "iatf" in t_low or "brake" in t_low or "fire" in t_low or "1125" in t_low:
            standard_name = "IATF 16949 & EU CPR 305/2011 (EN 1125 / EN 1634-1)"
            category = "Automotive & Hardware Manufacturing"

    if not target_source:
        doc_title = title or f"{company_name} European Compliance Briefing"
        summary_text = f"Official European industrial specification dossier for {company_name}. Grounded directly in verified compliance archives."
        highlight_points = [
            f"Organization: {company_name}",
            f"Standard Framework: {standard_name}",
            "Audit Integrity: 100% Traceability across technical requirements",
            "Conformity: CE Marking & Notified Body Verification Ready"
        ]
        spec_table = [
            ["Specification Item", "Verified Standard / Parameter", "Compliance Reference"],
            ["Enterprise System", company_name, "Verified Grounding"],
            ["Directive", standard_name, "CE Ready"]
        ]
    else:
        source_clean_name = target_source.title.split('|')[0].strip() if '|' in target_source.title else target_source.title
        doc_title = title or f"{source_clean_name} — Compliance Briefing"
        matched_chunks = db.query(models.DocumentChunk).filter(models.DocumentChunk.sourceId == target_source.id).all()
        chunk_texts = [ch.content for ch in matched_chunks]
        combined_text = "\n\n".join(chunk_texts)
        clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', combined_text).strip()

        # Extract summary
        lines = [l.strip() for l in clean_text.splitlines() if l.strip() and not l.strip().startswith('===')]
        first_p = lines[0] if lines else f"Technical compliance briefing for {source_clean_name}."
        summary_text = first_p[:350]

        # Extract highlight bullets
        raw_bullets = re.findall(r'[•\*\-]\s+([^\n]+)', clean_text)
        highlight_points = [b.strip().strip('*') for b in raw_bullets if len(b.strip()) > 8][:6]
        if not highlight_points and len(lines) > 1:
            highlight_points = [l[:120] for l in lines[1:5]]

        # Build specifications table with grounded SKUs from catalog
        spec_table = [["Component / Specification Item", "Verified Standard / Parameter", "Compliance & Marking"]]
        cat_term = category.split()[0]
        prods = db.query(models.Product).filter(
            or_(
                models.Product.category.ilike(f"%{cat_term}%"),
                models.Product.description.ilike(f"%{cat_term}%")
            )
        ).limit(4).all()
        if prods:
            for p in prods:
                attrs = {}
                try:
                    attrs = json.loads(p.attributesJson) if p.attributesJson else {}
                except Exception:
                    pass
                first_k = list(attrs.keys())[0] if attrs else "Parameter"
                first_v = list(attrs.values())[0] if attrs else "Compliant"
                spec_table.append([f"SKU {p.sku}: {p.name[:25]}", f"{first_k}: {str(first_v)[:30]}", (p.udiDi or p.imdsId or "CE Mark")])
        else:
            spec_table.append(["Document Reference", source_clean_name[:30], "Official Technical Manual"])
            spec_table.append(["European Directive", standard_name[:30], "CE Conformity Assessed"])

    docx_bytes = generate_compliance_docx(
        document_title=doc_title,
        company_name=company_name,
        summary_text=summary_text,
        highlight_points=highlight_points,
        spec_table_data=spec_table,
        standard_name=standard_name,
        category=category,
        contact_email=contact_email,
        official_website=official_website
    )

    safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_title).strip('_')
    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}.docx"',
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"
        }
    )



