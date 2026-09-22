import asyncio
import json
import re
from app.db.database import SessionLocal
from app.db import models
from app.services.rag import classify_conversational_intent, stream_rag_pipeline, generate_llm_response

def test_classifier():
    print("\n=== STEP 1: VERIFYING CONVERSATIONAL INTENT CLASSIFIER ===")
    test_cases = [
        ("what is this?", True, "identity"),
        ("what is this bot?", True, "identity"),
        ("what is this platform", True, "identity"),
        ("kya hai ye?", True, "identity"),
        ("ho are you?", True, "wellbeing"),
        ("wat is this?", True, "identity"),
        ("Documentation & FAQs", True, "documentation_overview"),
        ("documentation", True, "documentation_overview"),
        ("Contact Support", True, "contact_support"),
        ("how to contact", True, "contact_support"),
        ("Yeh test karega ki bot process workflows ko visual Mermaid diagrams me convert karta hai ya nahi.", True, "capability_test"),
        ("can you make diagrams", True, "capability_test"),
        ("Show me a diagram of the ISO 14971 risk management workflow described in Figure 10.5", False, ""),
        ("What is the thickness of ASTM A36 steel plate?", False, ""),
    ]

    all_passed = True
    for query, expected_is_conv, expected_type in test_cases:
        is_conv, intent_type = classify_conversational_intent(query, "Navjot")
        passed = (is_conv == expected_is_conv) and (intent_type == expected_type)
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"{status} | '{query}' -> is_conv={is_conv}, type='{intent_type}' (expected: {expected_is_conv}, '{expected_type}')")

    return all_passed

async def test_stream_pipeline():
    print("\n=== STEP 2: VERIFYING STREAM RAG PIPELINE SCENARIOS ===")
    db = SessionLocal()
    bot = db.query(models.Bot).filter(models.Bot.id == "6f39b7c8-dd9e-4f50-b502-b6969dac19b8").first()
    if not bot:
        bot = db.query(models.Bot).filter(models.Bot.name == "Navjot").first()
    bot_id = str(bot.id)
    print(f"Testing with Bot: '{bot.name}' (ID: {bot_id}, Org: {bot.orgId})")

    queries_to_test = [
        ("what is this?", "Identity & Meta Question"),
        ("ho are you?", "Typo Greeting / Wellbeing"),
        ("Documentation & FAQs", "Action Chip: Documentation Overview"),
        ("Contact Support", "Action Chip: Contact & Support"),
        ("Yeh test karega ki bot process workflows ko visual Mermaid diagrams me convert karta hai ya nahi.", "Capability Test Prompt"),
        ("Show me a diagram of the ISO 14971 risk management workflow described in Figure 10.5", "Interactive Diagram & Flowchart"),
        ("Does this company sell tickets to Mars?", "True Content Gap Fallback")
    ]

    for q, desc in queries_to_test:
        print(f"\n--- [Scenario: {desc}] ---")
        print(f"User Query: '{q}'")
        collected_tokens = []
        lead_form_emitted = False
        followup_emitted = False
        full_text = ""

        gen = stream_rag_pipeline(
            bot_id=bot_id,
            question=q,
            db=db,
            user_name="Sahil"
        )

        async for chunk in gen:
            chunk_str = chunk.strip()
            if not chunk_str.startswith("data: "):
                continue
            payload = json.loads(chunk_str[6:])
            ptype = payload.get("type")
            if ptype == "token":
                collected_tokens.append(payload.get("content", ""))
            elif ptype == "lead_form":
                lead_form_emitted = True
            elif ptype == "followup":
                followup_emitted = True
            elif ptype == "done":
                full_text = payload.get("full_text", "".join(collected_tokens))

        if not full_text:
            full_text = "".join(collected_tokens)

        print(f"Lead Form Emitted: {lead_form_emitted} | Followup Emitted: {followup_emitted}")
        print(f"Response Preview:\n{full_text[:350]}...\n")

        # Specific assertions
        if desc == "Identity & Meta Question":
            assert "AI assistant" in full_text or "Navjot" in full_text, "Identity missing bot role"
            assert "I cannot find the answer" not in full_text, "Inappropriately dropped to contact fallback!"
            print("✅ Verified: Rich conversational identity delivered without fallback!")

        elif desc == "Typo Greeting / Wellbeing":
            assert ("great" in full_text.lower() or "doing" in full_text.lower() or "theek" in full_text.lower() or "hello" in full_text.lower()), "Typo wellbeing failed"
            assert "I cannot find the answer" not in full_text, "Inappropriately dropped to contact fallback!"
            print("✅ Verified: Normalized typo and responded conversationally!")

        elif desc == "Action Chip: Documentation Overview":
            assert ("documentation" in full_text.lower() or "catalogue" in full_text.lower() or "indexed" in full_text.lower()), "Missing doc overview"
            assert "I cannot find the answer" not in full_text, "Inappropriately dropped to contact fallback!"
            print("✅ Verified: Dynamic documentation directory delivered with suggestions!")

        elif desc == "Action Chip: Contact & Support":
            assert ("support" in full_text.lower() or "email" in full_text.lower()), "Missing support channels"
            assert lead_form_emitted, "Lead form should be emitted for Contact Support"
            print("✅ Verified: Official support channels and lead form provided!")

        elif desc == "Capability Test Prompt":
            assert ("mermaid" in full_text.lower() or "diagram" in full_text.lower() or "workflow" in full_text.lower()), "Missing capability confirmation"
            assert "I cannot find the answer" not in full_text, "Inappropriately dropped to contact fallback!"
            print("✅ Verified: Responded affirmatively to capability test!")

        elif desc == "Interactive Diagram & Flowchart":
            assert "```mermaid" in full_text, "Mermaid diagram code block missing!"
            assert "graph TD" in full_text, "graph TD layout missing!"
            assert "Stage 1" in full_text or "Planning" in full_text, "Stages missing in diagram!"
            print("✅ Verified: Valid interactive Mermaid flowchart and stage specifications generated!")

        elif desc == "True Content Gap Fallback":
            assert lead_form_emitted, "Lead form missing for content gap"
            assert any(k in full_text.lower() for k in ["specializes in", "indexed", "knowledge base", "don't have", "do not have", "cannot find", "no information"]), "Missing active document summary in fallback"
            print("✅ Verified: Context-aware content gap fallback with lead form emitted!")

    db.close()
    print("\n🎉 ALL 7 SCENARIOS VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    if test_classifier():
        asyncio.run(test_stream_pipeline())
