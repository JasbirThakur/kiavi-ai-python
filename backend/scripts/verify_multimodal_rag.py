import asyncio
import json
import re
from app.db.database import SessionLocal
from app.db import models
from app.services.rag import stream_rag_pipeline

async def run_pipeline_collector(bot_id: str, question: str, conv_id: str = None, user_name: str = None):
    db = SessionLocal()
    tokens = []
    followup = None
    confidence = 0.0
    full_text = ""
    try:
        async for raw_chunk in stream_rag_pipeline(bot_id, question, db, conversation_id=conv_id, user_name=user_name):
            for line in raw_chunk.strip().split("\n"):
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "token":
                            tokens.append(data.get("content", ""))
                        elif data.get("type") == "done":
                            full_text = data.get("full_text", "".join(tokens))
                            followup = data.get("followup")
                        elif data.get("type") == "start":
                            confidence = data.get("confidence", 0.0)
                    except Exception:
                        pass
    finally:
        db.close()
    if not full_text:
        full_text = "".join(tokens)
    return full_text, followup, confidence

async def main():
    db = SessionLocal()
    bot = db.query(models.Bot).first()
    if not bot:
        print("❌ No bot found in database")
        return
    bot_id = bot.id
    print(f"🤖 Testing Bot: '{bot.name}' (ID: {bot_id})")

    test_results = []

    # Test 1: Conversational Greeting
    print("\n--- TEST 1: Conversational Greeting ---")
    txt1, f1, _ = await run_pipeline_collector(bot_id, "Hello, how are you?")
    print(f"Response: {txt1[:150]}...")
    no_refusal_1 = "text-based" not in txt1.lower() and "cannot display" not in txt1.lower()
    friendly_1 = any(w in txt1.lower() for w in ["great", "good", "hello", "hi", "assist"])
    print(f"✅ Friendly: {friendly_1}, No refusal: {no_refusal_1}")
    test_results.append(("Conversational Greeting", no_refusal_1 and friendly_1))

    # Test 2: The exact question from user screenshot:
    # First turn: Cartography question
    # Second turn: "Do you have any digram or visual representation regarding that?"
    print("\n--- TEST 2: Multi-turn Cartography + Diagram Follow-up ---")
    conv = models.Conversation(botId=bot_id, isTest=True)
    db.add(conv)
    db.commit()
    conv_id = conv.id

    # Turn 1
    t1_q = "What is cartography and mapping?"
    txt_t1, _, _ = await run_pipeline_collector(bot_id, t1_q, conv_id=conv_id, user_name="Jasbir")
    msg1_u = models.Message(conversationId=conv_id, role="USER", content=t1_q)
    msg1_b = models.Message(conversationId=conv_id, role="BOT", content=txt_t1)
    db.add(msg1_u)
    db.add(msg1_b)
    db.commit()
    print(f"Turn 1 Response: {txt_t1[:120]}...")

    # Turn 2: Exact user query with typo "digram"
    t2_q = "Do you have any digram or visual representation regarding that?"
    txt_t2, f_t2, c_t2 = await run_pipeline_collector(bot_id, t2_q, conv_id=conv_id, user_name="Jasbir")
    print(f"Turn 2 Confidence: {c_t2}")
    print(f"Turn 2 Full Response:\n{txt_t2}\n")

    # Verification criteria for Turn 2
    has_refusal = "text-based model" in txt_t2.lower() or "do not have the capability to display images" in txt_t2.lower()
    has_visual = ("![" in txt_t2) or ("```mermaid" in txt_t2) or ("🎥" in txt_t2)
    print(f"Turn 2 Refusal Detected: {has_refusal}")
    print(f"Turn 2 Visual Asset Included: {has_visual}")
    test_results.append(("Multi-turn Diagram Intent (Zero Refusal + Visual Present)", (not has_refusal) and has_visual))

    # Test 3: Technical Sterilization Flowchart request
    print("\n--- TEST 3: Sterilization Flowchart Request ---")
    t3_q = "Provide a sterilization workflow flowchart"
    txt3, _, _ = await run_pipeline_collector(bot_id, t3_q, user_name="Jasbir")
    print(f"Turn 3 snippet: {txt3[:200]}...")
    has_mermaid = "```mermaid" in txt3
    quotes_in_nodes = '["' in txt3 or "graph TD" in txt3
    print(f"Mermaid Present: {has_mermaid}, Proper Quotes: {quotes_in_nodes}")
    test_results.append(("Interactive Mermaid Flowchart", has_mermaid))

    # Test 4: Video demonstration request
    print("\n--- TEST 4: Video Demonstration Request ---")
    t4_q = "Do you have a video walkthrough or tutorial?"
    txt4, _, _ = await run_pipeline_collector(bot_id, t4_q, user_name="Jasbir")
    print(f"Turn 4 snippet: {txt4[:200]}...")
    no_refusal_4 = "text-based" not in txt4.lower()
    print(f"No refusal in video response: {no_refusal_4}")
    test_results.append(("Video Request Zero Refusal", no_refusal_4))

    print("\n================ TEST SUMMARY ================")
    all_passed = True
    for name, passed in test_results:
        status = "PASSED ✅" if passed else "FAILED ❌"
        if not passed:
            all_passed = False
        print(f"{name}: {status}")
    
    if all_passed:
        print("\n🎉 ALL MULTIMEDIA & CONVERSATIONAL TESTS PASSED!")
    else:
        print("\n⚠️ SOME TESTS FAILED!")

    db.close()

if __name__ == "__main__":
    asyncio.run(main())
