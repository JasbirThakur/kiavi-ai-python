import urllib.request
import urllib.parse
import json
import time

BOT_ID = "6f39b7c8-dd9e-4f50-b502-b6969dac19b8"  # Navjot bot (has 30 rich sources)
VEDAONE_BOT_ID = "ebaf27b5-47bb-4095-a35b-e77ce8561b07"

test_cases = [
    {
        "category": "1. Casual Greeting (English)",
        "query": "hi",
        "bot_id": BOT_ID,
        "expect_tone": "Warm, natural greeting without corporate fluff"
    },
    {
        "category": "2. Casual Greeting (Hindi/Hinglish)",
        "query": "namaste bhai",
        "bot_id": BOT_ID,
        "expect_tone": "Natural friendly Hindi greeting"
    },
    {
        "category": "3. Identity Defense - Are You AI/Bot? (English)",
        "query": "are you an ai or a bot?",
        "bot_id": BOT_ID,
        "expect_tone": "Real person on the team defense, zero AI admission"
    },
    {
        "category": "4. Identity Defense - Are You Real? (Hindi)",
        "query": "kya tum ai ho ya real insaan?",
        "bot_id": BOT_ID,
        "expect_tone": "Haha nahi main team se hoon, zero AI admission"
    },
    {
        "category": "5. Wellbeing & Colleague Check-in (English)",
        "query": "how are you doing today?",
        "bot_id": BOT_ID,
        "expect_tone": "Doing good, thanks for asking! Colleague tone"
    },
    {
        "category": "6. Wellbeing & Colleague Check-in (Hindi)",
        "query": "aur bhai kaisa chal raha hai?",
        "bot_id": BOT_ID,
        "expect_tone": "Natural Hindi chit-chat response"
    },
    {
        "category": "7. Assistance Request",
        "query": "can you help me out?",
        "bot_id": BOT_ID,
        "expect_tone": "Sure happy to help, what are you working on?"
    },
    {
        "category": "8. Gratitude",
        "query": "thank you so much, that was awesome!",
        "bot_id": BOT_ID,
        "expect_tone": "Anytime! Glad I could help."
    },
    {
        "category": "9. Farewell",
        "query": "bye, talk to you later!",
        "bot_id": BOT_ID,
        "expect_tone": "Take care! Have a good one."
    },
    {
        "category": "10. Documentation Overview",
        "query": "Documentation & FAQs",
        "bot_id": BOT_ID,
        "expect_tone": "Clean bullets of indexed documentation without bold headers"
    },
    {
        "category": "11. Contact Support Channels",
        "query": "Contact Support",
        "bot_id": BOT_ID,
        "expect_tone": "Clear contact info, direct team callback"
    },
    {
        "category": "12. Capability Test / Flowchart Offer",
        "query": "can you make diagrams and flowcharts?",
        "bot_id": BOT_ID,
        "expect_tone": "Yeah definitely, list what you need and I'll generate it"
    },
    {
        "category": "13. Unknown Query / Zero Hallucination (English)",
        "query": "what is the policy for deep space exploration in 2099?",
        "bot_id": BOT_ID,
        "expect_tone": "Don't have that in front of me, will check with team"
    },
    {
        "category": "14. Unknown Query / Zero Hallucination (Hindi)",
        "query": "Mars par real estate khareedne ki policy kya hai?",
        "bot_id": BOT_ID,
        "expect_tone": "Mere paas abhi ye detail nahi hai, team se confirm karke batata hoon"
    },
    {
        "category": "15. Grounded Factual Knowledge (From Indexed SLA Policy)",
        "query": "What is the support SLA response time?",
        "bot_id": BOT_ID,
        "expect_tone": "Direct factual answer from doc, no marketing fluff"
    },
    {
        "category": "16. Interactive Mermaid Diagram Generation",
        "query": "Can you show me a flowchart diagram of the process?",
        "bot_id": VEDAONE_BOT_ID,
        "expect_tone": "Clean graph TD Mermaid diagram with double-quoted node syntax"
    }
]

def query_chat_stream(bot_id, question):
    data = urllib.parse.urlencode({
        "bot_id": bot_id,
        "question": question
    }).encode("utf-8")

    req = urllib.request.Request("http://127.0.0.1:8000/api/public/chat/stream", data=data)
    try:
        resp = urllib.request.urlopen(req, timeout=40)
        tokens = []
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                payload = json.loads(line_str[6:])
                if payload.get("type") == "token":
                    tokens.append(payload.get("content", ""))
        return "".join(tokens).strip()
    except Exception as e:
        return f"[ERROR]: {e}"

def audit_anti_patterns(text):
    banned_ai_words = ['delve', 'landscape', 'pivotal', 'robust', 'testament', 'foster', 'enhance', 'bolster', 'showcase', 'intricate', 'tapestry', 'vibrant', 'game-changer', 'seamless', 'revolutionary']
    violations = []
    
    if "—" in text:
        violations.append("Contains Em-dash (—)")
    if any(b in text.lower() for b in banned_ai_words):
        matched = [b for b in banned_ai_words if b in text.lower()]
        violations.append(f"Contains banned AI words: {matched}")
    if any(em in text for em in ['🚀', '✨', '💡', '🔥', '🎉', '🤖']):
        violations.append("Contains decorative emoji spam")
    if any(cl in text for cl in ['I hope this helps', 'Certainly!', "I'd be happy to help", 'Great question!']):
        violations.append("Contains robotic chatbot residue")
    if "as an ai" in text.lower() or "as a language model" in text.lower():
        violations.append("Contains AI identity disclaimer")
        
    return violations

print("=================================================================")
print("  EXECUTING LIVE RESPONSES TEST SUITE FOR HUMANIZER TRANSFORMATION")
print("=================================================================\n")

results = []

for idx, tc in enumerate(test_cases, 1):
    cat = tc["category"]
    q = tc["query"]
    bot = tc["bot_id"]
    
    print(f"[{idx}/{len(test_cases)}] Testing: {cat} ('{q}')...", flush=True)
    ans = query_chat_stream(bot, q)
    violations = audit_anti_patterns(ans)
    if ans.startswith("[ERROR]"):
        violations.append(ans)
    
    res_entry = {
        "index": idx,
        "category": cat,
        "query": q,
        "response": ans,
        "violations": violations,
        "status": "PASS" if not violations else "FAIL (Anti-Pattern or Error Detected)"
    }
    results.append(res_entry)
    time.sleep(0.5)

print("\n\n=================================================================")
print("                      DETAILED TEST REPORT")
print("=================================================================\n")

for r in results:
    print(f"### Case {r['index']}: {r['category']}")
    print(f"**Query Asked**: \"{r['query']}\"")
    print(f"**Status**: {'✅ ' + r['status'] if r['status'] == 'PASS' else '❌ ' + r['status']}")
    if r['violations']:
        print(f"**Violations**: {r['violations']}")
    print(f"**Live Bot Response**:\n> {r['response']}\n")
    print("-" * 65 + "\n")

all_passed = all(r['status'] == 'PASS' for r in results)
print(f"Summary: {len(results)} cases tested. {'All PASSED ✅' if all_passed else 'Some failed ❌'}")

