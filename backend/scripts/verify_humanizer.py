import sys
import re

# Add backend directory to sys.path
sys.path.insert(0, '/app')

from app.services.rag import (
    classify_conversational_intent,
    sanitize_humanizer_text,
    sanitize_mermaid_syntax,
    clean_llm_text,
    get_chat_system_prompt
)

def run_tests():
    passed = 0
    failed = 0

    def assert_test(name, condition, details=""):
        nonlocal passed, failed
        if condition:
            print(f"✅ PASS: {name}")
            passed += 1
        else:
            print(f"❌ FAIL: {name} - {details}")
            failed += 1

    print("\n--- TEST 1: Intent Classification & Human Identity Defense ---")
    ai_queries = [
        "are you an ai",
        "are you a bot",
        "are you ai or human",
        "are you real",
        "r u a bot",
        "kya tum ai ho",
        "are you an ai or a bot?"
    ]
    for q in ai_queries:
        is_conv, intent = classify_conversational_intent(q, "Kiavi")
        assert_test(f"Intent for '{q}' is 'identity'", is_conv and intent == 'identity', f"got is_conv={is_conv}, intent={intent}")

    casual_queries = [
        ("hi", "greeting"),
        ("hey", "greeting"),
        ("how are you", "wellbeing"),
        ("kaise ho", "wellbeing"),
        ("thank you", "gratitude"),
        ("bye", "farewell"),
        ("can you help me", "assistance"),
        ("documentation & faqs", "documentation_overview"),
        ("contact support", "contact_support")
    ]
    for q, expected in casual_queries:
        is_conv, intent = classify_conversational_intent(q, "Kiavi")
        assert_test(f"Intent for '{q}' is '{expected}'", is_conv and intent == expected, f"got is_conv={is_conv}, intent={intent}")

    print("\n--- TEST 2: Sanitize Humanizer Text (Anti-Patterns Removal) ---")
    # Rule 8: No Em-dashes
    em_dash_sample = "Here is the item — which arrived yesterday — ready for review."
    sanitized_dash = sanitize_humanizer_text(em_dash_sample)
    assert_test("Em-dash replaced", "—" not in sanitized_dash and "," in sanitized_dash, f"Result: {sanitized_dash}")

    # Rule 19 & 20: No Decorative Emojis, Straight Quotes, Keep Video 🎥
    emoji_sample = '“We have groundbreaking updates! 🚀✨💡 Check the video: 🎥 [Watch Video](https://test.com)”'
    sanitized_emoji = sanitize_humanizer_text(emoji_sample)
    assert_test("Decorative emojis removed, 🎥 preserved", "🚀" not in sanitized_emoji and "✨" not in sanitized_emoji and "🎥" in sanitized_emoji, f"Result: {sanitized_emoji}")
    assert_test("Straight quotes applied", '“' not in sanitized_emoji and '"' in sanitized_emoji, f"Result: {sanitized_emoji}")

    # Rule 18: No bold list headers
    bold_list_sample = "• **Specification:** 134°C autoclave cycle\n• **Standard:** EN 285"
    sanitized_list = sanitize_humanizer_text(bold_list_sample)
    assert_test("Bold list headers removed", "**Specification:**" not in sanitized_list and "• Specification: 134°C" in sanitized_list, f"Result: {sanitized_list}")

    # Rule 4, 22, 23: Strip AI Preambles, Closers, Refusals
    ai_cliches = "Certainly! Here's what you need to know: the system is ready. I hope this helps!"
    sanitized_cliches = sanitize_humanizer_text(ai_cliches)
    assert_test("AI preambles & closers stripped", "Certainly" not in sanitized_cliches and "I hope this helps" not in sanitized_cliches, f"Result: {sanitized_cliches}")

    refusal_sample = "Unfortunately, I am a text-based model and do not have the capability to display images directly. However, here is the spec."
    sanitized_refusal = sanitize_humanizer_text(refusal_sample)
    assert_test("Refusal stripped", "text-based model" not in sanitized_refusal and "spec" in sanitized_refusal, f"Result: {sanitized_refusal}")

    print("\n--- TEST 3: System Prompt Compliance Check ---")
    prompt = get_chat_system_prompt()
    assert_test("Prompt contains CRITICAL IDENTITY RULES", "NEVER say or admit that you are an AI" in prompt)
    assert_test("Prompt contains 25 AI ANTI-PATTERNS", "THE 25 AI ANTI-PATTERNS" in prompt)
    assert_test("Prompt bans AI buzzwords", "delve, landscape, pivotal" in prompt)
    assert_test("Prompt contains Human Friend tone guidelines", "Talk Like a Real Colleague" in prompt)
    assert_test("Prompt contains zero-refusal multimedia rule", "ZERO REFUSAL RULE" in prompt)
    assert_test("Prompt contains Mermaid flowchart standards", "graph TD" in prompt and "ZERO TEXT OVERLAP RULE" in prompt)

    print("\n--- TEST 4: Mermaid Flowchart Sanitization ---")
    sample_mermaid = """
    A -->|Checking hydraulic line pressure| B
    B --> C --> D
    """
    clean_flowchart = sanitize_mermaid_syntax(sample_mermaid)
    assert_test("Flowchart starts with graph TD", clean_flowchart.strip().startswith("graph TD"))
    assert_test("Chained arrows unrolled", "-->" in clean_flowchart and len(clean_flowchart.strip().splitlines()) >= 3)
    assert_test("Nodes formatted with double quotes", '["' in clean_flowchart and '"]' in clean_flowchart)

    print(f"\n==============================")
    print(f"Total Tests: {passed + failed} | Passed: {passed} | Failed: {failed}")
    print(f"==============================\n")
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
