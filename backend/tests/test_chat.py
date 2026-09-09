from app.services.rag import clean_llm_text

def test_clean_llm_text():
    raw_thought = "<think>Analyzing user question about pricing...</think>Our pricing starts at $99/mo."
    assert clean_llm_text(raw_thought) == "Our pricing starts at $99/mo."

    with_answer_prefix = "<think>scratchpad</think>ANSWER: The delivery time is 2-3 business days."
    assert clean_llm_text(with_answer_prefix) == "The delivery time is 2-3 business days."

    preamble = "Okay, looking through the provided knowledge base. Appdeft provides custom software development."
    cleaned = clean_llm_text(preamble)
    assert "Appdeft provides custom software development." in cleaned

if __name__ == "__main__":
    test_clean_llm_text()
    print("✅ test_chat passed!")
