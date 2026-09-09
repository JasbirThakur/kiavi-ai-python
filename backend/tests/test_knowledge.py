from app.services.chunking import chunk_text, estimate_tokens
from app.services.ingestion import get_chunks_from_text

def test_chunking():
    sample_text = (
        "Alorica is a leading global provider of next-generation customer experience (CX) solutions.\n\n"
        "With tens of thousands of employees across the globe, we create insanely great digital customer experiences.\n\n"
        "Our clients include Fortune 500 companies in healthcare, financial services, and technology."
    )
    chunks = chunk_text(sample_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)
    assert "Alorica" in chunks[0]

def test_token_estimation():
    text = "Hello world! This is a test."
    tokens = estimate_tokens(text)
    assert tokens > 0

if __name__ == "__main__":
    test_chunking()
    test_token_estimation()
    print("✅ test_knowledge passed!")
