from typing import List
import re

def estimate_tokens(text: str) -> int:
    """Estimates tokens assuming ~4 characters per token for English/code."""
    return max(1, len(text) // 4)

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Recursively chunks text by paragraph, sentence, and word boundaries.
    """
    if not text or not text.strip():
        return []

    # Clean redundant whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Look for suitable split point: paragraph -> sentence -> space
            para_break = text.rfind('\n\n', start, end)
            if para_break != -1 and para_break > start + (chunk_size // 3):
                end = para_break
            else:
                sent_break = max(
                    text.rfind('. ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end)
                )
                if sent_break != -1 and sent_break > start + (chunk_size // 3):
                    end = sent_break + 1
                else:
                    space_break = text.rfind(' ', start, end)
                    if space_break != -1 and space_break > start + (chunk_size // 3):
                        end = space_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap if end < text_len else text_len
        if start >= end:
            start = end

    return chunks

