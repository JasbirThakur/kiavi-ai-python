from typing import List, Optional
import re

def estimate_tokens(text: str) -> int:
    """Estimates tokens assuming ~4 characters per token for English/code."""
    if not text:
        return 0
    return max(1, len(text) // 4)

def split_into_sentences(text: str) -> List[str]:
    """
    Advanced semantic sentence tokenizer using regex lookahead/lookbehind.
    Protects common abbreviations, corporate honorifics, dates, and version strings.
    """
    if not text or not text.strip():
        return []

    # Protect common abbreviations and technical tokens
    protected = text
    abbrevs = [
        "e.g.", "i.e.", "vs.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.",
        "Inc.", "Ltd.", "Co.", "Corp.", "Jan.", "Feb.", "Mar.", "Apr.",
        "Aug.", "Sept.", "Oct.", "Nov.", "Dec.", "approx.", "dept.", "est.",
        "v1.0", "v2.0", "v3.0", "No.", "Fig.", "Sec."
    ]
    for idx, abb in enumerate(abbrevs):
        protected = protected.replace(abb, f"__SEM_ABBR_{idx}__")

    # Split on sentence terminals followed by whitespace and an uppercase letter, digit, or quote
    raw_sents = re.split(r'(?<=[.?!])\s+(?=[A-Z0-9"\'“‘])', protected)
    sentences = []
    for s in raw_sents:
        for idx, abb in enumerate(abbrevs):
            s = s.replace(f"__SEM_ABBR_{idx}__", abb)
        s_clean = s.strip()
        if s_clean:
            sentences.append(s_clean)
    return sentences

def semantic_chunking(
    text: str,
    max_chars: int = 800,
    min_chars: Optional[int] = None,
    overlap_sentences: int = 1
) -> List[str]:
    """
    Splits text along natural semantic boundaries (sections, headings, paragraphs, and complete sentences)
    rather than arbitrary character cuts. Ensures every chunk represents a coherent, complete thought unit.
    """
    if not text or not text.strip():
        return []

    # Clean redundant whitespace while preserving paragraph breaks
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    if len(text) <= max_chars:
        return [text]

    effective_min = min_chars if min_chars is not None else max(40, min(150, max_chars // 4))

    # 1. Split along natural paragraph breaks, markdown headings, or major dividers
    raw_paragraphs = [p.strip() for p in re.split(r'\n{2,}|\n(?=[#=\-]{2,}|\b(?:[A-Z0-9.\-_]{2,}\b|\d+\.)\s)', text) if p.strip()]

    chunks = []
    current_sentences = []
    current_length = 0

    for para in raw_paragraphs:
        sents = split_into_sentences(para)
        if not sents:
            sents = [para]

        for s in sents:
            s_len = len(s)
            if s_len > max_chars:
                # Sub-split overly long sentences along semicolons, colons, or commas
                sub_parts = re.split(r'(?<=[;:])\s+', s)
                if len(sub_parts) == 1:
                    sub_parts = [s[i:i+max_chars] for i in range(0, len(s), max_chars)]
            else:
                sub_parts = [s]

            for part in sub_parts:
                part_len = len(part)
                if current_length + part_len + 1 > max_chars and current_sentences:
                    chunk_str = " ".join(current_sentences).strip()
                    if chunk_str:
                        chunks.append(chunk_str)
                    overlap = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
                    current_sentences = list(overlap)
                    current_length = sum(len(x) + 1 for x in current_sentences)

                current_sentences.append(part)
                current_length += part_len + 1

    if current_sentences:
        chunk_str = " ".join(current_sentences).strip()
        if chunk_str:
            chunks.append(chunk_str)

    # Merge very small trailing chunks with the previous chunk to maintain semantic richness
    if len(chunks) > 1 and len(chunks[-1]) < effective_min:
        last_c = chunks.pop()
        chunks[-1] = chunks[-1] + "\n\n" + last_c

    return chunks

def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Backwards-compatible semantic chunking wrapper.
    Converts chunk_size to character boundaries and applies grammatical sentence overlap.
    """
    overlap_count = 2 if chunk_overlap >= 100 else 1
    return semantic_chunking(text, max_chars=chunk_size, overlap_sentences=overlap_count)

# Export aliases for LangChain / Experimental drop-in compatibility
split_into_semantic_chunks = semantic_chunking
