import io
import re
from pypdf import PdfReader
import docx

def get_chunks_from_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Splits text into meaningful semantic segments (>24 chars)"""
    clean_text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    start = 0
    while start < len(clean_text):
        end = start + chunk_size
        chunk = clean_text[start:end]
        if len(chunk.strip()) > 24:
            chunks.append(chunk.strip())
        start += (chunk_size - overlap)
    return chunks

def extract_file_text(file_bytes: bytes, filename: str) -> str:
    """Extracts raw text from PDF, DOCX, or TXT documents"""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif name.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return file_bytes.decode("utf-8", errors="ignore")