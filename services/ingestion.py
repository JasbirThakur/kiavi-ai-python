import io
import re
import base64
from typing import List
from pypdf import PdfReader

def ocr_image_with_vision(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Uses NVIDIA NIM Multimodal Vision LLM to perform high-precision OCR on images, charts, and diagrams."""
    try:
        from openai import OpenAI
        from config import NVIDIA_API_KEY, NVIDIA_BASE_URL
        if not NVIDIA_API_KEY:
            return ""

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
        resp = client.chat.completions.create(
            model="meta/llama-3.2-11b-vision-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Perform high-precision Optical Character Recognition (OCR). Transcribe all text, numbers, labels, tables, headings, and data from this image or diagram verbatim. Do not summarize; extract the exact content."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"}
                        }
                    ]
                }
            ],
            max_tokens=600,
            temperature=0.1
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Vision OCR Warning]: {e}")
        return ""

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


import hashlib
from pathlib import Path

DIAG_DIR = Path(__file__).resolve().parent.parent / "static" / "extracted_diagrams"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    extracted_text = []
    
    # 1. Native Digital PDF Extraction + Embedded Diagram OCR & Image Persist
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            
            # 2. Check for embedded diagrams, figures, or charts on this page
            if hasattr(page, "images") and page.images:
                for img_idx, img_obj in enumerate(page.images):
                    try:
                        if img_idx >= 3:
                            break
                        img_bytes = getattr(img_obj, "data", None)
                        if img_bytes and len(img_bytes) > 2048:
                            # Save diagram image to static folder
                            img_hash = hashlib.md5(img_bytes).hexdigest()[:12]
                            diag_filename = f"diag_p{page_idx + 1}_{img_idx + 1}_{img_hash}.png"
                            diag_path = DIAG_DIR / diag_filename
                            if not diag_path.exists():
                                diag_path.write_bytes(img_bytes)

                            diag_text = ocr_image_with_vision(img_bytes)
                            img_url = f"/static/extracted_diagrams/{diag_filename}"
                            caption = f"Figure {page_idx + 1}.{img_idx + 1}"
                            if diag_text and len(diag_text) > 15:
                                page_text += f"\n\n![{caption}]({img_url})\n[Diagram / Visual Plate {page_idx + 1}.{img_idx + 1} Data]:\n{diag_text}"
                            else:
                                page_text += f"\n\n![{caption}]({img_url})"
                    except Exception as diag_err:
                        print(f"[Diagram OCR Warning]: {diag_err}")
            
            if page_text.strip():
                extracted_text.append(page_text.strip())
    except Exception as e:
        print(f"[PDF Parse Error] Direct extraction failed: {e}")

    return "\n\n".join(extracted_text)


def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 50) -> List[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    
    words = cleaned.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += (chunk_size - chunk_overlap)
        
    return chunks


def extract_file_text(file_bytes: bytes, filename: str = "") -> str:
    """Extracts text from PDF, DOCX, or raw text files."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join([p.text for p in doc.paragraphs if p.text])
        except Exception as e:
            print(f"[DOCX Parse Error]: {e}")
    elif name.endswith(".csv"):
        try:
            import csv
            content_str = file_bytes.decode("utf-8", errors="ignore")
            reader = csv.reader(io.StringIO(content_str))
            rows = list(reader)
            if not rows:
                return ""
            header = rows[0]
            formatted_lines = [f"CSV Table: {filename}"]
            for idx, row in enumerate(rows[1:], 1):
                row_items = []
                for col_idx, val in enumerate(row):
                    col_name = header[col_idx] if col_idx < len(header) else f"Col_{col_idx+1}"
                    val_str = val.strip()
                    if val_str:
                        row_items.append(f"{col_name}: {val_str}")
                if row_items:
                    formatted_lines.append(f"Record {idx}: " + ", ".join(row_items))
            return "\n".join(formatted_lines)
        except Exception as e:
            print(f"[CSV Parse Error]: {e}")
            return file_bytes.decode("utf-8", errors="ignore")
    elif name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        mime = "image/png"
        if name.endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        elif name.endswith(".webp"):
            mime = "image/webp"
        return ocr_image_with_vision(file_bytes, mime)
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return extract_text_from_pdf(file_bytes)


# routers/knowledge.py compatibility alias
get_chunks_from_text = chunk_text