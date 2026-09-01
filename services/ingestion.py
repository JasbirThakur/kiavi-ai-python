import io
import re
from typing import List
from pypdf import PdfReader

try:
    from PIL import Image
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    extracted_text = []
    
    # 1. Native Digital PDF Extraction
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            
            # 2. OCR Fallback for Scanned / Image-only Pages
            if len(page_text.strip()) < 40 and OCR_AVAILABLE:
                try:
                    images = convert_from_bytes(
                        file_bytes,
                        first_page=page_idx + 1,
                        last_page=page_idx + 1
                    )
                    if images:
                        ocr_result = pytesseract.image_to_string(images[0])
                        page_text = ocr_result
                except Exception as ocr_err:
                    print(f"[OCR Warning] Page {page_idx + 1} OCR failed: {ocr_err}")
            
            if page_text.strip():
                extracted_text.append(page_text.strip())
    except Exception as e:
        print(f"[PDF Parse Error] Direct extraction failed: {e}")
        # Total OCR Fallback
        if OCR_AVAILABLE:
            try:
                images = convert_from_bytes(file_bytes)
                for img in images:
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        extracted_text.append(ocr_text.strip())
            except Exception as ocr_all_err:
                print(f"[OCR Critical] Total PDF OCR failed: {ocr_all_err}")

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
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return extract_text_from_pdf(file_bytes)


# routers/knowledge.py compatibility alias
get_chunks_from_text = chunk_text