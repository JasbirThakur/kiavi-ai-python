import io
import re
import base64
import hashlib
from pathlib import Path
from typing import List
from pypdf import PdfReader

DIAG_DIR = Path(__file__).resolve().parent.parent / "static" / "extracted_diagrams"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

def ocr_image_with_vision(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Extracts text, numbers, diagrams, and tables via NVIDIA NIM Vision LLM or local Tesseract OCR fallback."""
    # 1. Try NVIDIA Multimodal Vision if key is valid
    try:
        from openai import OpenAI
        from config import NVIDIA_API_KEY, NVIDIA_BASE_URL
        if NVIDIA_API_KEY and not NVIDIA_API_KEY.startswith("your_"):
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=12.0)
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
            ans = resp.choices[0].message.content.strip()
            if ans:
                return ans
    except Exception as e:
        print(f"[Vision OCR API Fallback]: {e}")

    # 2. Local Tesseract OCR Fallback (Zero API Key, 100% offline & reliable)
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        ocr_result = pytesseract.image_to_string(img).strip()
        if ocr_result:
            return ocr_result
    except Exception as tess_err:
        print(f"[Local Tesseract OCR Warning]: {tess_err}")

    return ""

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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

        # 3. Scanned PDF Fallback (if direct text was completely empty)
        if not extracted_text:
            try:
                from pdf2image import convert_from_bytes
                import pytesseract
                images = convert_from_bytes(file_bytes, first_page=1, last_page=5)
                for p_idx, p_img in enumerate(images):
                    p_txt = pytesseract.image_to_string(p_img).strip()
                    if p_txt:
                        extracted_text.append(f"[Scanned PDF Page {p_idx+1}]:\n{p_txt}")
            except Exception as scan_err:
                print(f"[Scanned PDF OCR Notice]: {scan_err}")

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
    """Extracts text from PDF, DOCX, CSV, images (PNG, JPG, WEBP), or raw text files."""
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
            header = [h.strip() for h in rows[0]]
            formatted_lines = [f"=== CSV Table: {filename} ===", f"Columns: {', '.join(header)}"]
            for idx, row in enumerate(rows[1:], 1):
                row_items = []
                for col_idx, val in enumerate(row):
                    col_name = header[col_idx] if col_idx < len(header) else f"Col_{col_idx+1}"
                    val_str = val.strip()
                    if val_str:
                        row_items.append(f"{col_name}: {val_str}")
                if row_items:
                    formatted_lines.append(f"[Row {idx}]: " + " | ".join(row_items))
            return "\n".join(formatted_lines)
        except Exception as e:
            print(f"[CSV Parse Error]: {e}")
            return file_bytes.decode("utf-8", errors="ignore")
    elif name.endswith(".zip"):
        try:
            import zipfile
            extracted_sections = []
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                for zip_entry in zf.infolist():
                    if zip_entry.is_dir() or zip_entry.filename.startswith("__MACOSX/") or "/." in zip_entry.filename or zip_entry.filename.startswith("."):
                        continue
                    entry_name = Path(zip_entry.filename).name
                    # Skip common non-document binary formats
                    if entry_name.lower().endswith((".exe", ".bin", ".pyc", ".class", ".dll", ".so", ".dylib")):
                        continue
                    entry_bytes = zf.read(zip_entry.filename)
                    entry_text = extract_file_text(entry_bytes, entry_name)
                    if entry_text and entry_text.strip():
                        extracted_sections.append(f"=== Document from Archive ({zip_entry.filename}) ===\n{entry_text.strip()}")
            if extracted_sections:
                return "\n\n".join(extracted_sections)
            return ""
        except Exception as e:
            print(f"[ZIP Archive Extraction Error]: {e}")
            return ""
    elif name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        mime = "image/png"
        if name.endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        elif name.endswith(".webp"):
            mime = "image/webp"

        # Save uploaded image/diagram so RAG can render it in chat
        img_hash = hashlib.md5(file_bytes).hexdigest()[:12]
        ext = name.split(".")[-1]
        img_filename = f"upload_{img_hash}.{ext}"
        img_path = DIAG_DIR / img_filename
        if not img_path.exists():
            img_path.write_bytes(file_bytes)

        img_url = f"/static/extracted_diagrams/{img_filename}"
        ocr_text = ocr_image_with_vision(file_bytes, mime)
        caption = filename or "Diagram / Technical Visual"

        if ocr_text:
            return f"![{caption}]({img_url})\n\n[Visual Image / Diagram Data ({caption})]:\n{ocr_text}"
        else:
            return f"![{caption}]({img_url})\n\n[Visual Diagram / Photo]: {caption}"

    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return extract_text_from_pdf(file_bytes)

# routers/knowledge.py compatibility alias
get_chunks_from_text = chunk_text
