import io
import re
import base64
import hashlib
from pathlib import Path
from typing import List, Optional
from pypdf import PdfReader

DIAG_DIR = Path(__file__).resolve().parent.parent / "static" / "extracted_diagrams"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

def ocr_image_with_vision(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Extracts text, numbers, diagrams, and tables via NVIDIA NIM Vision LLM or local Tesseract OCR fallback."""
    # 1. Try NVIDIA Multimodal Vision if key is valid
    try:
        from openai import OpenAI
        from app.config.settings import NVIDIA_API_KEY, NVIDIA_BASE_URL
        if NVIDIA_API_KEY and not NVIDIA_API_KEY.startswith("your_"):
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=4.0)
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

    # 1. Native Digital PDF Extraction
    try:
        reader = PdfReader(io.BytesIO(file_bytes), strict=False)
        for page_idx, page in enumerate(reader.pages):
            page_components = []
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_components.append(page_text.strip())
            except Exception as pe:
                print(f"[PDF Page {page_idx+1} Extract Error]: {pe}")

            has_direct_text = bool(page_text and len(page_text.strip()) > 40)

            # Extract embedded images and diagrams from this page (Fast & non-blocking)
            try:
                if hasattr(page, 'images') and page.images:
                    figures_saved = 0
                    for img_idx, img_file in enumerate(page.images):
                        if figures_saved >= 3:
                            break
                        raw_bytes = img_file.data
                        # Diagram/Schematic filter: must be substantive (>15KB)
                        if len(raw_bytes) > 15360:
                            img_hash = hashlib.md5(raw_bytes).hexdigest()[:12]
                            ext = img_file.name.split(".")[-1].lower() if "." in img_file.name else "png"
                            if ext not in ["png", "jpg", "jpeg", "webp"]:
                                ext = "png"
                            img_filename = f"diag_p{page_idx+1}_{img_idx+1}_{img_hash}.{ext}"
                            img_path = DIAG_DIR / img_filename
                            if not img_path.exists():
                                img_path.write_bytes(raw_bytes)
                            img_url = f"/static/extracted_diagrams/{img_filename}"
                            caption = f"Figure {page_idx+1}.{img_idx+1}"

                            # If page already has rich digital text, directly attach the visual markdown
                            # Only call OCR if page has NO text at all (pure image page)
                            if not has_direct_text:
                                ocr_text = ocr_image_with_vision(raw_bytes, f"image/{ext}")
                                diag_block = (
                                    f"![{caption}]({img_url})\n"
                                    f"[Diagram / Visual Plate {page_idx+1}.{img_idx+1} Data]:\n{ocr_text}"
                                    if ocr_text else f"![{caption}]({img_url})"
                                )
                            else:
                                diag_block = f"![{caption}]({img_url})"

                            page_components.append(diag_block)
                            figures_saved += 1
            except Exception:
                pass

            if page_components:
                extracted_text.append("\n\n".join(page_components))

        # 2. Scanned PDF Fallback (if direct text was completely empty or extremely short)
        if sum(len(t) for t in extracted_text) < 50:
            try:
                from pdf2image import convert_from_bytes
                import pytesseract
                # Convert first 15 pages max for scanned documents
                images = convert_from_bytes(file_bytes, first_page=1, last_page=15)
                for p_idx, p_img in enumerate(images):
                    p_txt = pytesseract.image_to_string(p_img).strip()
                    if p_txt:
                        extracted_text.append(f"[Scanned PDF Page {p_idx+1}]:\n{p_txt}")
            except Exception as scan_err:
                print(f"[Scanned PDF OCR Notice]: {scan_err}")

    except Exception as e:
        print(f"[PDF Parse Error] Direct extraction failed: {e}")
        # Try scanned fallback directly if PdfReader crashed
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            images = convert_from_bytes(file_bytes, first_page=1, last_page=15)
            for p_idx, p_img in enumerate(images):
                p_txt = pytesseract.image_to_string(p_img).strip()
                if p_txt:
                    extracted_text.append(f"[Scanned PDF Page {p_idx+1}]:\n{p_txt}")
        except Exception as scan_err:
            print(f"[Scanned PDF Emergency OCR Notice]: {scan_err}")

    return "\n\n".join(extracted_text)

from app.services.chunking import (
    split_into_sentences,
    semantic_chunking,
    chunk_text,
    estimate_tokens
)

def decode_text_bytes(file_bytes: bytes) -> str:
    """Decodes bytes to string supporting UTF-8 BOM, UTF-16 LE/BE BOM, UTF-8, CP1252, and Latin-1."""
    if file_bytes.startswith(b'\xef\xbb\xbf'):
        return file_bytes[3:].decode('utf-8', errors='replace')
    if file_bytes.startswith(b'\xff\xfe'):
        return file_bytes[2:].decode('utf-16le', errors='replace')
    if file_bytes.startswith(b'\xfe\xff'):
        return file_bytes[2:].decode('utf-16be', errors='replace')
    for enc in ('utf-8', 'cp1252', 'latin-1'):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode('utf-8', errors='replace')

def extract_docx_paragraphs(file_bytes: bytes) -> str:
    """Native pure-Python standard-library DOCX parser using zipfile and xml.etree.ElementTree."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                r_txt = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                if r_txt:
                    texts.append(r_txt)
        if texts:
            return "\n\n".join(texts)
    except Exception:
        pass

    try:
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            if "word/document.xml" in z.namelist():
                xml_data = z.read("word/document.xml")
                tree = ET.fromstring(xml_data)
                paragraphs = []
                for p in tree.iter():
                    if p.tag.endswith("}p"):
                        texts = [t.text for t in p.iter() if t.tag.endswith("}t") and t.text]
                        if texts:
                            p_str = "".join(texts).strip()
                            if p_str:
                                paragraphs.append(p_str)
                if paragraphs:
                    return "\n\n".join(paragraphs)
    except Exception as e:
        print(f"[Native DOCX Parse Error]: {e}")
    return ""

def extract_xlsx_tables(file_bytes: bytes, filename: str = "") -> str:
    """Native pure-Python standard-library XLSX parser using zipfile and xml.etree.ElementTree."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
        sheets_out = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                non_empty = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if non_empty:
                    rows.append([str(c).strip() if c is not None else "" for c in row])
            if rows:
                header = rows[0]
                lines = [f"=== Excel Sheet: {sheet_name} ===", f"Columns: {', '.join(header)}"]
                for r_idx, r in enumerate(rows[1:1000], 1):
                    items = [f"{header[c_idx] if c_idx < len(header) else f'Col_{c_idx+1}'}: {cell}" for c_idx, cell in enumerate(r) if cell]
                    if items:
                        lines.append(f"[Row {r_idx}]: " + " | ".join(items))
                sheets_out.append("\n".join(lines))
        if sheets_out:
            return "\n\n".join(sheets_out)
    except Exception:
        pass

    try:
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            shared_strings = []
            if "xl/sharedStrings.xml" in z.namelist():
                ss_tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
                for si in ss_tree.iter():
                    if si.tag.endswith("}si"):
                        texts = [t.text for t in si.iter() if t.tag.endswith("}t") and t.text]
                        shared_strings.append("".join(texts))

            sheet_files = [n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
            sheets_out = []
            for sf in sheet_files:
                s_tree = ET.fromstring(z.read(sf))
                sheet_rows = []
                for row in s_tree.iter():
                    if row.tag.endswith("}row"):
                        r_vals = []
                        for c in row.iter():
                            if c.tag.endswith("}c"):
                                c_type = c.attrib.get("t")
                                v_el = c.find("{*}v")
                                val = v_el.text if (v_el is not None and v_el.text) else ""
                                if c_type == "s" and val.isdigit() and int(val) < len(shared_strings):
                                    val = shared_strings[int(val)]
                                elif c_type == "inlineStr":
                                    t_el = c.find(".//{*}t")
                                    if t_el is not None and t_el.text:
                                        val = t_el.text
                                r_vals.append(val.strip() if val else "")
                        if any(r_vals):
                            sheet_rows.append(r_vals)
                if sheet_rows:
                    header = sheet_rows[0]
                    lines = [f"=== Excel Sheet: {sf.split('/')[-1]} ===", f"Columns: {', '.join(header)}"]
                    for r_idx, r in enumerate(sheet_rows[1:1000], 1):
                        items = [f"{header[c_idx] if c_idx < len(header) else f'Col_{c_idx+1}'}: {cell}" for c_idx, cell in enumerate(r) if cell]
                        if items:
                            lines.append(f"[Row {r_idx}]: " + " | ".join(items))
                    sheets_out.append("\n".join(lines))
            if sheets_out:
                return "\n\n".join(sheets_out)
    except Exception as e:
        print(f"[Native XLSX Parse Error]: {e}")
    return ""

def extract_file_text(file_bytes: bytes, filename: str = "") -> str:
    """Extracts text from PDF, DOCX, XLSX, CSV, images (PNG, JPG, WEBP), ZIP/VSIX packages, or raw text files."""
    name = filename.lower()

    # 1. PDF Detection (by extension or magic byte header)
    if name.endswith(".pdf") or file_bytes.startswith(b"%PDF-"):
        return extract_text_from_pdf(file_bytes)

    # 2. DOCX Detection
    if name.endswith(".docx") or (file_bytes.startswith(b"PK\x03\x04") and name.endswith(".docx")):
        res = extract_docx_paragraphs(file_bytes)
        if res:
            return res

    # 3. Excel Spreadsheet (.xlsx, .xlsm, .xls)
    if name.endswith((".xlsx", ".xlsm", ".xls")) or (file_bytes.startswith(b"PK\x03\x04") and name.endswith((".xlsx", ".xlsm"))):
        res = extract_xlsx_tables(file_bytes, filename)
        if res:
            return res

    # 4. CSV / TSV Tabular Detection
    if name.endswith((".csv", ".tsv")):
        try:
            import csv
            content_str = decode_text_bytes(file_bytes)
            sample = content_str[:4096]
            delim = ','
            if '\t' in sample and sample.count('\t') > sample.count(','):
                delim = '\t'
            elif ';' in sample and sample.count(';') > sample.count(','):
                delim = ';'
            elif '|' in sample and sample.count('|') > sample.count(','):
                delim = '|'

            reader = csv.reader(io.StringIO(content_str), delimiter=delim)
            rows = []
            for r in reader:
                if r and any(c.strip() for c in r):
                    rows.append(r)
                if len(rows) >= 3000:
                    break

            if rows:
                raw_header = [h.strip() for h in rows[0]]
                skip_or_truncate_cols = set()
                for c_idx, h in enumerate(raw_header):
                    h_l = h.lower()
                    if any(k in h_l for k in ['imageurl', 'image_url', 'sourceurl', 'source_url', 'keys', 'asins']):
                        skip_or_truncate_cols.add(c_idx)

                formatted_lines = [f"=== CSV Table: {filename} ===", f"Columns: {', '.join(raw_header)}"]
                for idx, row in enumerate(rows[1:], 1):
                    row_items = []
                    for col_idx, val in enumerate(row):
                        col_name = raw_header[col_idx] if col_idx < len(raw_header) else f"Col_{col_idx+1}"
                        val_str = val.strip()
                        if not val_str:
                            continue
                        if col_idx in skip_or_truncate_cols:
                            if ',' in val_str or ';' in val_str:
                                first_u = re.split(r'[,;]', val_str)[0].strip()
                                val_str = first_u[:120]
                            else:
                                val_str = val_str[:120]
                        else:
                            val_str = val_str[:250]
                        if val_str:
                            row_items.append(f"{col_name}: {val_str}")
                    if row_items:
                        formatted_lines.append(f"[Row {idx}]: " + " | ".join(row_items))
                return "\n".join(formatted_lines)
        except Exception as e:
            print(f"[CSV Parse Error]: {e}")
            return decode_text_bytes(file_bytes)

    # 5. ZIP & Package Archive Detection (.zip, .vsix, .vsixpackage, .nupkg, .jar, .war, .apk, etc. or PK magic bytes)
    is_archive = name.endswith((".zip", ".vsix", ".vsixpackage", ".nupkg", ".jar", ".war", ".apk"))
    if not is_archive and file_bytes.startswith(b"PK\x03\x04") and not name.endswith((".docx", ".xlsx", ".xlsm")):
        try:
            import zipfile
            is_archive = zipfile.is_zipfile(io.BytesIO(file_bytes))
        except Exception:
            is_archive = False

    if is_archive:
        try:
            import zipfile
            extracted_sections = []
            readable_exts = (
                ".txt", ".md", ".markdown", ".json", ".xml", ".vsixmanifest",
                ".csv", ".tsv", ".yaml", ".yml", ".py", ".js", ".ts", ".html",
                ".htm", ".css", ".sql", ".sh", ".env", ".properties", ".ini",
                ".cfg", ".conf", ".rst", ".toml"
            )
            total_chars = 0
            MAX_ARCHIVE_CHARS = 500000

            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                entries = zf.infolist()
                def entry_priority(e):
                    fn = e.filename.lower()
                    if "manifest" in fn: return 0
                    if "package.json" in fn: return 1
                    if "readme" in fn: return 2
                    if fn.endswith((".md", ".txt")): return 3
                    return 10

                sorted_entries = sorted(entries, key=entry_priority)

                for zip_entry in sorted_entries:
                    if total_chars >= MAX_ARCHIVE_CHARS:
                        extracted_sections.append("[Note: Archive content truncated after reaching 500,000 characters limit.]")
                        break

                    if zip_entry.is_dir() or zip_entry.filename.startswith("__MACOSX/") or "/." in zip_entry.filename or zip_entry.filename.startswith("."):
                        continue

                    entry_lower = zip_entry.filename.lower()
                    if entry_lower.endswith((".exe", ".bin", ".pyc", ".class", ".dll", ".so", ".dylib", ".ico", ".png", ".jpg", ".jpeg", ".gif", ".woff", ".woff2", ".ttf", ".eot", ".node")):
                        continue

                    if entry_lower.endswith(readable_exts) or "manifest" in entry_lower:
                        try:
                            entry_data = zf.read(zip_entry.filename)
                            if len(entry_data) > 200000:
                                entry_data = entry_data[:200000]
                            entry_text = entry_data.decode("utf-8", errors="ignore").strip()
                            if entry_text and len(entry_text) > 10:
                                extracted_sections.append(f"=== Archive File: {zip_entry.filename} ===\n{entry_text}")
                                total_chars += len(entry_text)
                        except Exception as e_err:
                            print(f"[Archive entry read error {zip_entry.filename}]: {e_err}")

            if extracted_sections:
                return "\n\n".join(extracted_sections)
            return ""
        except Exception as e:
            print(f"[ZIP Archive Extraction Error]: {e}")
            return ""

    # 5. Image OCR (PNG, JPG, WEBP)
    if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        mime = "image/png"
        if name.endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        elif name.endswith(".webp"):
            mime = "image/webp"

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

    # 6. Direct UTF-8 and Latin-1 Text Decoders (for code files, logs, json, xml, txt, etc.)
    try:
        text = file_bytes.decode("utf-8")
        if text.strip():
            return text
    except UnicodeDecodeError:
        pass

    try:
        text = file_bytes.decode("latin-1", errors="ignore")
        printable_ratio = sum(c.isprintable() for c in text[:1000]) / max(len(text[:1000]), 1)
        if printable_ratio > 0.85 and text.strip():
            return text
    except Exception:
        pass

    return ""

# routers/knowledge.py compatibility alias
get_chunks_from_text = chunk_text
split_into_semantic_chunks = semantic_chunking
