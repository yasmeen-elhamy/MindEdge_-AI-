"""
ocr.py — OCR Pipeline
======================
Text extraction from PDFs and images using pytesseract.
"""

import os
import logging
import concurrent.futures
from PIL import Image
import platform
try:
    import google.generativeai as genai  # type: ignore[import]
except ImportError:
    genai = None
try:
    import pytesseract  # type: ignore[import]
except ImportError:
    pytesseract = None
try:
    from pdf2image import convert_from_path  # type: ignore[import]
except ImportError:
    convert_from_path = None

if platform.system() == "Windows":
    _TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    _TESSERACT_PATH = "/usr/bin/tesseract"

if pytesseract is not None:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH

def setup_tesseract():
    if pytesseract is None:
        raise RuntimeError("[❌] pytesseract is not installed.")

    if not os.path.exists(_TESSERACT_PATH):
        raise RuntimeError(f"[❌] Tesseract not found at: {_TESSERACT_PATH}")

    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH

    try:
        version = pytesseract.get_tesseract_version()
        print(f"[✅] Tesseract ready | version: {version}")
    except Exception as e:
        raise RuntimeError(f"[❌] Tesseract not working: {e}")

setup_tesseract()

OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_tesseract():
    if pytesseract is None:
        raise RuntimeError("[❌] pytesseract is not installed.")

    if not os.path.exists(_TESSERACT_PATH):
        raise RuntimeError(f"[❌] Tesseract not found at: {_TESSERACT_PATH}")

    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH

    try:
        version = pytesseract.get_tesseract_version()
        print(f"[✅] Tesseract ready | version: {version}")
    except Exception as e:
        raise RuntimeError(f"[❌] Tesseract not working: {e}")

def run_ocr(path: str) -> str | None:
    """Runs OCR on an image or PDF. Returns raw text string or None."""
    
    # ✅ FIX 1: normalize Windows path
    path = os.path.normpath(path)

    print(f"[🔍] Running OCR on {path}…")

    if pytesseract is None:
        print("[❌] pytesseract is required for OCR but is not installed.")
        return None

    if not os.path.exists(path):
        print(f"[❌] File not found: {path}")
        return None

    try:
        ext = path.lower()

        if ext.endswith(("jpg", "jpeg", "png", "bmp", "webp")):
            text = pytesseract.image_to_string(Image.open(path))

        else:
            # ✅ FIX 2: check pdf2image
            if convert_from_path is None:
                print("[❌] pdf2image is not installed.")
                return None

            pages = convert_from_path(path, dpi=200)
            if not pages:
                print("[❌] PDF conversion returned no pages.")
                return None
            parts = []
            for i, page_img in enumerate(pages):
                print(f"     OCR page {i+1}/{len(pages)}…")
                parts.append(pytesseract.image_to_string(page_img))

            text = "\n".join(parts)

        print("     OCR completed.")
        return text

    except Exception as exc:
        print(f"[❌] OCR error: {exc}")
        return None


def extract_text(result: str | None) -> str:
    """Cleans and previews extracted text."""
    if not result:
        return ""
    text    = result.strip()
    preview = text[:120].replace("\n", " ")
    print(f"[📝] Extracted {len(text)} chars | preview: {preview}…")
    return text


def correct_text(text: str) -> str:
    """Corrects the extracted text using AI."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f"Correct the following text for grammar, clarity, and accuracy: {text}")
        return response.text
    except Exception as exc:
        return f"Error: {exc}"


def process_file(path: str) -> tuple:
    """OCR + correction for a single file. Returns (path, corrected, raw)."""
    logging.info(f"Processing: {path}")

    raw_text = extract_text(run_ocr(path))
    if not raw_text:
        logging.error(f"No text extracted from {path}")
        return None, None, None

    corrected = correct_text(raw_text)
    if corrected.startswith("Error:"):
        logging.warning(f"Correction failed for {path} — using raw text.")
        corrected = raw_text

    stem = os.path.basename(path).replace(" ", "_")
    with open(os.path.join(OUTPUT_DIR, f"{stem}_raw.md"),       "w", encoding="utf-8") as f:
        f.write(raw_text)
    with open(os.path.join(OUTPUT_DIR, f"{stem}_corrected.md"), "w", encoding="utf-8") as f:
        f.write(corrected)

    logging.info(f"Done: {path}")
    return path, corrected, raw_text


def run_ocr_pipeline(file_paths: list) -> tuple[dict, str]:
    """
    Runs OCR on all files in parallel.
    Returns (all_corrected_texts dict, last_raw_text string).
    """
    print("\n" + "═" * 60)
    print("📸  STAGE 1 — OCR PIPELINE")
    print("═" * 60)

    all_corrected: dict = {}
    last_raw: str = ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_file, p) for p in file_paths]
        for future in concurrent.futures.as_completed(futures):
            path, corrected, raw = future.result()
            if path is not None:
                all_corrected[path] = corrected
                last_raw = raw

    if not all_corrected:
        print("[❌] No text extracted from any file.")
    else:
        print(f"\n[✅] OCR done. {len(all_corrected)} file(s) processed.")

    return all_corrected, last_raw