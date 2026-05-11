"""
image.py — Graph & Image Analysis Module
==========================================
Extracts images from PDFs, then analyzes them using GPT-4o-mini
via langchain_chain.py
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

PROJECT_ROOT       = Path(__file__).parent.resolve()
GRAPHS_DIR         = str(PROJECT_ROOT / "output" / "graphs")
GRAPH_ANALYSIS_DIR = str(PROJECT_ROOT / "output" / "graph_analysis")

for _d in [GRAPHS_DIR, GRAPH_ANALYSIS_DIR]:
    os.makedirs(_d, exist_ok=True)

print("[OK] Image folders ready:")
print(f"     input    -> {GRAPHS_DIR}")
print(f"     analyzed -> {GRAPH_ANALYSIS_DIR}")


# ══════════════════════════════════════════════════════════════════════════
# استخراج الصور من PDF
# ══════════════════════════════════════════════════════════════════════════

def extract_graphs_from_pdf(pdf_path: str) -> list:
    """استخراج كل الصور من PDF وحفظها في GRAPHS_DIR"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[WARN] Install pymupdf: pip install pymupdf")
        return []

    extracted = []
    print(f"[PDF] Extracting images from: {os.path.basename(pdf_path)}")

    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref       = img[0]
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                img_ext    = base_image["ext"]

                # تجاهل الصور الصغيرة جداً (أيقونات)
                if len(img_bytes) < 5000:
                    continue

                filename  = f"page{page_num+1}_img{img_index+1}.{img_ext}"
                save_path = os.path.join(GRAPHS_DIR, filename)

                with open(save_path, "wb") as f:
                    f.write(img_bytes)

                extracted.append(save_path)
                print(f"    Saved: {filename}")

        doc.close()
        print(f"[OK] Extracted {len(extracted)} images.")

    except Exception as e:
        print(f"[ERR] PDF extraction error: {e}")

    return extracted


# ══════════════════════════════════════════════════════════════════════════
# تحليل صورة واحدة بـ GPT-4o-mini عبر langchain_chain
# ══════════════════════════════════════════════════════════════════════════

def analyze_image_with_gpt(image_path: str) -> str:
    """بتبعت الصورة لـ GPT-4o-mini عبر langchain_chain وبترجع التحليل"""
    try:
        from langchain_chain import describe_image
    except ImportError:
        return "Error: langchain_chain.py not found."

    ext  = Path(image_path).suffix.lower()
    mime = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")

    print(f"[>>] Analyzing with GPT-4o-mini: {os.path.basename(image_path)}")

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # بنغير الـ prompt عشان يكون أنسب للجرافات العلمية
        from langchain_chain import _build_llm
        from langchain_core.messages import HumanMessage
        from langchain_core.output_parsers import StrOutputParser
        import base64

        b64 = base64.b64encode(image_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "You are an expert in scientific graphs and diagrams. "
                        "Analyze this image and provide:\n"
                        "1. Type of graph/diagram\n"
                        "2. Axes and units (if applicable)\n"
                        "3. Main trend or finding\n"
                        "4. Important values or equations shown\n"
                        "5. What a student should learn from this\n"
                        "If this is not a graph/diagram, describe what you see."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                        "detail": "auto",
                    },
                },
            ]
        )

        llm    = _build_llm()
        parser = StrOutputParser()
        chain  = llm | parser
        result = chain.invoke([message])

        print(f"[OK] GPT-4o-mini analysis done ({len(result)} chars).")
        return result

    except EnvironmentError as e:
        return f"Error: {e}"
    except Exception as e:
        print(f"[ERR] GPT analysis failed: {e}")
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════
# الدالة الرئيسية — استخراج + تحليل
# ══════════════════════════════════════════════════════════════════════════

def extract_and_analyze_graphs(
    file_path: str,
    hf_token: Optional[str] = None,  # kept for backwards compatibility
) -> list:
    """
    الدالة الرئيسية:
    - بتاخد PDF أو صورة
    - بتستخرج الصور (لو PDF)
    - بتحللهم بـ GPT-4o-mini
    - بتحفظ التحليل في output/graph_analysis/
    - بترجع list من dicts
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        image_paths = extract_graphs_from_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
        image_paths = [file_path]
    else:
        print(f"[WARN] Unsupported file type: {ext}")
        return []

    if not image_paths:
        print("[WARN] No images found to analyze.")
        return []

    print(f"\n[>>] Analyzing {len(image_paths)} image(s) with GPT-4o-mini...")
    results = []

    for img_path in image_paths:
        analysis    = analyze_image_with_gpt(img_path)
        output_file = os.path.join(
            GRAPH_ANALYSIS_DIR,
            f"{Path(img_path).stem}_analysis.txt"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Image: {img_path}\n{'='*60}\n{analysis}\n")

        print(f"[SAVE] {output_file}")
        results.append({
            "image_path": img_path,
            "analysis":   analysis,
            "saved_to":   output_file,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════
# تشغيل مباشر للتجربة
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python image.py <path_to_pdf_or_image>")
        sys.exit(1)

    results = extract_and_analyze_graphs(sys.argv[1])
    for r in results:
        print(f"\n{'='*60}\n{r['image_path']}\n{'='*60}\n{r['analysis']}")
