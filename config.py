"""
config.py — Configuration & Setup
================================
All paths, tokens, and directory setup.
"""

import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

OUTPUT_DIR   = str(PROJECT_ROOT / "output")
RAG_DIR      = str(PROJECT_ROOT / "rag_index")
CHAT_LOG_DIR = str(PROJECT_ROOT / "chat_logs")
MODEL_DIR    = str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")
TEMP_DIR     = str(PROJECT_ROOT / "temp")


# ──────────────────────────────────────────────────────────────────────────
# Setup Directories
# ──────────────────────────────────────────────────────────────────────────
def setup_output_dirs():
    """Create all required directories safely."""
    dirs = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, "rules"),
        os.path.join(OUTPUT_DIR, "definitions"),
        RAG_DIR,
        CHAT_LOG_DIR,
        MODEL_DIR,
        TEMP_DIR,
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Load Token
# ──────────────────────────────────────────────────────────────────────────
def load_token() -> str:
    """Load HF_TOKEN from .env or environment variables."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    token = os.environ.get("HF_TOKEN", "").strip()

    if token:
        print("[✅] HF_TOKEN loaded")
    else:
        print("[⚠️] HF_TOKEN not found")

    return token


# ──────────────────────────────────────────────────────────────────────────
# File Picker / CLI Input
# ──────────────────────────────────────────────────────────────────────────
def pick_files() -> list:
    """Get file paths from CLI OR open file picker (local only)."""

    # ── CLI mode (Production) ─────────────────────────
    if len(sys.argv) > 1:
        paths = [os.path.normpath(p) for p in sys.argv[1:]]

        print(f"[✅] {len(paths)} file(s) received:")
        for p in paths:
            print(f"   • {p}")

        return paths

    # ── Local testing (GUI) ───────────────────────────
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()

        files = filedialog.askopenfilenames(
            title="Select files",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("Images", "*.png *.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )

        if not files:
            print("[⚠️] No files selected")
            sys.exit(0)

        files = [os.path.normpath(f) for f in files]

        print(f"[📂] {len(files)} file(s) selected:")
        for f in files:
            print(f"   • {f}")

        return list(files)

    except Exception as e:
        print(f"[❌] File picker failed: {e}")
        sys.exit(0)


# ──────────────────────────────────────────────────────────────────────────
# Export (Rules / Definitions)
# ──────────────────────────────────────────────────────────────────────────
def export_to_folder(category: str, title: str, content: str):
    """
    Save extracted content into structured files.

    - One file per entry
    - Safe filenames
    - No overwrite
    """

    if not content or not content.strip():
        return  # 🔥 ignore empty content

    folder_path = os.path.join(OUTPUT_DIR, category)
    os.makedirs(folder_path, exist_ok=True)

    # ── Clean title ─────────────────────────
    safe_title = (
        title.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    if not safe_title:
        safe_title = "unknown"

    # ── Prevent overwrite ───────────────────
    unique_id = uuid.uuid4().hex[:6]
    filename = f"{safe_title}_{unique_id}.txt"

    filepath = os.path.join(folder_path, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())

        print(f"[✅] Saved → {category}/{filename}")

    except Exception as e:
        print(f"[❌] Failed to save {category}: {e}")