"""
api_service.py — FastAPI Backend  (TTS-enhanced)
==================================
Run:  uvicorn api_service:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from topic_extractor import extract_topics_from_text
from study_plan import generate_study_plan

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Local modules ──────────────────────────────────────────────────────────
from config    import setup_output_dirs, OUTPUT_DIR, CHAT_LOG_DIR
from llm       import correct_text, summarize_text, dev_ask_llm
from ocr       import setup_tesseract, run_ocr, extract_text
from rag       import build_or_load_index, retrieve_passages, auto_scan_text, save_chat_log
from image     import extract_and_analyze_graphs
from tts_utils import safe_generate_tts
from quiz_utils import generate_quiz_from_context, grade_quiz

# ── Bootstrap ──────────────────────────────────────────────────────────────
setup_output_dirs()
setup_tesseract()

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not set — add it to your .env file")

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="MindEdge API",
    description="Intelligent Study Scanner & Tutor",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve audio files
os.makedirs("audio_cache", exist_ok=True)
app.mount("/audio", StaticFiles(directory="audio_cache"), name="audio")

# ── Router ─────────────────────────────────────────────────────────────────
router = APIRouter()

# ── Global RAG collection ──────────────────────────────────────────────────
_collection = None
_last_corrected_text = None
_last_pdf_name = None


def get_collection():
    global _collection
    if _collection is None:
        _collection = build_or_load_index(folder=OUTPUT_DIR)
    return _collection


# ══════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    tts: Optional[bool] = False
    tts_source: Optional[str] = "response"  # "response" or "summary"


class ChatResponse(BaseModel):
    answer:     str
    audio_url:  Optional[str] = None
    session_id: str


class QuizRequest(BaseModel):
    topic: str
    num_questions: int = 5


class QuizSubmitRequest(BaseModel):
    quiz_id: str
    answers: list


class StudyPlanRequest(BaseModel):
    days: int
    hours_per_day: float
    level: Optional[str] = "Intermediate"


# ══════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _build_audio_url(text: str) -> Optional[str]:
    """Generate audio safely."""
    try:
        safe_text = text[:1500]
        filename = safe_generate_tts(safe_text)
        if not filename:
            return None
        return f"/audio/{filename}"
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# GLOBAL STORES
# ══════════════════════════════════════════════════════════════════════════

QUIZ_STORE = {}


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "MindEdge API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok", "hf_token_set": bool(HF_TOKEN)}


# ── 1. Upload & Analyze Document ──────────────────────────────────────────
@router.post("/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    tts: bool = Query(False)
):
    """Accepts a PDF or image. Returns OCR text, corrected text, summary, graph analysis."""

    allowed = {".pdf", ".jpg", ".jpeg", ".png"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    temp_path = os.path.join("temp", f"{uuid.uuid4().hex}{ext}")
    os.makedirs("temp", exist_ok=True)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_text = extract_text(run_ocr(temp_path))
        if not raw_text:
            raise HTTPException(status_code=422, detail="Could not extract text from file.")

        corrected = correct_text(raw_text)
        summary   = summarize_text(corrected)

        global _last_corrected_text, _last_pdf_name
        _last_corrected_text = corrected
        _last_pdf_name = file.filename

        graph_results = extract_and_analyze_graphs(temp_path, hf_token=HF_TOKEN)
        graphs = [
            {"image": os.path.basename(r["image_path"]), "analysis": r["analysis"]}
            for r in graph_results
        ]

        from llm import _QW_MODEL_ID
        auto_scan_text(corrected, os.getenv("OPENAI_API_KEY"), _QW_MODEL_ID)
        auto_scan_text(summary, os.getenv("OPENAI_API_KEY"), _QW_MODEL_ID)  # 🔥 بدل التكرار


        stem = Path(file.filename).stem.replace(" ", "_")
        with open(os.path.join(OUTPUT_DIR, f"{stem}_corrected.md"), "w", encoding="utf-8") as f:
            f.write(corrected)
        with open(os.path.join(OUTPUT_DIR, f"{stem}_summary.md"), "w", encoding="utf-8") as f:
            f.write(summary)

        global _collection
        _collection = build_or_load_index(folder=OUTPUT_DIR)

        summary_audio_url = _build_audio_url(summary[:800]) if tts else None

        return {
            "status": "success",
            "document_name": file.filename,
            "raw_text": raw_text[:500] + "…" if len(raw_text) > 500 else raw_text,
            "corrected_text": corrected,
            "summary": summary,
            "audio_url": summary_audio_url,
            "graphs_analyzed": len(graphs),
            "graphs": graphs,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── 2. Chat with uploaded document ────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.tts_source not in {"response", "summary"}:
        raise HTTPException(status_code=400, detail="Invalid tts_source")

    collection = get_collection()

    passages = retrieve_passages(request.question, collection, top_k=5)
    context  = "\n\n".join(passages) if passages else "No context found."

    log_path = os.path.join(CHAT_LOG_DIR, f"session_{request.session_id}.md")
    memory   = ""
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            memory = f.read()[-3000:]

    # ── Split into system / user prompts for dev_ask_llm ─────────────────
    system_prompt = "You are a helpful study assistant."

    user_prompt = (
        f"[MEMORY]:\n{memory}\n\n"
        f"[PDF CONTEXT]:\n{context}\n\n"
        f"[QUESTION]: {request.question}\n\n"
        "Answer from context first. Format laws as [RULE: Name] Formula. English only."
    )

    # ── Call LLM via TTS-aware wrapper ────────────────────────────────────
    answer = dev_ask_llm(system_prompt, user_prompt)
    save_chat_log(request.question, answer, log_filename=f"session_{request.session_id}.md")

    # ── TTS logic ────────────────────────────────────────────────────────
    audio_url = None
    if request.tts:
        audio_url = _build_audio_url(answer[:800])

    return ChatResponse(
        answer=answer,
        audio_url=audio_url,
        session_id=request.session_id
    )


# ── 3. Quiz endpoints ─────────────────────────────────────────────────────
@app.post("/quiz/generate")
def generate_quiz(request: QuizRequest):
    collection = get_collection()

    passages = retrieve_passages(request.topic, collection, top_k=5)
    context = "\n\n".join(passages) if passages else "No context found."

    quiz = generate_quiz_from_context(context, request.num_questions)

    if isinstance(quiz, dict) and "error" in quiz:
        raise HTTPException(status_code=500, detail=quiz["error"])

    quiz_id = str(uuid.uuid4())
    QUIZ_STORE[quiz_id] = quiz

    quiz_for_user = []
    for q in quiz:
        q_copy = q.copy()
        q_copy.pop("answer", None)
        quiz_for_user.append(q_copy)

    return {
        "quiz_id": quiz_id,
        "quiz": quiz_for_user
    }


@app.post("/quiz/submit")
def submit_quiz(request: QuizSubmitRequest):
    quiz = QUIZ_STORE.get(request.quiz_id)

    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    if len(request.answers) != len(quiz):
        raise HTTPException(status_code=400, detail="Answers count mismatch")

    result = grade_quiz(request.answers, quiz)

    return result


@app.get("/summary")
def get_summary(
    filename: str = Query(..., description="Document filename"),
    tts: bool = Query(False, description="Enable TTS")
):
    stem         = Path(filename).stem.replace(" ", "_")
    summary_path = os.path.join(OUTPUT_DIR, f"{stem}_summary.md")

    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Summary not found. Upload the document first.")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = f.read()

    summary_audio_url = None

    if tts and summary and summary.strip():
        print("[🔊] Generating TTS for summary...")
        clean_summary = summary.replace("```", "").strip()
        filename_audio = safe_generate_tts(clean_summary[:1000])
        if filename_audio:
            summary_audio_url = f"/audio/{filename_audio}"
        else:
            print("[⚠️] TTS failed.")

    return {
        "filename": filename,
        "summary": summary,
        "audio_url": summary_audio_url
    }


# ── 4. Get extracted rules & definitions ──────────────────────────────────
import glob

@app.get("/rules")
def get_rules():
    """Returns all extracted physics rules."""

    folder = os.path.join(OUTPUT_DIR, "rules")

    if not os.path.exists(folder):
        return {"rules": []}

    results = []

    for file_path in glob.glob(os.path.join(folder, "*.txt")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    results.append(content)
        except Exception as e:
            print(f"[⚠️] Failed reading {file_path}: {e}")
            continue

    return {"rules": results}


@app.get("/definitions")
def get_definitions():
    folder = os.path.join(OUTPUT_DIR, "definitions")

    if not os.path.exists(folder):
        return {"definitions": []}

    results = []

    for file in glob.glob(os.path.join(folder, "*.txt")):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                results.append(content)

    return {"definitions": results}

# ── 5. Get graph analysis results ─────────────────────────────────────────
@app.get("/graphs")
def get_graphs():
    """Returns all graph analysis results."""
    results = []
    for path in glob.glob(os.path.join(OUTPUT_DIR, "graph_analysis", "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        results.append({"file": os.path.basename(path), "analysis": content})
    return {"graphs": results}


# ── 6. Study Plan ─────────────────────────────────────────────────────────
@router.post("/study-plan")
def get_study_plan(
    request: StudyPlanRequest,
    tts: bool = Query(False)
):
    """Generate study plan from last uploaded PDF."""

    # ── Check document exists ──
    if not _last_corrected_text:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

    # ── Validate input ──
    if request.days <= 0 or request.hours_per_day <= 0:
        raise HTTPException(status_code=400, detail="Invalid input values")

    # ── Extract topics ──
    topics = extract_topics_from_text(_last_corrected_text[:3000])

    if not topics:
        raise HTTPException(status_code=500, detail="Failed to extract topics.")

    # ── Generate plan ──
    plan = generate_study_plan(
        topics=topics,
        days=request.days,
        hours_per_day=request.hours_per_day,
        level=request.level,
    )

    # ── Convert plan to text (for TTS) ──
    plan_text = f"Your study plan for {_last_pdf_name} is ready"

    # ── TTS ──
    audio_url = None
    if tts:
        audio_url = _build_audio_url(plan_text[:1000])

    # ── Response ──
    return {
        "status": "success",
        "pdf": _last_pdf_name or "unknown",
        "topics_found": len(topics),
        "topics": topics,
        "plan": plan,
        "audio_url": audio_url
    }


# ── Register router ────────────────────────────────────────────────────────
app.include_router(router)