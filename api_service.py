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

from config    import setup_output_dirs, OUTPUT_DIR, CHAT_LOG_DIR
from llm       import correct_text, summarize_text, dev_ask_llm
from ocr       import setup_tesseract, run_ocr, extract_text
from rag       import build_or_load_index, retrieve_passages, auto_scan_text, save_chat_log
from image     import extract_and_analyze_graphs
from tts_utils import safe_generate_tts
from quiz_utils import generate_quiz_from_context, grade_quiz

setup_output_dirs()
setup_tesseract()

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not set — add it to your .env file")

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

os.makedirs("audio_cache", exist_ok=True)
app.mount("/audio", StaticFiles(directory="audio_cache"), name="audio")

router = APIRouter()

_collection = None
_last_corrected_text = None
_last_pdf_name = None


def get_collection():
    global _collection
    if _collection is None:
        _collection = build_or_load_index(folder=OUTPUT_DIR)
    return _collection




class ChatRequest(BaseModel):
    question: str
    filename: str
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




QUIZ_STORE = {}




@app.get("/")
def root():
    return {"status": "MindEdge API is running "}


@app.get("/health")
def health():
    return {"status": "ok", "hf_token_set": bool(HF_TOKEN)}



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
            "graphs_analyzed": len(graphs),
            "graphs": graphs,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        log_path = os.path.join(CHAT_LOG_DIR, f"session_{request.session_id}.md")
        current_filename = request.filename

        if not current_filename or current_filename == "":
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line.startswith("FILE:"):
                        current_filename = first_line.replace("FILE:", "")

        if current_filename and not os.path.exists(log_path):
            os.makedirs(CHAT_LOG_DIR, exist_ok=True) 
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"FILE:{current_filename}\n")

        combined_context = ""
        if current_filename:
            stem = Path(current_filename).stem.replace(" ", "_")
            doc_path = os.path.join(OUTPUT_DIR, f"{stem}_corrected.md")
            if os.path.exists(doc_path):
                with open(doc_path, "r", encoding="utf-8") as f:
                    combined_context = f.read()[:5000] # نحدد الحجم عشان الـ Performance

        memory = ""
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                memory = "".join([l for l in all_lines if not l.startswith("FILE:")][-10:])

        system_prompt = "You are a helpful study assistant. Use the PDF Context to answer. If not found, use general knowledge but mention it."
        user_prompt = f"[MEMORY]:\n{memory}\n\n[PDF CONTEXT]:\n{combined_context}\n\n[QUESTION]: {request.question}"
        
        answer = dev_ask_llm(system_prompt, user_prompt)
        
        save_chat_log(request.question, answer, log_filename=f"session_{request.session_id}.md")

        return ChatResponse(
            answer=answer,
            audio_url=_build_audio_url(answer[:800]) if request.tts else None,
            session_id=request.session_id
        )

    except Exception as e:
        print(f"[ ERROR IN CHAT]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class QuizRequest(BaseModel):
    filename: str 
    num_questions: int = 5

class QuizSubmitRequest(BaseModel):
    quiz_id: str
    answers: list

QUIZ_STORE = {}

@app.post("/quiz/generate")
def generate_quiz(request: QuizRequest):
    stem = Path(request.filename).stem.replace(" ", "_")
    doc_path = os.path.join(OUTPUT_DIR, f"{stem}_corrected.md")

    if not os.path.exists(doc_path):
        raise HTTPException(status_code=404, detail="المحاضرة دي مش موجودة، ارفعيها الأول.")

    with open(doc_path, "r", encoding="utf-8") as f:
        context = f.read()

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
        "quiz": quiz_for_user,
        "filename": request.filename
    }


@app.post("/quiz/submit")
def submit_quiz(request: QuizSubmitRequest):
    quiz = QUIZ_STORE.get(request.quiz_id)

    if not quiz:
        raise HTTPException(status_code=404, detail="الكويز ده انتهى أو مش موجود.")

    if len(request.answers) != len(quiz):
        raise HTTPException(status_code=400, detail="عدد الإجابات مش مطابق لعدد الأسئلة.")

    
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
        print(" Generating TTS for summary...")
        clean_summary = summary.replace("```", "").strip()
        filename_audio = safe_generate_tts(clean_summary[:1000])
        if filename_audio:
            summary_audio_url = f"/audio/{filename_audio}"
        else:
            print(" TTS failed.")

    return {
        "filename": filename,
        "summary": summary,
        "audio_url": summary_audio_url
    }


import glob

@app.get("/rules")
def get_rules(filename: str = Query(..., description="Document filename (e.g., lecture1.pdf)")):
    """Returns physics/math rules strictly for the requested file."""
    
    stem = Path(filename).stem.replace(" ", "_")
    doc_path = os.path.join(OUTPUT_DIR, f"{stem}_corrected.md")

    if not os.path.exists(doc_path):
        raise HTTPException(status_code=404, detail="Lecture not found. Please upload it first.")

    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    system_prompt = "You are an expert scientific AI assistant."
    user_prompt = (
        "Extract physical laws and formulas from the lecture text. "
        "Format each rule clearly using the following Markdown template:\n\n"
        "###  [Rule Name]\n"
        "**Formula:** `[The Formula Here]`\n"
        "**Description:** [Brief explanation of what the law does]\n"
        "**Variables:**\n"
        "- `var1`: definition\n"
        "- `var2`: definition\n"
        "--- \n\n" 
        "STRICT RULE: Do NOT include general definitions. Only scientific laws and equations. "
        f"\n\nLecture Text:\n{content[:4000]}"
    )
    extracted_rules = dev_ask_llm(system_prompt, user_prompt)

    return {
        "filename": filename,
        "rules": extracted_rules
    }


@app.get("/definitions")
def get_definitions(filename: str = Query(..., description="Document filename (e.g., lecture1.pdf)")):
    """Returns definitions strictly for the requested file."""
    
    stem = Path(filename).stem.replace(" ", "_")
    doc_path = os.path.join(OUTPUT_DIR, f"{stem}_corrected.md")

    if not os.path.exists(doc_path):
        raise HTTPException(status_code=404, detail="Lecture not found. Please upload it first.")

    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    system_prompt = "You are an expert educational AI assistant."
    user_prompt = (
        "Extract all the key definitions, terms, and core concepts from the following lecture text. "
        "Format them clearly. If none are found, reply with 'No definitions found in this lecture'.\n\n"
        f"Lecture Text:\n{content[:4000]}"
    )

    extracted_definitions = dev_ask_llm(system_prompt, user_prompt)

    return {
        "filename": filename,
        "definitions": extracted_definitions
    }

@app.get("/graphs")
def get_graphs():
    """Returns all graph analysis results."""
    results = []
    for path in glob.glob(os.path.join(OUTPUT_DIR, "graph_analysis", "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        results.append({"file": os.path.basename(path), "analysis": content})
    return {"graphs": results}

@router.post("/study-plan")
def get_study_plan(
    request: StudyPlanRequest,
    filename: str = Query(..., description="The name of the lecture file"),
):
    import json
    
    stem = Path(filename).stem.replace(" ", "_")
    doc_path = os.path.join(OUTPUT_DIR, f"{stem}_corrected.md")

    if not os.path.exists(doc_path):
        raise HTTPException(status_code=400, detail="Lecture file not found.")

    with open(doc_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    system_prompt = "You are an expert academic personal coach."
    user_prompt = (
        f"Create a personalized study plan for a {request.level} student. "
        f"The plan must span {request.days} days, with {request.hours_per_day} hours of study per day. "
        "STRICT RULE: Return ONLY a JSON list of day objects. "
        "Structure: [{'day': 1, 'topic': '...', 'tasks': [{'task_name': '...', 'duration': '... min', 'priority': '...'}]}] "
        f"\n\nLecture Content:\n{text_content[:4000]}"
    )

    raw_response = dev_ask_llm(system_prompt, user_prompt)
    
    try:
        clean_json = raw_response.strip().replace("```json", "").replace("```", "")
        plan_json = json.loads(clean_json)
        
        return {
            "status": "success",
            "filename": filename,
            "student_level": request.level,
            "total_days": request.days,
            "hours_per_day": request.hours_per_day,
            "study_plan": plan_json
        }
    except Exception as e:
        return {"status": "error", "message": "AI formatting error", "raw": raw_response}
app.include_router(router)
