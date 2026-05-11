"""
quiz_utils.py — FINAL CLEAN VERSION
"""

import json
import re
from typing import Any, Dict, List

from llm import generate_quiz_llm, grade_quiz_llm


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────


def _clean_json_response(text: str) -> str:
    if not text:
        return text

    text = text.strip()

    # remove ```json ... ```
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    # remove leading "json" لو موجودة لوحدها
    if text.lower().startswith("json"):
        text = text[4:].strip()

    return text.strip()


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception as e:
        print("[❌] JSON parse error:", e)
        return {
            "error": "Invalid JSON from model",
            "raw": text
        }


# ─────────────────────────────────────────
# Quiz Generation
# ─────────────────────────────────────────

def generate_quiz_from_context(
    context: str,
    num_questions: int = 5
) -> Dict[str, Any]:

    if not context or not context.strip():
        return {"error": "Empty context"}

    response = generate_quiz_llm(context, num_questions)

    cleaned = _clean_json_response(response)
    parsed  = _safe_json_loads(cleaned)

    return parsed


# ─────────────────────────────────────────
# Quiz Grading
# ─────────────────────────────────────────

def grade_quiz(
    user_answers: List[Any],
    quiz: List[Dict[str, Any]]
) -> Dict[str, Any]:

    if not quiz:
        return {"error": "Quiz is empty"}

    if not user_answers:
        return {"error": "Answers are empty"}

    if len(user_answers) != len(quiz):
        return {
            "error": "Answers count mismatch",
            "expected": len(quiz),
            "received": len(user_answers)
        }

    response = grade_quiz_llm(quiz, user_answers)

    cleaned = _clean_json_response(response)
    parsed  = _safe_json_loads(cleaned)
    return parsed

