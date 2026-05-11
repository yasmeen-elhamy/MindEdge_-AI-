import json
import re
import uuid
from typing import Any, Dict, List
from llm import dev_ask_llm, generate_quiz_llm



def _clean_json_response(text: str) -> str:
    """
    Extracts and cleans the raw JSON string from the model's response, 
    removing Markdown formatting tags and extra whitespace.
    """
    if not text: return ""
    
    text = re.sub(r"```json\s*|```", "", text, flags=re.IGNORECASE).strip()
    
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        cleaned_text = match.group(0)
        
        cleaned_text = cleaned_text.replace('\\', '\\\\')
        cleaned_text = cleaned_text.replace('\\\\\\\\', '\\\\') 
        return cleaned_text
    return text

def _safe_json_loads(text: str) -> Any:
    """
    Safely parses and handles potential JSON errors during string conversion.
    """
    try:
        cleaned = _clean_json_response(text)
        return json.loads(cleaned)
    except Exception as e:
        print(f" JSON parse error: {e}")
        return {
            "error": "Invalid JSON from model", 
            "details": str(e),
            "raw_output": text[:300] 
        }

def generate_quiz_from_context(context: str, num_questions: int = 5) -> Any:
    if not context or not context.strip():
        return {"error": "Empty context"}
    
    response = generate_quiz_llm(context, num_questions)
    return _safe_json_loads(response)



def grade_quiz(user_answers: List[Any], quiz: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs semantic comparison using an LLM to ensure students receive credit 
    for correct answers, even if phrased differently from the model answer.
    """
    if not quiz or not user_answers:
        return {"error": "Quiz or answers missing"}

    comparison = []
    for i, q in enumerate(quiz):
        comparison.append({
            "question": q.get("question"),
            "model_answer": q.get("answer"),
            "student_answer": user_answers[i] if i < len(user_answers) else ""
        })

    system_prompt = "You are an expert academic grader."
    user_prompt = f"""
    Grade these student answers. 
    RULE: If the student's answer conveys the same core meaning as the model answer, mark 'is_correct': true.
    Be flexible with phrasing, strict with facts.

    DATA:
    {json.dumps(comparison, ensure_ascii=False)}

    Return ONLY JSON in this format:
    {{
      "score": number,
      "total": number,
      "results": [{{ "question": "...", "is_correct": true, "explanation": "..." }}]
    }}
    """

    response = dev_ask_llm(system_prompt, user_prompt)
    return _safe_json_loads(response)
