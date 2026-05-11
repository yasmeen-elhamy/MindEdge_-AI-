"""
study_plan.py — Production-ready Study Plan Generator
"""

from __future__ import annotations

import math
import logging
import re
import time
from typing import Optional, List, Dict

from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError

import os
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
_QW_BASE_URL      = "https://api.openai.com/v1"
_QW_MODEL_ID      = "gpt-4o-mini"
_QW_MAX_TOKENS    = 2000
_QW_TEMPERATURE   = 0.7
_QW_TOP_P         = 0.9
_QW_TIMEOUT       = 60
_QW_MAX_RETRIES   = 3
_QW_BACKOFF_BASE  = 2.0

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
log = logging.getLogger("StudyPlan")

# ─────────────────────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────────────────────
_CLIENT: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    return _CLIENT


# ─────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────
def llm_chat(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    client = _get_client()

    for attempt in range(1, _QW_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=_QW_MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=_QW_TEMPERATURE,
                top_p=_QW_TOP_P,
                timeout=_QW_TIMEOUT,
            )

            return response.choices[0].message.content.strip()

        except (APIStatusError, APITimeoutError, APIConnectionError) as e:
            if attempt == _QW_MAX_RETRIES:
                return f"Error: {str(e)}"
            time.sleep(_QW_BACKOFF_BASE ** attempt)

        except Exception as e:
            return f"Error: {str(e)}"

    return "Error: LLM failed"


# ─────────────────────────────────────────────────────────────
# FALLBACK
# ─────────────────────────────────────────────────────────────
def _fallback_plan(subject, days, hours, level, topics):
    result = []
    for i in range(1, days + 1):
        topic = topics[(i - 1) % len(topics)]
        result.append(f"## Day {i}\n- Topic: {topic}\n- Study {hours}h\n")

    return "\n".join(result)


# ─────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────
def _build_prompt(subject, days, hours, level, topics):

    topics_str = "\n".join(f"- {t}" for t in topics)

    system = "You are an expert study planner."

    user = f"""
Create a detailed study plan:

Subject: {subject}
Days: {days}
Hours/day: {hours}
Level: {level}

Topics:
{topics_str}

Format:
## Day 1
- Topic:
- Tasks:
- Time:

Repeat for all days.
"""

    return system, user


# ─────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────
def _parse_plan(text: str) -> Dict[str, str]:
    result = {}
    current = None
    buffer = []

    for line in text.splitlines():
        if re.match(r"^##\s*Day", line):
            if current:
                result[current] = "\n".join(buffer).strip()
            current = line.strip()
            buffer = []
        elif current:
            buffer.append(line)

    if current:
        result[current] = "\n".join(buffer).strip()

    if not result:
        result["Study Plan"] = text

    return result


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────
def generate_study_plan(
    topics: List[str],
    days: int,
    hours_per_day: int,
    subject: Optional[str] = None,
    level: str = "Intermediate",
    collection=None
) -> Dict[str, str]:

    # ── Validation ──
    if not topics:
        raise ValueError("Topics list is empty")

    if isinstance(topics, str):
        topics = [t.strip() for t in topics.split("\n") if t.strip()]

    topics = [t for t in topics if t.strip()]

    if not topics:
        raise ValueError("Topics invalid after cleaning")

    if subject is None:
        subject = next((t for t in topics if t.strip()), "Study Material")

    # ── Build prompt ──
    system, user = _build_prompt(subject, days, hours_per_day, level, topics)

    max_tokens = min(2000, 80 * days)

    # ── Generate ──
    plan_text = llm_chat(system, user, max_tokens)

    if plan_text.startswith("Error"):
        log.warning("LLM failed → fallback")
        plan_text = _fallback_plan(subject, days, hours_per_day, level, topics)

    # ── Parse ──
    return _parse_plan(plan_text)