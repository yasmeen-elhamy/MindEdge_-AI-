"""
topic_extractor.py — Extract clean study topics from text
=========================================================
Uses LLM (dev_ask_llm) to extract structured topics with fallback handling.
"""

from typing import List
from llm import dev_ask_llm


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MAX_INPUT_CHARS = 2500
MAX_TOPICS      = 10


# ─────────────────────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────────────────────
def extract_topics_from_text(text: str) -> List[str]:
    """
    Extracts a clean list of study topics from input text.

    Returns:
        List[str] → ["Topic1", "Topic2", ...]
    """

    if not text or not text.strip():
        return ["General Concepts"]

    # ── truncate (important for stability) ──
    clean_text = text[:MAX_INPUT_CHARS]

    # ── prompt ─────────────────────────────
    system_prompt = (
        "You are an expert at extracting clean academic topics from educational text. "
        "Return ONLY a list of concise topic names."
    )

    user_prompt = f"""
Extract the MAIN study topics from the text below.

RULES:
- Return ONLY topic names
- No explanations
- No sentences
- Each topic on a NEW LINE
- Maximum {MAX_TOPICS} topics
- Ignore page numbers, noise, and formatting

TEXT:
{clean_text}
"""

    try:
        result = dev_ask_llm(system_prompt, user_prompt)

        topics = _clean_topics(result)

        # ── fallback if weak output ──
        if len(topics) < 2:
            return _fallback_topics(clean_text)

        return topics

    except Exception as e:
        print(f"[⚠️] Topic extraction failed: {e}")
        return _fallback_topics(clean_text)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _clean_topics(raw_output: str) -> List[str]:
    """Cleans LLM output into a list of topics."""

    topics = []

    for line in raw_output.split("\n"):
        line = line.strip()

        # remove bullets
        line = line.lstrip("-•*0123456789. ").strip()

        # skip garbage
        if not line:
            continue
        if len(line) < 3:
            continue
        if len(line) > 80:
            continue

        topics.append(line)

    # remove duplicates
    topics = list(dict.fromkeys(topics))

    return topics[:MAX_TOPICS]


def _fallback_topics(text: str) -> List[str]:
    """
    Simple fallback using heuristics when LLM fails.
    """

    # very basic fallback (safe, not smart)
    keywords = []

    for word in text.split():
        word = word.strip(",.()[]")

        if len(word) > 5 and word[0].isupper():
            keywords.append(word)

    keywords = list(dict.fromkeys(keywords))

    if len(keywords) >= 3:
        return keywords[:5]

    return ["Core Concepts", "Key Principles", "Applications"]