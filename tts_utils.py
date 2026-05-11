"""
tts_utils.py — Text-to-Speech Module
======================================
Handles all TTS logic: caching, generation, fault tolerance.
"""

import os
import hashlib
from typing import Optional
from openai import OpenAI, APIError

# ── Constants ──────────────────────────────────────────────────────────────
AUDIO_DIR      = "audio_cache"
TTS_MODEL      = "gpt-4o-mini-tts"
TTS_VOICE      = "alloy"
TTS_FORMAT     = "mp3"
TTS_MAX_CHARS  = 1500

os.makedirs(AUDIO_DIR, exist_ok=True)

# ── Client ─────────────────────────────────────────────────────────────────
_TTS_CLIENT: Optional[OpenAI] = None


def _get_tts_client() -> OpenAI:
    global _TTS_CLIENT
    if _TTS_CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("[tts] OPENAI_API_KEY is not set.")
        _TTS_CLIENT = OpenAI(api_key=api_key)
    return _TTS_CLIENT


# ── Helpers ────────────────────────────────────────────────────────────────

def _hash_text(text: str) -> str:
    """Returns a stable SHA-256 hex digest for cache-keying."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path(filename: str) -> str:
    return os.path.join(AUDIO_DIR, filename)


def _is_cached(filename: str) -> bool:
    return os.path.isfile(_cache_path(filename))


# ── Core ───────────────────────────────────────────────────────────────────

def generate_tts(text: str) -> str:
    """
    Converts text to speech and caches the result.

    Returns:
        filename (str): The MP3 filename inside AUDIO_DIR.

    Raises:
        APIError: Propagated on TTS API failure.
    """
    truncated = text[:TTS_MAX_CHARS]
    digest    = _hash_text(truncated)
    filename  = f"{digest}.{TTS_FORMAT}"

    if _is_cached(filename):
        print(f"[🔊] TTS cache hit → {filename}")
        return filename

    print(f"[🔊] Generating TTS | len={len(text)} | cached=False")
    client   = _get_tts_client()
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=truncated,
        response_format=TTS_FORMAT,
    )
    response.stream_to_file(_cache_path(filename))
    print(f"[✅] TTS saved → {filename}")
    return filename


def safe_generate_tts(text: str) -> Optional[str]:
    """
    Fault-tolerant wrapper around generate_tts.

    Returns:
        filename (str) on success, None on any failure.
    """
    try:
        return generate_tts(text)
    except Exception as exc:
        print(f"[⚠️] TTS failed (non-fatal): {exc}")
        return None
