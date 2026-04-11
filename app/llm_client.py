"""
================================================================================
L.O.V.E. Relationship Support Agent — LLM Client
RSM 8430 Group 18
================================================================================

Thin wrapper around the LLM endpoint. Swap the implementation here to change
providers without touching any other file.

Currently supports:
  - Course-provided Qwen endpoint (OpenAI-compatible API)
  - Any OpenAI-compatible endpoint

Configuration via environment variables:
  LLM_API_BASE  — base URL (e.g. http://localhost:8000/v1)
  LLM_API_KEY   — API key (default: "empty" for local endpoints)
  LLM_MODEL     — model name (default: Qwen3-30b-a3b-fp8)

Usage:
    from app.llm_client import generate_text
    response = generate_text("You are a helpful assistant.", "Hello!")
"""

from __future__ import annotations

import os
import time

import requests

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

LLM_API_BASE: str = os.environ.get("LLM_API_BASE", "http://localhost:8000/v1")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "empty")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "Qwen3-30b-a3b-fp8")

# Safety: cap generation length to avoid runaway responses
MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.7"))


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_text(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_seconds: int = 60,
    retries: int = 2,
) -> str:
    """
    Send a system + user message to the LLM and return the assistant's reply.

    Args:
        system_prompt: The system-level instruction.
        user_prompt:   The user-level message / prompt.

    Returns:
        The LLM's response text.

    Raises:
        RuntimeError: If the API call fails after the request completes.
        requests.RequestException: If the network call itself fails.
    """
    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
        "temperature": temperature if temperature is not None else TEMPERATURE,
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Unexpected LLM response format: {data}") from exc
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(0.4 * (attempt + 1))

    if last_error:
        raise last_error
    raise RuntimeError("LLM request failed with unknown error.")
