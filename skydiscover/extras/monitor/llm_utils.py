"""
Shared LLM API calling utility for monitor features.

Extracted from MonitorServer to be reusable by AIFeedbackReader
and other components that need simple LLM API calls.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)


def call_llm_api(
    *,
    model: str,
    system: str,
    user_message: str,
    api_key: str = "",
    api_base: str = "https://api.openai.com/v1",
    max_tokens: int = 8192,
    timeout: int = 30,
) -> str:
    """Call an OpenAI-compatible chat completions API (blocking).

    Args:
        model: Model name (e.g. "gpt-5-mini", "claude-sonnet-4-20250514").
        system: System message content.
        user_message: User message content.
        api_key: API key. Falls back to OPENAI_API_KEY env var.
        api_base: API base URL.
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.

    Returns:
        The assistant's response text.

    Raises:
        RuntimeError: On API errors or network failures.
    """
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    messages: List[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    url = f"{api_base.rstrip('/')}/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM API error {e.code}: {error_body}") from e
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {e}") from e
