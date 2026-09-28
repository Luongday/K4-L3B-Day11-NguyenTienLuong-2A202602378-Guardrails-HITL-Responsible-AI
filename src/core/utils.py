"""
Lab 11 — Helper Utilities
"""
import asyncio
import re
import time

from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner

# Gemini free tier is ~5 requests/minute per model; retry transient
# 429 RESOURCE_EXHAUSTED / 503 UNAVAILABLE with backoff instead of failing
# a whole attack run because of a single rate-limit hiccup. A proactive
# minimum spacing between calls keeps us under the 5 RPM quota in the
# first place, so we hit the ceiling less often.
_GEMINI_MIN_SPACING_SECONDS = 13.0
_GEMINI_RETRY_ATTEMPTS = 12
_GEMINI_RETRY_FALLBACK_SECONDS = (15, 20, 30, 40, 60)
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", re.IGNORECASE)
_gemini_last_call_at = 0.0


def _is_transient_gemini_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "UNAVAILABLE" in text or "503" in text or "429" in text


def _parse_retry_delay(exc: Exception, attempt: int) -> float:
    match = _RETRY_DELAY_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 1
    return _GEMINI_RETRY_FALLBACK_SECONDS[
        min(attempt, len(_GEMINI_RETRY_FALLBACK_SECONDS) - 1)
    ]


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    global _gemini_last_call_at
    for attempt in range(_GEMINI_RETRY_ATTEMPTS):
        wait = _GEMINI_MIN_SPACING_SECONDS - (time.monotonic() - _gemini_last_call_at)
        if wait > 0:
            await asyncio.sleep(wait)
        _gemini_last_call_at = time.monotonic()
        try:
            final_response = ""
            async for event in runner.run_async(
                user_id=user_id, session_id=session.id, new_message=content
            ):
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response += part.text
            return final_response, session
        except Exception as e:
            is_last = attempt == _GEMINI_RETRY_ATTEMPTS - 1
            if is_last or not _is_transient_gemini_error(e):
                raise
            delay = _parse_retry_delay(e, attempt)
            print(f"  (rate-limited by Gemini, retrying in {delay:.0f}s...)")
            await asyncio.sleep(delay)
