"""OpenRouter client: streams from free models, falling back down the list on errors."""
from __future__ import annotations

import logging
from typing import AsyncIterator

from openai import AsyncOpenAI

from app import config

log = logging.getLogger("app.llm")


class LLMUnavailable(RuntimeError):
    pass


_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL, timeout=60, max_retries=0,
            default_headers={"HTTP-Referer": config.APP_URL, "X-Title": "DarGlobal Property Assistant (demo)"},
        )
    return _client


async def stream_answer(messages: list[dict]) -> AsyncIterator[tuple[str, str]]:
    """Yield ("model", name) once, then ("delta", text) chunks. Tries each model in order;
    a model is only abandoned if it fails before producing any text."""
    errors = []
    for model in config.OPENROUTER_MODELS:
        produced = False
        try:
            stream = await client().chat.completions.create(
                model=model, messages=messages, stream=True, temperature=0.2, max_tokens=config.MAX_TOKENS,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content or ""
                if text:
                    if not produced:
                        produced = True
                        yield "model", getattr(chunk, "model", None) or model
                    yield "delta", text
            if produced:
                return
            errors.append(f"{model}: empty response")
        except Exception as e:  # noqa: BLE001 - rate limits, 404 for retired free models, timeouts
            if produced:
                yield "delta", "\n\n_(The model connection dropped; the answer may be incomplete.)_"
                return
            log.warning("model %s failed: %s", model, e)
            errors.append(f"{model}: {type(e).__name__}: {str(e)[:160]}")
    raise LLMUnavailable("; ".join(errors))
