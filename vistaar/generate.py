"""Gemini generation client.

Talks to the REST API directly so we control `thinkingBudget`. Gemini 3.x
defaults to thinking mode, which spends the whole output budget on
`thoughtsTokenCount` and returns an empty candidate -- the failure mode that
made the original app look like it was permanently "busy".
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_PROMPT = (
    "You are a retrieval-grounded question answering system.\n"
    "Rules:\n"
    "1. Answer ONLY from the provided context passages.\n"
    "2. Cite every claim with the passage ids that support it.\n"
    "3. If the context does not contain the answer, set found=false and do not guess.\n"
    "4. Be concise. Prefer a short span or sentence over a paragraph.\n"
    "Respond with strict JSON only, no markdown fences, using this schema:\n"
    '{"answer": "<string>", "citations": [<passage ids>], "found": true|false}'
)


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class RateLimiter:
    """Simple thread-safe token bucket in requests/second."""

    def __init__(self, rate_per_second: float, burst: int = 4):
        self.rate = max(rate_per_second, 0.01)
        self.capacity = max(burst, 1)
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            time.sleep(wait)


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash",
        fallback_models: tuple[str, ...] = (),
        timeout: int = 60,
        rate_per_second: float = 4.0,
        max_retries: int = 3,
        thinking_budget: int = 0,
    ):
        self.api_key = api_key
        self.models = (model, *fallback_models)
        self.timeout = timeout
        self.max_retries = max_retries
        self.thinking_budget = thinking_budget
        self.limiter = RateLimiter(rate_per_second)

    def complete(self, prompt: str, system: str = SYSTEM_PROMPT, temperature: float = 0.0) -> Completion:
        last_error: Exception | None = None
        for model in self.models:
            for attempt in range(self.max_retries):
                self.limiter.acquire()
                started = time.monotonic()
                try:
                    payload = {
                        "systemInstruction": {"parts": [{"text": system}]},
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": temperature,
                            "maxOutputTokens": 1024,
                            "thinkingConfig": {"thinkingBudget": self.thinking_budget},
                        },
                    }
                    request = urllib.request.Request(
                        _ENDPOINT.format(model=model),
                        data=json.dumps(payload).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": self.api_key,
                        },
                    )
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        body = json.load(response)

                    candidate = body["candidates"][0]
                    usage = body.get("usageMetadata", {})
                    parts = candidate.get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts).strip()

                    if not text:
                        # Empty candidate: usually MAX_TOKENS from thinking budget.
                        last_error = RuntimeError(
                            f"{model} returned no text (finishReason="
                            f"{candidate.get('finishReason')})"
                        )
                        time.sleep(min(2 ** attempt, 8))
                        continue

                    return Completion(
                        text=text,
                        model=body.get("modelVersion", model),
                        prompt_tokens=usage.get("promptTokenCount", 0),
                        output_tokens=usage.get("candidatesTokenCount", 0),
                        latency_s=time.monotonic() - started,
                    )
                except urllib.error.HTTPError as exc:
                    detail = ""
                    try:
                        detail = json.loads(exc.read())["error"]["message"][:200]
                    except Exception:
                        pass
                    last_error = RuntimeError(f"{model} HTTP {exc.code}: {detail}")
                    if exc.code in (400, 401, 403, 404):
                        break  # bad key / model gone: try next model
                    time.sleep(min(2 ** attempt, 8))
                except Exception as exc:  # network, timeout, malformed JSON
                    last_error = exc
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


def parse_answer(text: str) -> dict:
    """Parse the model's JSON reply, tolerating fences and stray prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned
        cleaned = cleaned.removeprefix("json").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return {
                "answer": str(data.get("answer", "")).strip(),
                "citations": [str(c) for c in data.get("citations", []) or []],
                "found": bool(data.get("found", True)),
            }
    except json.JSONDecodeError:
        pass
    return {"answer": cleaned, "citations": [], "found": bool(cleaned)}
