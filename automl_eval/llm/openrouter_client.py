"""OpenRouter chat-completions client for GRACE experiments"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import requests

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class LLMCallResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str = ""
    raw_finish_reason: str | None = None
    retries: int = 0
    error: str | None = None


@dataclass
class OpenRouterClient:
    """OpenAI-compatible OpenRouter client with retries and usage accounting."""

    model: str
    api_key: str | None = None
    base_url: str = OPENROUTER_BASE_URL
    timeout: int = 180
    max_retries: int = 5
    backoff_base: float = 2.0
    backoff_cap: float = 60.0
    # Optional headers OpenRouter recommends for ranking/attribution.
    http_referer: str | None = None
    x_title: str | None = "GRACE-experiments"
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENROUTER_API_KEY")

    def configured(self) -> bool:
        return bool(self.api_key) and bool(self.model)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title
        return headers

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
    ) -> LLMCallResult:
        if not self.configured():
            raise RuntimeError(
                "OpenRouterClient is not configured: set OPENROUTER_API_KEY and a model slug."
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_completion_tokens is not None:
            payload["max_tokens"] = max_completion_tokens
        if seed is not None:
            payload["seed"] = seed

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(
                    self.base_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choice = (data.get("choices") or [{}])[0]
                    msg = choice.get("message", {}) or {}
                    text = msg.get("content", "") or ""
                    usage = data.get("usage", {}) or {}
                    return LLMCallResult(
                        text=text,
                        input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                        model=data.get("model", self.model),
                        raw_finish_reason=choice.get("finish_reason"),
                        retries=attempt,
                    )
                # Retry on rate-limit and server errors.
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    self._sleep_backoff(attempt, resp)
                    continue
                # Non-retryable.
                return LLMCallResult(
                    text="",
                    model=self.model,
                    retries=attempt,
                    error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._sleep_backoff(attempt, None)
                continue
            except Exception as exc:  # noqa: BLE001 - defensive
                return LLMCallResult(
                    text="",
                    model=self.model,
                    retries=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )

        return LLMCallResult(
            text="",
            model=self.model,
            retries=self.max_retries,
            error=last_error or "exhausted retries",
        )

    def _sleep_backoff(self, attempt: int, resp: requests.Response | None) -> None:
        # Honour Retry-After when present, else exponential backoff with jitter.
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    time.sleep(min(float(retry_after), self.backoff_cap))
                    return
                except ValueError:
                    pass
        delay = min(self.backoff_base**attempt, self.backoff_cap)
        delay = delay * (0.5 + random.random())
        time.sleep(delay)
