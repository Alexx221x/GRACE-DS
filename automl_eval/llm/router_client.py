"""Small OpenAI-compatible router client with immutable audit logging."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import requests


class OpenAICompatibleRouterClient:
    def __init__(
        self,
        url: str,
        token: str,
        model: str,
        timeout: int = 180,
        audit_jsonl_path: str | Path | None = None,
    ) -> None:
        self.url = url
        self.token = token
        self.model = model
        self.timeout = timeout
        self.audit_jsonl_path = Path(audit_jsonl_path) if audit_jsonl_path else None
        self.communication_log: list[dict[str, Any]] = []

    def clear_log(self) -> None:
        self.communication_log.clear()
        if self.audit_jsonl_path and self.audit_jsonl_path.exists():
            self.audit_jsonl_path.unlink()

    def _record(self, record: dict[str, Any]) -> None:
        frozen = deepcopy(record)
        self.communication_log.append(frozen)
        if self.audit_jsonl_path:
            self.audit_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(frozen, ensure_ascii=False) + "\n")

    def configured(self) -> bool:
        values = (self.url, self.token, self.model)
        return all(value and "<YOUR_" not in value for value in values)

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_completion_tokens: int | None = None,
    ) -> str:
        if not self.configured():
            raise RuntimeError(
                "Fill LLM_ROUTER_URL, LLM_ROUTER_TOKEN and LLM_MODEL before running the router episode."
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": deepcopy(messages),
            "temperature": temperature,
        }
        if max_completion_tokens is not None:
            payload["max_completion_tokens"] = max_completion_tokens
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        response = requests.post(
            self.url, headers=headers, json=payload, timeout=self.timeout
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        self._record(
            {
                "elapsed_seconds": elapsed,
                "request": payload,
                "response": body,
                "assistant_action": content,
            }
        )
        return content
