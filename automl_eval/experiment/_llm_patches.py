"""Runtime patches that teach the extracted LLM client about OpenRouter extras."""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


_PROVIDER_ENV = "OPENROUTER_PROVIDER_JSON"
_REASONING_ENV = "OPENROUTER_REASONING_JSON"
_SEED_ENV = "OPENROUTER_LLM_SEED"


def _payload_extras() -> dict[str, Any]:
    """Return the extras dict {provider, reasoning} based on env vars."""
    extras: dict[str, Any] = {}
    prov = os.environ.get(_PROVIDER_ENV, "").strip()
    if prov:
        try:
            extras["provider"] = json.loads(prov)
        except Exception:  # noqa: BLE001
            pass
    reas = os.environ.get(_REASONING_ENV, "").strip()
    if reas:
        try:
            extras["reasoning"] = json.loads(reas)
        except Exception:  # noqa: BLE001
            pass
    seed = os.environ.get(_SEED_ENV, "").strip()
    if seed:
        try:
            extras["seed"] = int(seed)
        except ValueError:
            pass
    return extras


def patch_openai_compatible_llm(klass) -> None:
    """Replace ``klass.invoke`` with a version that injects OpenRouter extras."""
    if getattr(klass, "_grace_v3_4_patched", False):
        return

    def invoke(self, messages, *, context=None, max_tokens_override=None):
        from automl_eval.experiment._regimes_extracted import (
            LLMResponse,
            LLMEndpointError,
            LLMReadTimeoutError,
        )

        effective_max_tokens = (
            self.max_tokens
            if max_tokens_override is None
            else min(self.max_tokens, max_tokens_override)
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": effective_max_tokens,
        }
        payload.update(_payload_extras())
        encoded_payload = json.dumps(payload).encode("utf-8")
        total_attempts = self.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            started = time.time()
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            x_title = os.environ.get("OPENROUTER_X_TITLE")
            if x_title:
                headers["X-Title"] = x_title
            referer = os.environ.get("OPENROUTER_HTTP_REFERER")
            if referer:
                headers["HTTP-Referer"] = referer

            req = urlrequest.Request(
                self.endpoint_url,
                data=encoded_payload,
                headers=headers,
                method="POST",
            )
            try:
                with urlrequest.urlopen(req, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                elapsed = time.time() - started
                self._log_attempt(
                    attempt=attempt,
                    status="ok",
                    elapsed=elapsed,
                    messages=messages,
                    context=context,
                    effective_max_tokens=effective_max_tokens,
                )
                usage = body.get("usage", {}) or {}
                content = body["choices"][0]["message"]["content"]
                return LLMResponse(
                    content=content,
                    usage_metadata={
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                    },
                )
            except urlerror.HTTPError as exc:
                elapsed = time.time() - started
                detail = exc.read().decode("utf-8", errors="replace")
                message = f"LLM HTTP error {exc.code}: {detail[:1000]}"
                self._log_attempt(
                    attempt=attempt,
                    status=f"http_{exc.code}",
                    elapsed=elapsed,
                    messages=messages,
                    context=context,
                    error=message,
                    effective_max_tokens=effective_max_tokens,
                )
                if (
                    exc.code in {408, 429, 500, 502, 503, 504}
                    and attempt < total_attempts
                ):
                    self._wait_before_retry(attempt, message)
                    continue
                raise LLMEndpointError(message) from exc
            except (TimeoutError, socket.timeout) as exc:
                elapsed = time.time() - started
                message = f"LLM read timed out after {elapsed:.1f}s on attempt {attempt}/{total_attempts}."
                self._log_attempt(
                    attempt=attempt,
                    status="timeout",
                    elapsed=elapsed,
                    messages=messages,
                    context=context,
                    error=message,
                    effective_max_tokens=effective_max_tokens,
                )
                if attempt < total_attempts:
                    self._wait_before_retry(attempt, message)
                    continue
                raise LLMReadTimeoutError(message) from exc
            except urlerror.URLError as exc:
                elapsed = time.time() - started
                message = f"Could not reach LLM endpoint: {exc}"
                self._log_attempt(
                    attempt=attempt,
                    status="url_error",
                    elapsed=elapsed,
                    messages=messages,
                    context=context,
                    error=message,
                    effective_max_tokens=effective_max_tokens,
                )
                if attempt < total_attempts:
                    self._wait_before_retry(attempt, message)
                    continue
                raise LLMEndpointError(message) from exc
        raise AssertionError("Unreachable retry loop termination.")

    klass.invoke = invoke
    klass._grace_v3_4_patched = True
