"""Audit-log tests for the optional OpenAI-compatible notebook router client."""

from __future__ import annotations

from automl_eval.llm.router_client import OpenAICompatibleRouterClient


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"choices": [{"message": {"content": "ACTION: PLAN\nUse a pipeline."}}]}


def test_router_audit_log_snapshots_messages_at_request_time(monkeypatch) -> None:
    monkeypatch.setattr(
        "automl_eval.llm.router_client.requests.post",
        lambda *args, **kwargs: _FakeResponse(),
    )
    router = OpenAICompatibleRouterClient(
        "https://router.invalid/v1/chat/completions", "token", "model"
    )
    messages = [
        {"role": "system", "content": "contract"},
        {"role": "user", "content": "turn 0"},
    ]
    router.complete(messages)
    messages.append({"role": "assistant", "content": "later mutation"})
    assert len(router.communication_log[0]["request"]["messages"]) == 2
    assert all(
        message["content"] != "later mutation"
        for message in router.communication_log[0]["request"]["messages"]
    )
