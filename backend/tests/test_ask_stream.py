import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Self

import anthropic
import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openmfp_copilot.api.dependencies import get_anthropic_client


@dataclass
class FakeStream:
    chunks: list[str]
    stop_reason: str
    error: Exception | None

    async def __aenter__(self) -> Self:
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    @property
    def text_stream(self) -> AsyncIterator[str]:
        return self._texts()

    async def _texts(self) -> AsyncIterator[str]:
        for chunk in self.chunks:
            yield chunk

    async def get_final_message(self) -> SimpleNamespace:
        return SimpleNamespace(stop_reason=self.stop_reason)


@dataclass
class FakeStreamingMessages:
    chunks: list[str] = field(default_factory=lambda: ["Hello ", "world"])
    stop_reason: str = "end_turn"
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        return FakeStream(self.chunks, self.stop_reason, self.error)


@pytest.fixture
def fake_messages(app: FastAPI) -> FakeStreamingMessages:
    messages = FakeStreamingMessages()
    fake_client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    app.dependency_overrides[get_anthropic_client] = lambda: fake_client
    return messages


def parse_events(body: str) -> list[tuple[str, Any]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_ask_stream_sends_text_deltas_then_done(
    client: TestClient, fake_messages: FakeStreamingMessages
) -> None:
    response = client.post("/ask/stream", json={"question": "What is portal-ui-lib?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert parse_events(response.text) == [
        ("delta", {"text": "Hello "}),
        ("delta", {"text": "world"}),
        ("done", {}),
    ]


def test_ask_stream_sends_question_with_configured_model(
    client: TestClient, fake_messages: FakeStreamingMessages
) -> None:
    client.post("/ask/stream", json={"question": "What is portal-ui-lib?"})

    [call] = fake_messages.calls
    assert call["model"] == "claude-haiku-5-5"
    assert call["max_tokens"] == 64_000
    assert call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default"
    assert call["messages"] == [{"role": "user", "content": "What is portal-ui-lib?"}]
    assert "output_format" not in call


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_ask_stream_ends_with_error_when_claude_stops_without_answer(
    client: TestClient, fake_messages: FakeStreamingMessages, stop_reason: str
) -> None:
    fake_messages.stop_reason = stop_reason

    response = client.post("/ask/stream", json={"question": "What is portal-ui-lib?"})

    assert parse_events(response.text) == [
        ("delta", {"text": "Hello "}),
        ("delta", {"text": "world"}),
        ("error", {"detail": f"Claude did not return an answer: {stop_reason}"}),
    ]


def test_ask_stream_sends_error_event_when_claude_api_fails(
    client: TestClient, fake_messages: FakeStreamingMessages
) -> None:
    fake_messages.error = anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    )

    response = client.post("/ask/stream", json={"question": "What is portal-ui-lib?"})

    assert response.status_code == 200
    assert parse_events(response.text) == [
        ("error", {"detail": "The Claude API request failed"}),
    ]


def test_ask_stream_rejects_empty_question_without_calling_claude(
    client: TestClient, fake_messages: FakeStreamingMessages
) -> None:
    response = client.post("/ask/stream", json={"question": ""})

    assert response.status_code == 422
    assert fake_messages.calls == []
