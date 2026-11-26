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
from openmfp_copilot.ask.service import MAX_TOOL_ROUNDS
from tests.fakes import get_readme_call


@dataclass
class StreamRound:
    chunks: list[str]
    stop_reason: str = "end_turn"
    content: list[object] = field(default_factory=list)


@dataclass
class FakeStream:
    round: StreamRound
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
        for chunk in self.round.chunks:
            yield chunk

    async def get_final_message(self) -> SimpleNamespace:
        content = self.round.content
        return SimpleNamespace(
            stop_reason=self.round.stop_reason,
            content=content,
            to_param=lambda: {"role": "assistant", "content": content},
        )


@dataclass
class FakeStreamingMessages:
    rounds: list[StreamRound] = field(default_factory=lambda: [StreamRound(["Hello ", "world"])])
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        current = self.rounds.pop(0) if len(self.rounds) > 1 else self.rounds[0]
        return FakeStream(current, self.error)


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


def test_ask_stream_offers_get_readme_tool(
    client: TestClient, fake_messages: FakeStreamingMessages
) -> None:
    client.post("/ask/stream", json={"question": "What is portal-ui-lib?"})

    [call] = fake_messages.calls
    assert [tool["name"] for tool in call["tools"]] == ["get_readme"]
    assert "read its README with the get_readme tool" in call["system"]


def test_ask_stream_announces_tool_call_and_streams_answer_after_reading_readme(
    client: TestClient,
    fake_messages: FakeStreamingMessages,
    github_requests: list[httpx2.Request],
) -> None:
    readme_call = get_readme_call("openmfp", "portal-ui-lib")
    fake_messages.rounds = [
        StreamRound(["Let me read the README. "], "tool_use", [readme_call]),
        StreamRound(["It bootstraps ", "the portal."]),
    ]

    response = client.post("/ask/stream", json={"question": "What is portal-ui-lib?"})

    assert parse_events(response.text) == [
        ("delta", {"text": "Let me read the README. "}),
        (
            "tool_call",
            {"name": "get_readme", "input": {"organization": "openmfp", "name": "portal-ui-lib"}},
        ),
        ("delta", {"text": "It bootstraps "}),
        ("delta", {"text": "the portal."}),
        ("done", {}),
    ]
    [request] = github_requests
    assert request.url.path == "/repos/openmfp/portal-ui-lib/readme"
    _, second_call = fake_messages.calls
    assert second_call["messages"] == [
        {"role": "user", "content": "What is portal-ui-lib?"},
        {"role": "assistant", "content": [readme_call]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "# portal-ui-lib README",
                }
            ],
        },
    ]


def test_ask_stream_ends_with_error_when_claude_keeps_calling_tools(
    client: TestClient,
    fake_messages: FakeStreamingMessages,
    github_requests: list[httpx2.Request],
) -> None:
    fake_messages.rounds = [
        StreamRound([], "tool_use", [get_readme_call("openmfp", "portal-ui-lib")])
    ]

    response = client.post("/ask/stream", json={"question": "What is portal-ui-lib?"})

    events = parse_events(response.text)
    assert events[-1] == (
        "error",
        {"detail": "Claude did not return an answer: too many tool calls"},
    )
    assert [name for name, _ in events].count("tool_call") == MAX_TOOL_ROUNDS
    assert len(github_requests) == MAX_TOOL_ROUNDS
    assert len(fake_messages.calls) == MAX_TOOL_ROUNDS + 1


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_ask_stream_ends_with_error_when_claude_stops_without_answer(
    client: TestClient, fake_messages: FakeStreamingMessages, stop_reason: str
) -> None:
    fake_messages.rounds = [StreamRound(["Hello ", "world"], stop_reason)]

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
