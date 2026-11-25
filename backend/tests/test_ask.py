from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openmfp_copilot.api.dependencies import get_anthropic_client
from openmfp_copilot.ask.models import Answer
from openmfp_copilot.libraries.models import Organization

ANSWER = Answer(
    answer="Use portal-ui-lib to bootstrap the portal shell.",
    organization=Organization.OPENMFP,
    confidence="medium",
)


@dataclass
class FakeMessages:
    stop_reason: str = "end_turn"
    parsed_output: Answer | None = field(default_factory=lambda: ANSWER)
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(stop_reason=self.stop_reason, parsed_output=self.parsed_output)


@pytest.fixture
def fake_messages(app: FastAPI) -> FakeMessages:
    messages = FakeMessages()
    fake_client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    app.dependency_overrides[get_anthropic_client] = lambda: fake_client
    return messages


def test_ask_returns_structured_answer(client: TestClient, fake_messages: FakeMessages) -> None:
    response = client.post("/ask", json={"question": "How do I start a portal?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Use portal-ui-lib to bootstrap the portal shell.",
        "organization": "openmfp",
        "confidence": "medium",
    }


def test_ask_sends_question_with_configured_model_and_structured_output(
    client: TestClient, fake_messages: FakeMessages
) -> None:
    client.post("/ask", json={"question": "How do I start a portal?"})

    [call] = fake_messages.calls
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_format"] is Answer
    assert call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default"
    assert call["messages"] == [{"role": "user", "content": "How do I start a portal?"}]


def test_ask_lists_known_repositories_in_system_prompt(
    client: TestClient, fake_messages: FakeMessages
) -> None:
    client.post("/ask", json={"question": "Which libraries exist?"})

    system_prompt = fake_messages.calls[0]["system"]
    assert "openmfp/portal-ui-lib (ui)" in system_prompt
    assert "platform-mesh/portal (app)" in system_prompt


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_ask_returns_502_when_claude_stops_without_answer(
    client: TestClient, fake_messages: FakeMessages, stop_reason: str
) -> None:
    fake_messages.stop_reason = stop_reason

    response = client.post("/ask", json={"question": "How do I start a portal?"})

    assert response.status_code == 502
    assert response.json() == {"detail": f"Claude did not return an answer: {stop_reason}"}


def test_ask_returns_502_when_output_is_missing(
    client: TestClient, fake_messages: FakeMessages
) -> None:
    fake_messages.parsed_output = None

    response = client.post("/ask", json={"question": "How do I start a portal?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Claude did not return an answer: no structured output"}


def test_ask_returns_502_when_claude_api_fails(
    client: TestClient, fake_messages: FakeMessages
) -> None:
    fake_messages.error = anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    )

    response = client.post("/ask", json={"question": "How do I start a portal?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The Claude API request failed"}


def test_ask_rejects_empty_question_without_calling_claude(
    client: TestClient, fake_messages: FakeMessages
) -> None:
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422
    assert fake_messages.calls == []
