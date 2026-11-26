from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaToolUseBlock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openmfp_copilot.api.dependencies import get_anthropic_client, get_github_client
from openmfp_copilot.ask.models import Answer
from openmfp_copilot.ask.service import MAX_TOOL_ROUNDS
from openmfp_copilot.github.client import GitHubClient
from openmfp_copilot.libraries.models import Organization

ANSWER = Answer(
    answer="Use portal-ui-lib to bootstrap the portal shell.",
    organization=Organization.OPENMFP,
    confidence="medium",
)


def answer_response(
    stop_reason: str = "end_turn", parsed_output: Answer | None = ANSWER
) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed_output)


def tool_use_response(*tool_uses: BetaToolUseBlock) -> SimpleNamespace:
    content = list(tool_uses)
    return SimpleNamespace(
        stop_reason="tool_use",
        content=content,
        parsed_output=None,
        to_param=lambda: {"role": "assistant", "content": content},
    )


def get_readme_call(tool_use_id: str, organization: str, name: str) -> BetaToolUseBlock:
    return BetaToolUseBlock(
        id=tool_use_id,
        name="get_readme",
        input={"organization": organization, "name": name},
        type="tool_use",
    )


@dataclass
class FakeMessages:
    responses: list[SimpleNamespace] = field(default_factory=lambda: [answer_response()])
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if self.error is not None:
            raise self.error
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


@pytest.fixture
def fake_messages(app: FastAPI) -> FakeMessages:
    messages = FakeMessages()
    fake_client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    app.dependency_overrides[get_anthropic_client] = lambda: fake_client
    return messages


@pytest.fixture
def github_requests(app: FastAPI) -> list[httpx2.Request]:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        repo = request.url.path.split("/")[3]
        return httpx2.Response(200, text=f"# {repo} README")

    github = GitHubClient(httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    app.dependency_overrides[get_github_client] = lambda: github
    return requests


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


def test_ask_offers_get_readme_tool_and_tells_claude_to_use_it(
    client: TestClient, fake_messages: FakeMessages
) -> None:
    client.post("/ask", json={"question": "What is portal-ui-lib for?"})

    [call] = fake_messages.calls
    assert [tool["name"] for tool in call["tools"]] == ["get_readme"]
    assert "read its README with the get_readme tool" in call["system"]


def test_ask_sends_readme_back_to_claude_and_returns_final_answer(
    client: TestClient, fake_messages: FakeMessages, github_requests: list[httpx2.Request]
) -> None:
    readme_call = get_readme_call("toolu_1", "openmfp", "portal-ui-lib")
    fake_messages.responses = [tool_use_response(readme_call), answer_response()]

    response = client.post("/ask", json={"question": "What is portal-ui-lib for?"})

    assert response.status_code == 200
    assert response.json()["answer"] == ANSWER.answer
    [request] = github_requests
    assert request.url.path == "/repos/openmfp/portal-ui-lib/readme"
    _, second_call = fake_messages.calls
    assert second_call["messages"] == [
        {"role": "user", "content": "What is portal-ui-lib for?"},
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


def test_ask_returns_all_parallel_tool_results_in_one_message(
    client: TestClient, fake_messages: FakeMessages, github_requests: list[httpx2.Request]
) -> None:
    fake_messages.responses = [
        tool_use_response(
            get_readme_call("toolu_1", "openmfp", "portal-ui-lib"),
            get_readme_call("toolu_2", "platform-mesh", "portal-ui-lib"),
        ),
        answer_response(),
    ]

    client.post("/ask", json={"question": "How do the two portal-ui-libs differ?"})

    tool_results = fake_messages.calls[1]["messages"][2]
    assert tool_results["role"] == "user"
    assert [result["tool_use_id"] for result in tool_results["content"]] == [
        "toolu_1",
        "toolu_2",
    ]
    assert len(github_requests) == 2


def test_ask_returns_502_when_claude_keeps_calling_tools(
    client: TestClient, fake_messages: FakeMessages, github_requests: list[httpx2.Request]
) -> None:
    fake_messages.responses = [
        tool_use_response(get_readme_call("toolu_1", "openmfp", "portal-ui-lib"))
    ]

    response = client.post("/ask", json={"question": "What is portal-ui-lib for?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Claude did not return an answer: too many tool calls"}
    assert len(fake_messages.calls) == MAX_TOOL_ROUNDS + 1


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_ask_returns_502_when_claude_stops_without_answer(
    client: TestClient, fake_messages: FakeMessages, stop_reason: str
) -> None:
    fake_messages.responses = [answer_response(stop_reason=stop_reason, parsed_output=None)]

    response = client.post("/ask", json={"question": "How do I start a portal?"})

    assert response.status_code == 502
    assert response.json() == {"detail": f"Claude did not return an answer: {stop_reason}"}


def test_ask_returns_502_when_output_is_missing(
    client: TestClient, fake_messages: FakeMessages
) -> None:
    fake_messages.responses = [answer_response(parsed_output=None)]

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
