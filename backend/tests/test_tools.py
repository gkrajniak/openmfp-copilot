from dataclasses import dataclass, field

import httpx2
import pytest
from anthropic.types.beta import BetaToolUseBlock

from openmfp_copilot.ask.tools import ToolExecutor
from openmfp_copilot.github.client import GitHubClient
from openmfp_copilot.libraries.repository import KNOWN_LIBRARIES

pytestmark = pytest.mark.anyio


@dataclass
class FakeGitHub:
    response: httpx2.Response = field(default_factory=lambda: httpx2.Response(200, text="# Readme"))
    requests: list[httpx2.Request] = field(default_factory=list)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.response

    def executor(self) -> ToolExecutor:
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(self))
        return ToolExecutor(GitHubClient(http), KNOWN_LIBRARIES)


def get_readme_call(organization: str, name: str) -> BetaToolUseBlock:
    return BetaToolUseBlock(
        id="toolu_1",
        name="get_readme",
        input={"organization": organization, "name": name},
        type="tool_use",
    )


def test_get_readme_definition_is_strict_and_limited_to_known_repositories() -> None:
    [definition] = FakeGitHub().executor().definitions

    assert definition["name"] == "get_readme"
    assert definition.get("strict") is True
    schema = definition["input_schema"]
    assert schema.get("additionalProperties") is False
    assert schema.get("required") == ["organization", "name"]
    assert schema.get("properties") == {
        "organization": {
            "type": "string",
            "enum": ["openmfp", "platform-mesh"],
            "description": "GitHub organization that owns the repository.",
        },
        "name": {
            "type": "string",
            "enum": [
                "create-micro-frontend",
                "portal",
                "portal-server-lib",
                "portal-ui-lib",
                "webcomponents",
            ],
            "description": "Repository name, e.g. portal-ui-lib.",
        },
    }


async def test_get_readme_returns_readme_as_tool_result() -> None:
    github = FakeGitHub(httpx2.Response(200, text="# Portal UI Library\n"))

    result = await github.executor().run(get_readme_call("openmfp", "portal-ui-lib"))

    assert result == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "# Portal UI Library\n",
    }
    [request] = github.requests
    assert request.url.path == "/repos/openmfp/portal-ui-lib/readme"


async def test_get_readme_rejects_repository_outside_scope_without_calling_github() -> None:
    github = FakeGitHub()

    result = await github.executor().run(get_readme_call("platform-mesh", "webcomponents"))

    assert result == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "'platform-mesh/webcomponents' is not a repository the copilot covers",
        "is_error": True,
    }
    assert github.requests == []


async def test_get_readme_reports_missing_readme_as_error() -> None:
    github = FakeGitHub(httpx2.Response(404))

    result = await github.executor().run(get_readme_call("openmfp", "portal"))

    assert result.get("is_error") is True
    assert result.get("content") == "Repository 'openmfp/portal' has no README or does not exist"


async def test_get_readme_reports_github_failure_as_error() -> None:
    github = FakeGitHub(httpx2.Response(403, headers={"X-RateLimit-Remaining": "0"}))

    result = await github.executor().run(get_readme_call("openmfp", "portal"))

    assert result.get("is_error") is True
    assert result.get("content") == "GitHub returned status 403 for 'openmfp/portal'"


async def test_run_reports_invalid_input_as_error() -> None:
    result = await FakeGitHub().executor().run(get_readme_call("sap", "portal"))

    assert result.get("is_error") is True
    assert str(result.get("content")).startswith("Invalid input for get_readme")


async def test_run_reports_unknown_tool_as_error() -> None:
    call = BetaToolUseBlock(id="toolu_2", name="delete_repo", input={}, type="tool_use")

    result = await FakeGitHub().executor().run(call)

    assert result == {
        "type": "tool_result",
        "tool_use_id": "toolu_2",
        "content": "Unknown tool 'delete_repo'",
        "is_error": True,
    }
