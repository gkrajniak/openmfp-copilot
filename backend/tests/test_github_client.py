from dataclasses import dataclass, field

import httpx2
import pytest

from openmfp_copilot.github.client import GitHubClient, ReadmeNotFoundError

pytestmark = pytest.mark.anyio


@dataclass
class FakeGitHub:
    response: httpx2.Response
    requests: list[httpx2.Request] = field(default_factory=list)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.response

    async def get_readme(self, owner: str = "openmfp", repo: str = "portal-ui-lib") -> str:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(self)) as http:
            return await GitHubClient(http).get_readme(owner, repo)


async def test_get_readme_returns_raw_markdown() -> None:
    github = FakeGitHub(httpx2.Response(200, text="# Portal UI Lib\n"))

    assert await github.get_readme() == "# Portal UI Lib\n"


async def test_get_readme_requests_raw_readme_from_github_api() -> None:
    github = FakeGitHub(httpx2.Response(200, text="# Portal\n"))

    await github.get_readme(owner="platform-mesh", repo="portal")

    [request] = github.requests
    assert request.method == "GET"
    assert str(request.url) == "https://api.github.com/repos/platform-mesh/portal/readme"
    assert request.headers["Accept"] == "application/vnd.github.raw+json"
    assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert request.headers["User-Agent"] == "openmfp-copilot"


async def test_get_readme_raises_not_found_for_missing_readme() -> None:
    github = FakeGitHub(httpx2.Response(404))

    with pytest.raises(ReadmeNotFoundError) as error:
        await github.get_readme(repo="does-not-exist")

    assert error.value.owner == "openmfp"
    assert error.value.repo == "does-not-exist"


async def test_get_readme_raises_on_rate_limit() -> None:
    github = FakeGitHub(httpx2.Response(403, headers={"X-RateLimit-Remaining": "0"}))

    with pytest.raises(httpx2.HTTPStatusError) as error:
        await github.get_readme()

    assert error.value.response.status_code == 403
