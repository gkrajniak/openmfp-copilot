from collections.abc import Iterator

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openmfp_copilot.api.dependencies import get_github_client
from openmfp_copilot.config import Settings, get_settings
from openmfp_copilot.github.client import GitHubClient
from openmfp_copilot.main import create_app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings() -> Settings:
    return Settings(environment="test", _env_file=None)


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


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
