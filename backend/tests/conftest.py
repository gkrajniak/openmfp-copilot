import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import URL, make_url, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from openmfp_copilot.api.dependencies import get_github_client
from openmfp_copilot.config import Settings, get_settings
from openmfp_copilot.db.session import SessionFactory
from openmfp_copilot.github.client import GitHubClient
from openmfp_copilot.main import create_app

BACKEND_DIR = Path(__file__).resolve().parent.parent
TEST_DATABASE_URL = make_url(
    os.environ.get(
        "OPENMFP_COPILOT_TEST_DATABASE_URL",
        "postgresql+asyncpg://copilot:copilot@localhost:5432/openmfp_copilot_test",
    )
)


async def create_database_if_missing(url: URL) -> None:
    engine = create_async_engine(
        url.set(database="postgres"), isolation_level="AUTOCOMMIT", poolclass=NullPool
    )
    try:
        async with engine.connect() as connection:
            exists = await connection.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": url.database}
            )
            if not exists:
                await connection.execute(text(f'CREATE DATABASE "{url.database}"'))
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    try:
        asyncio.run(create_database_if_missing(TEST_DATABASE_URL))
    except OSError as error:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"PostgreSQL is not reachable ({error}); run `docker compose up -d --wait`")

    url = TEST_DATABASE_URL.render_as_string(hide_password=False)
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        env={**os.environ, "OPENMFP_COPILOT_DATABASE_URL": url},
        check=True,
        capture_output=True,
    )
    return url


@pytest.fixture
async def session_factory(test_database_url: str) -> AsyncIterator[SessionFactory]:
    engine = create_async_engine(test_database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        yield async_sessionmaker(
            bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
        await transaction.rollback()
    await engine.dispose()


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
