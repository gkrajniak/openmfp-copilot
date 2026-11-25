from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openmfp_copilot.config import Settings, get_settings
from openmfp_copilot.main import create_app


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
