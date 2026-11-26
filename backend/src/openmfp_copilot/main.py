from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openmfp_copilot.api import ask, health, libraries
from openmfp_copilot.config import Settings, get_settings
from openmfp_copilot.github.client import GitHubClient
from openmfp_copilot.libraries.repository import (
    KNOWN_LIBRARIES,
    InMemoryLibraryRepository,
)


def create_anthropic_client(settings: Settings) -> AsyncAnthropic:
    if settings.anthropic_api_key is None:
        return AsyncAnthropic()
    return AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with (
            create_anthropic_client(settings) as anthropic_client,
            httpx2.AsyncClient() as github_http,
        ):
            app.state.anthropic_client = anthropic_client
            app.state.github_client = GitHubClient(github_http)
            yield

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.library_repository = InMemoryLibraryRepository(KNOWN_LIBRARIES)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(libraries.router)
    app.include_router(ask.router)
    return app


app = create_app()
