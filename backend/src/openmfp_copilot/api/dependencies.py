from collections.abc import AsyncIterator
from typing import Annotated, cast

from anthropic import AsyncAnthropic
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from openmfp_copilot.ask.service import AskService
from openmfp_copilot.ask.tools import ToolExecutor
from openmfp_copilot.config import Settings, get_settings
from openmfp_copilot.db.session import SessionFactory
from openmfp_copilot.github.client import GitHubClient
from openmfp_copilot.libraries.repository import InMemoryLibraryRepository


def get_library_repository(request: Request) -> InMemoryLibraryRepository:
    return cast(InMemoryLibraryRepository, request.app.state.library_repository)


def get_anthropic_client(request: Request) -> AsyncAnthropic:
    return cast(AsyncAnthropic, request.app.state.anthropic_client)


def get_github_client(request: Request) -> GitHubClient:
    return cast(GitHubClient, request.app.state.github_client)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    session_factory = cast(SessionFactory, request.app.state.session_factory)
    async with session_factory() as session:
        yield session


SettingsDep = Annotated[Settings, Depends(get_settings)]
LibraryRepositoryDep = Annotated[InMemoryLibraryRepository, Depends(get_library_repository)]
AnthropicClientDep = Annotated[AsyncAnthropic, Depends(get_anthropic_client)]
GitHubClientDep = Annotated[GitHubClient, Depends(get_github_client)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_ask_service(
    client: AnthropicClientDep,
    github: GitHubClientDep,
    settings: SettingsDep,
    repository: LibraryRepositoryDep,
) -> AskService:
    libraries = repository.list()
    return AskService(
        client=client,
        model=settings.anthropic_model,
        effort=settings.anthropic_effort,
        libraries=libraries,
        tools=ToolExecutor(github, libraries),
    )


AskServiceDep = Annotated[AskService, Depends(get_ask_service)]
