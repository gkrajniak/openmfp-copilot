from typing import Annotated, cast

from anthropic import AsyncAnthropic
from fastapi import Depends, Request

from openmfp_copilot.ask.service import AskService
from openmfp_copilot.config import Settings, get_settings
from openmfp_copilot.libraries.repository import InMemoryLibraryRepository


def get_library_repository(request: Request) -> InMemoryLibraryRepository:
    return cast(InMemoryLibraryRepository, request.app.state.library_repository)


def get_anthropic_client(request: Request) -> AsyncAnthropic:
    return cast(AsyncAnthropic, request.app.state.anthropic_client)


SettingsDep = Annotated[Settings, Depends(get_settings)]
LibraryRepositoryDep = Annotated[InMemoryLibraryRepository, Depends(get_library_repository)]
AnthropicClientDep = Annotated[AsyncAnthropic, Depends(get_anthropic_client)]


def get_ask_service(
    client: AnthropicClientDep, settings: SettingsDep, repository: LibraryRepositoryDep
) -> AskService:
    return AskService(
        client=client,
        model=settings.anthropic_model,
        effort=settings.anthropic_effort,
        libraries=repository.list(),
    )


AskServiceDep = Annotated[AskService, Depends(get_ask_service)]
