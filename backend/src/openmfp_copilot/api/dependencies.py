from typing import Annotated, cast

from fastapi import Depends, Request

from openmfp_copilot.config import Settings, get_settings
from openmfp_copilot.libraries.repository import InMemoryLibraryRepository


def get_library_repository(request: Request) -> InMemoryLibraryRepository:
    return cast(InMemoryLibraryRepository, request.app.state.library_repository)


SettingsDep = Annotated[Settings, Depends(get_settings)]
LibraryRepositoryDep = Annotated[InMemoryLibraryRepository, Depends(get_library_repository)]
