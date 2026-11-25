from fastapi import APIRouter, HTTPException, status

from openmfp_copilot.api.dependencies import LibraryRepositoryDep
from openmfp_copilot.libraries.models import Library, LibraryKind
from openmfp_copilot.libraries.repository import LibraryAlreadyExistsError

router = APIRouter(prefix="/libraries", tags=["libraries"])


@router.get("")
def list_libraries(
    repository: LibraryRepositoryDep, kind: LibraryKind | None = None
) -> list[Library]:
    return repository.list(kind)


@router.get("/{name}", responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}})
def get_library(name: str, repository: LibraryRepositoryDep) -> Library:
    library = repository.get(name)
    if library is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Library '{name}' not found")
    return library


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_409_CONFLICT: {"description": "Already exists"}},
)
def create_library(library: Library, repository: LibraryRepositoryDep) -> Library:
    try:
        return repository.add(library)
    except LibraryAlreadyExistsError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
