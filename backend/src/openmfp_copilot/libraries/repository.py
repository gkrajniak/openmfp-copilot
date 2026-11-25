from collections.abc import Iterable

from pydantic import HttpUrl

from openmfp_copilot.libraries.models import Library, LibraryKind


class LibraryAlreadyExistsError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Library '{name}' already exists")
        self.name = name


class InMemoryLibraryRepository:
    def __init__(self, libraries: Iterable[Library] = ()) -> None:
        self._libraries: dict[str, Library] = {library.name: library for library in libraries}

    def list(self, kind: LibraryKind | None = None) -> list[Library]:
        return [
            library for library in self._libraries.values() if kind is None or library.kind == kind
        ]

    def get(self, name: str) -> Library | None:
        return self._libraries.get(name)

    def add(self, library: Library) -> Library:
        if library.name in self._libraries:
            raise LibraryAlreadyExistsError(library.name)
        self._libraries[library.name] = library
        return library


OPENMFP_LIBRARIES: tuple[Library, ...] = (
    Library(
        name="portal-ui-lib",
        kind=LibraryKind.UI,
        repository_url=HttpUrl("https://github.com/openmfp/portal-ui-lib"),
    ),
    Library(
        name="portal-server-lib",
        kind=LibraryKind.SERVER,
        repository_url=HttpUrl("https://github.com/openmfp/portal-server-lib"),
    ),
    Library(
        name="webcomponents",
        kind=LibraryKind.UI,
        repository_url=HttpUrl("https://github.com/openmfp/webcomponents"),
    ),
    Library(
        name="create-micro-frontend",
        kind=LibraryKind.CLI,
        repository_url=HttpUrl("https://github.com/openmfp/create-micro-frontend"),
    ),
)
