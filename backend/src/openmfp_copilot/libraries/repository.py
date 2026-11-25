from collections.abc import Iterable

from pydantic import HttpUrl

from openmfp_copilot.libraries.models import Library, LibraryKind, Organization

type LibraryKey = tuple[Organization, str]


class LibraryAlreadyExistsError(Exception):
    def __init__(self, organization: Organization, name: str) -> None:
        super().__init__(f"Library '{organization}/{name}' already exists")
        self.organization = organization
        self.name = name


class InMemoryLibraryRepository:
    def __init__(self, libraries: Iterable[Library] = ()) -> None:
        self._libraries: dict[LibraryKey, Library] = {
            (library.organization, library.name): library for library in libraries
        }

    def list(
        self,
        organization: Organization | None = None,
        kind: LibraryKind | None = None,
    ) -> list[Library]:
        return [
            library
            for library in self._libraries.values()
            if (organization is None or library.organization == organization)
            and (kind is None or library.kind == kind)
        ]

    def get(self, organization: Organization, name: str) -> Library | None:
        return self._libraries.get((organization, name))

    def add(self, library: Library) -> Library:
        key = (library.organization, library.name)
        if key in self._libraries:
            raise LibraryAlreadyExistsError(library.organization, library.name)
        self._libraries[key] = library
        return library


def _github_library(organization: Organization, name: str, kind: LibraryKind) -> Library:
    return Library(
        organization=organization,
        name=name,
        kind=kind,
        repository_url=HttpUrl(f"https://github.com/{organization}/{name}"),
    )


KNOWN_LIBRARIES: tuple[Library, ...] = (
    _github_library(Organization.OPENMFP, "portal-ui-lib", LibraryKind.UI),
    _github_library(Organization.OPENMFP, "portal-server-lib", LibraryKind.SERVER),
    _github_library(Organization.OPENMFP, "webcomponents", LibraryKind.UI),
    _github_library(Organization.OPENMFP, "create-micro-frontend", LibraryKind.CLI),
    _github_library(Organization.OPENMFP, "portal", LibraryKind.APP),
    _github_library(Organization.PLATFORM_MESH, "portal-ui-lib", LibraryKind.UI),
    _github_library(Organization.PLATFORM_MESH, "portal-server-lib", LibraryKind.SERVER),
    _github_library(Organization.PLATFORM_MESH, "portal", LibraryKind.APP),
)
