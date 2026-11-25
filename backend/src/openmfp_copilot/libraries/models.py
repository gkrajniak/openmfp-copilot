from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Organization(StrEnum):
    OPENMFP = "openmfp"
    PLATFORM_MESH = "platform-mesh"


class LibraryKind(StrEnum):
    UI = "ui"
    SERVER = "server"
    CLI = "cli"
    APP = "app"


class Library(BaseModel):
    model_config = ConfigDict(frozen=True)

    organization: Organization
    name: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", max_length=100)
    kind: LibraryKind
    repository_url: HttpUrl
