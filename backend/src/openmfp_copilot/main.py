from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openmfp_copilot.api import health, libraries
from openmfp_copilot.config import Settings, get_settings
from openmfp_copilot.libraries.repository import (
    KNOWN_LIBRARIES,
    InMemoryLibraryRepository,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(title=settings.app_name)
    app.state.library_repository = InMemoryLibraryRepository(KNOWN_LIBRARIES)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(libraries.router)
    return app


app = create_app()
