from importlib.metadata import version
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from openmfp_copilot.api.dependencies import SettingsDep

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app_name: str
    version: str
    environment: str


@router.get("/health")
def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=version("openmfp-copilot"),
        environment=settings.environment,
    )
