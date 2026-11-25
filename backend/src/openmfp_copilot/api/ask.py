import anthropic
from fastapi import APIRouter, HTTPException, status

from openmfp_copilot.api.dependencies import AskServiceDep
from openmfp_copilot.ask.models import Answer, AskRequest
from openmfp_copilot.ask.service import AnswerUnavailableError

router = APIRouter(tags=["ask"])


@router.post("/ask", responses={status.HTTP_502_BAD_GATEWAY: {"description": "No answer"}})
async def ask(request: AskRequest, service: AskServiceDep) -> Answer:
    try:
        return await service.ask(request.question)
    except AnswerUnavailableError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
    except anthropic.APIError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The Claude API request failed") from error
