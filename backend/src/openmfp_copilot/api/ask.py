from collections.abc import AsyncIterator

import anthropic
from fastapi import APIRouter, HTTPException, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from openmfp_copilot.api.dependencies import AskServiceDep
from openmfp_copilot.ask.models import Answer, AskRequest, TextDelta, ToolCall
from openmfp_copilot.ask.service import AnswerUnavailableError

router = APIRouter(tags=["ask"])

API_FAILURE_DETAIL = "The Claude API request failed"


@router.post("/ask", responses={status.HTTP_502_BAD_GATEWAY: {"description": "No answer"}})
async def ask(request: AskRequest, service: AskServiceDep) -> Answer:
    try:
        return await service.ask(request.question)
    except AnswerUnavailableError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
    except anthropic.APIError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, API_FAILURE_DETAIL) from error


@router.post("/ask/stream", response_class=EventSourceResponse)
async def ask_stream(request: AskRequest, service: AskServiceDep) -> AsyncIterator[ServerSentEvent]:
    try:
        async for event in service.stream(request.question):
            match event:
                case TextDelta(text=text):
                    yield ServerSentEvent(event="delta", data={"text": text})
                case ToolCall(name=name, input=tool_input):
                    yield ServerSentEvent(
                        event="tool_call", data={"name": name, "input": tool_input}
                    )
    except AnswerUnavailableError as error:
        yield ServerSentEvent(event="error", data={"detail": str(error)})
        return
    except anthropic.APIError:
        yield ServerSentEvent(event="error", data={"detail": API_FAILURE_DETAIL})
        return
    yield ServerSentEvent(event="done", data={})
