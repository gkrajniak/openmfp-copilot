from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from openmfp_copilot.libraries.models import Organization


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Answer(BaseModel):
    answer: str = Field(description="Concise answer to the developer's question.")
    organization: Organization | None = Field(
        description="Project the question is about, or null if it is unclear or about both."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="How sure you are; low when the answer depends on details you have not seen."
    )


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, object]


type StreamEvent = TextDelta | ToolCall
