from collections.abc import AsyncIterator, Iterable

from anthropic import AsyncAnthropic

from openmfp_copilot.ask.models import Answer
from openmfp_copilot.config import Effort
from openmfp_copilot.libraries.models import Library

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16_000
STREAM_MAX_TOKENS = 64_000


class AnswerUnavailableError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(f"Claude did not return an answer: {reason}")
        self.reason = reason


def build_system_prompt(libraries: Iterable[Library]) -> str:
    repositories = "\n".join(
        f"- {library.organization}/{library.name} ({library.kind}): {library.repository_url}"
        for library in libraries
    )
    return (
        "You are OpenMFP Copilot, an assistant for developers who build micro frontend "
        "portals with the OpenMFP and Platform Mesh libraries.\n\n"
        f"Repositories you cover:\n{repositories}\n\n"
        "You cannot read these repositories yet, so answer from what you already know. "
        "Do not invent function names, configuration keys or CLI flags; when the answer "
        "depends on details you have not seen, say so."
    )


class AskService:
    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        effort: Effort,
        libraries: Iterable[Library],
    ) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._system_prompt = build_system_prompt(libraries)

    async def ask(self, question: str) -> Answer:
        response = await self._client.beta.messages.parse(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=self._system_prompt,
            messages=[{"role": "user", "content": question}],
            output_format=Answer,
            output_config={"effort": self._effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )

        if response.stop_reason in ("refusal", "max_tokens"):
            raise AnswerUnavailableError(response.stop_reason)
        if response.parsed_output is None:
            raise AnswerUnavailableError("no structured output")
        return response.parsed_output

    async def stream(self, question: str) -> AsyncIterator[str]:
        async with self._client.beta.messages.stream(
            model=self._model,
            max_tokens=STREAM_MAX_TOKENS,
            system=self._system_prompt,
            messages=[{"role": "user", "content": question}],
            output_config={"effort": self._effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            async for text in stream.text_stream:
                yield text
            message = await stream.get_final_message()

        if message.stop_reason in ("refusal", "max_tokens"):
            raise AnswerUnavailableError(message.stop_reason)
