import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence

from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessageParam, BetaToolResultBlockParam, BetaToolUseBlock

from openmfp_copilot.ask.models import Answer
from openmfp_copilot.ask.tools import GET_README_TOOL_NAME, ToolExecutor
from openmfp_copilot.config import Effort
from openmfp_copilot.libraries.models import Library

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16_000
STREAM_MAX_TOKENS = 64_000
MAX_TOOL_ROUNDS = 5


class AnswerUnavailableError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(f"Claude did not return an answer: {reason}")
        self.reason = reason


def build_system_prompt(libraries: Iterable[Library], *, can_read_readmes: bool) -> str:
    repositories = "\n".join(
        f"- {library.organization}/{library.name} ({library.kind}): {library.repository_url}"
        for library in libraries
    )
    if can_read_readmes:
        sources = (
            "Before answering a question about one of these repositories, read its README "
            f"with the {GET_README_TOOL_NAME} tool and base the answer on it. If the README "
            "does not cover the question, say so and answer from what you already know."
        )
    else:
        sources = "You cannot read these repositories yet, so answer from what you already know."
    return (
        "You are OpenMFP Copilot, an assistant for developers who build micro frontend "
        "portals with the OpenMFP and Platform Mesh libraries.\n\n"
        f"Repositories you cover:\n{repositories}\n\n"
        f"{sources} Do not invent function names, configuration keys or CLI flags; when the "
        "answer depends on details you have not seen, say so."
    )


class AskService:
    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        effort: Effort,
        libraries: Iterable[Library],
        tools: ToolExecutor,
    ) -> None:
        libraries = list(libraries)
        self._client = client
        self._model = model
        self._effort = effort
        self._tools = tools
        self._system_prompt = build_system_prompt(libraries, can_read_readmes=True)
        self._stream_system_prompt = build_system_prompt(libraries, can_read_readmes=False)

    async def ask(self, question: str) -> Answer:
        messages: list[BetaMessageParam] = [{"role": "user", "content": question}]

        for _ in range(MAX_TOOL_ROUNDS + 1):
            response = await self._client.beta.messages.parse(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=self._system_prompt,
                messages=messages,
                tools=self._tools.definitions,
                output_format=Answer,
                output_config={"effort": self._effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )

            if response.stop_reason in ("refusal", "max_tokens"):
                raise AnswerUnavailableError(response.stop_reason)
            if response.stop_reason != "tool_use":
                if response.parsed_output is None:
                    raise AnswerUnavailableError("no structured output")
                return response.parsed_output

            messages.append(response.to_param())
            messages.append({"role": "user", "content": await self._run_tools(response.content)})

        raise AnswerUnavailableError("too many tool calls")

    async def _run_tools(self, content: Sequence[object]) -> list[BetaToolResultBlockParam]:
        tool_uses = [block for block in content if isinstance(block, BetaToolUseBlock)]
        return list(await asyncio.gather(*(self._tools.run(block) for block in tool_uses)))

    async def stream(self, question: str) -> AsyncIterator[str]:
        async with self._client.beta.messages.stream(
            model=self._model,
            max_tokens=STREAM_MAX_TOKENS,
            system=self._stream_system_prompt,
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
