import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence

from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessageParam, BetaToolResultBlockParam, BetaToolUseBlock

from openmfp_copilot.ask.models import Answer, StreamEvent, TextDelta, ToolCall
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


def build_system_prompt(libraries: Iterable[Library]) -> str:
    repositories = "\n".join(
        f"- {library.organization}/{library.name} ({library.kind}): {library.repository_url}"
        for library in libraries
    )
    return (
        "You are OpenMFP Copilot, an assistant for developers who build micro frontend "
        "portals with the OpenMFP and Platform Mesh libraries.\n\n"
        f"Repositories you cover:\n{repositories}\n\n"
        "Before answering a question about one of these repositories, read its README "
        f"with the {GET_README_TOOL_NAME} tool and base the answer on it. If the README "
        "does not cover the question, say so and answer from what you already know. "
        "Do not invent function names, configuration keys or CLI flags; when the answer "
        "depends on details you have not seen, say so."
    )


def tool_uses_in(content: Sequence[object]) -> list[BetaToolUseBlock]:
    return [block for block in content if isinstance(block, BetaToolUseBlock)]


class AskService:
    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        effort: Effort,
        libraries: Iterable[Library],
        tools: ToolExecutor,
    ) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._tools = tools
        self._system_prompt = build_system_prompt(libraries)

    async def ask(self, question: str) -> Answer:
        messages: list[BetaMessageParam] = [{"role": "user", "content": question}]

        for tool_round in range(MAX_TOOL_ROUNDS + 1):
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
            if tool_round == MAX_TOOL_ROUNDS:
                break

            messages.append(response.to_param())
            tool_results = await self._run_tools(tool_uses_in(response.content))
            messages.append({"role": "user", "content": tool_results})

        raise AnswerUnavailableError("too many tool calls")

    async def stream(self, question: str) -> AsyncIterator[StreamEvent]:
        messages: list[BetaMessageParam] = [{"role": "user", "content": question}]

        for tool_round in range(MAX_TOOL_ROUNDS + 1):
            async with self._client.beta.messages.stream(
                model=self._model,
                max_tokens=STREAM_MAX_TOKENS,
                system=self._system_prompt,
                messages=messages,
                tools=self._tools.definitions,
                output_config={"effort": self._effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                async for text in stream.text_stream:
                    yield TextDelta(text)
                message = await stream.get_final_message()

            if message.stop_reason in ("refusal", "max_tokens"):
                raise AnswerUnavailableError(message.stop_reason)
            if message.stop_reason != "tool_use":
                return
            if tool_round == MAX_TOOL_ROUNDS:
                break

            tool_uses = tool_uses_in(message.content)
            for tool_use in tool_uses:
                yield ToolCall(name=tool_use.name, input=tool_use.input)
            messages.append(message.to_param())
            messages.append({"role": "user", "content": await self._run_tools(tool_uses)})

        raise AnswerUnavailableError("too many tool calls")

    async def _run_tools(self, tool_uses: list[BetaToolUseBlock]) -> list[BetaToolResultBlockParam]:
        return list(await asyncio.gather(*(self._tools.run(block) for block in tool_uses)))
