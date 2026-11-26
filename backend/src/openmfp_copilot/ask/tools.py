from collections.abc import Iterable

import httpx2
from anthropic.types.beta import BetaToolParam, BetaToolResultBlockParam, BetaToolUseBlock
from pydantic import BaseModel, ValidationError

from openmfp_copilot.github.client import GitHubClient, ReadmeNotFoundError
from openmfp_copilot.libraries.models import Library, Organization

GET_README_TOOL_NAME = "get_readme"


class ToolError(Exception):
    pass


class ReadmeToolInput(BaseModel):
    organization: Organization
    name: str


def build_get_readme_tool(libraries: Iterable[Library]) -> BetaToolParam:
    libraries = list(libraries)
    return {
        "name": GET_README_TOOL_NAME,
        "description": (
            "Read the README of one repository the copilot covers, as markdown from GitHub. "
            "Use it before answering a question about a repository's purpose, setup, "
            "configuration or API. When a question involves several repositories, call it "
            "for each of them in parallel."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "organization": {
                    "type": "string",
                    "enum": sorted({library.organization.value for library in libraries}),
                    "description": "GitHub organization that owns the repository.",
                },
                "name": {
                    "type": "string",
                    "enum": sorted({library.name for library in libraries}),
                    "description": "Repository name, e.g. portal-ui-lib.",
                },
            },
            "required": ["organization", "name"],
            "additionalProperties": False,
        },
    }


class ToolExecutor:
    def __init__(self, github: GitHubClient, libraries: Iterable[Library]) -> None:
        libraries = list(libraries)
        self._github = github
        self._known_repositories = {(library.organization, library.name) for library in libraries}
        self.definitions: list[BetaToolParam] = [build_get_readme_tool(libraries)]

    async def run(self, tool_use: BetaToolUseBlock) -> BetaToolResultBlockParam:
        try:
            content = await self._dispatch(tool_use)
        except ToolError as error:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": str(error),
                "is_error": True,
            }
        return {"type": "tool_result", "tool_use_id": tool_use.id, "content": content}

    async def _dispatch(self, tool_use: BetaToolUseBlock) -> str:
        if tool_use.name != GET_README_TOOL_NAME:
            raise ToolError(f"Unknown tool '{tool_use.name}'")
        return await self._get_readme(tool_use.input)

    async def _get_readme(self, raw_input: object) -> str:
        try:
            tool_input = ReadmeToolInput.model_validate(raw_input)
        except ValidationError as error:
            raise ToolError(f"Invalid input for {GET_README_TOOL_NAME}: {error}") from error

        organization, name = tool_input.organization, tool_input.name
        if (organization, name) not in self._known_repositories:
            raise ToolError(f"'{organization}/{name}' is not a repository the copilot covers")

        try:
            return await self._github.get_readme(organization, name)
        except ReadmeNotFoundError as error:
            raise ToolError(str(error)) from error
        except httpx2.HTTPStatusError as error:
            raise ToolError(
                f"GitHub returned status {error.response.status_code} for '{organization}/{name}'"
            ) from error
        except httpx2.HTTPError as error:
            raise ToolError(f"GitHub request for '{organization}/{name}' failed") from error
