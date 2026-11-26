from anthropic.types.beta import BetaToolUseBlock


def get_readme_call(organization: str, name: str, tool_use_id: str = "toolu_1") -> BetaToolUseBlock:
    return BetaToolUseBlock(
        id=tool_use_id,
        name="get_readme",
        input={"organization": organization, "name": name},
        type="tool_use",
    )
