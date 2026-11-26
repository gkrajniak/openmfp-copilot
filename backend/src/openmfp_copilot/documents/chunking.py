import re
from dataclasses import dataclass

HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
HEADING_SEPARATOR = " > "


@dataclass(frozen=True)
class Section:
    headings: tuple[str, ...]
    body: str

    @property
    def text(self) -> str:
        return f"{HEADING_SEPARATOR.join(self.headings)}\n\n{self.body}"


@dataclass
class _Fence:
    marker: str

    def closes(self, line: str) -> bool:
        match = FENCE.match(line)
        return (
            match is not None
            and match.group(1)[0] == self.marker[0]
            and len(match.group(1)) >= len(self.marker)
            and not line[match.end() :].strip()
        )


def split_markdown(markdown: str, title: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []
    fence: _Fence | None = None

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            headings = (title, *(heading for _, heading in stack))
            sections.append(Section(headings=headings, body=body))
        lines.clear()

    for line in markdown.splitlines():
        if fence is not None:
            if fence.closes(line):
                fence = None
            lines.append(line)
            continue

        fence_match = FENCE.match(line)
        if fence_match is not None:
            fence = _Fence(fence_match.group(1))
            lines.append(line)
            continue

        heading_match = HEADING.match(line)
        if heading_match is None:
            lines.append(line)
            continue

        flush()
        level = len(heading_match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading_match.group(2)))

    flush()
    return sections
