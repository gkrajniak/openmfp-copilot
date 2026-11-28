import argparse
import asyncio
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openmfp_copilot.config import Settings, get_settings
from openmfp_copilot.db.session import database_sessions
from openmfp_copilot.documents.embeddings import create_embedder
from openmfp_copilot.documents.search import DocumentSearch

DEFAULT_QUESTIONS = Path("evals/retrieval_questions.toml")


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected: tuple[str, ...]


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    ranks: dict[str, int | None]

    @property
    def missing(self) -> list[str]:
        return [heading for heading, rank in self.ranks.items() if rank is None]

    @property
    def complete_at(self) -> int | None:
        found = [rank for rank in self.ranks.values() if rank is not None]
        if self.missing or not found:
            return None
        return max(found)


def load_cases(path: Path) -> list[EvalCase]:
    with path.open("rb") as file:
        data = tomllib.load(file)
    return [
        EvalCase(question=entry["question"], expected=tuple(entry["expected"]))
        for entry in data["questions"]
    ]


def heading_path(content: str) -> str:
    return content.split("\n", 1)[0]


def rank_expected(case: EvalCase, ranked_headings: Sequence[str]) -> CaseResult:
    positions: dict[str, int] = {}
    for rank, heading in enumerate(ranked_headings, start=1):
        positions.setdefault(heading, rank)
    return CaseResult(case, {heading: positions.get(heading) for heading in case.expected})


def covered(results: Sequence[CaseResult], k: int) -> int:
    return sum(
        1 for result in results if result.complete_at is not None and result.complete_at <= k
    )


def smallest_complete_k(results: Sequence[CaseResult]) -> int | None:
    complete = [result.complete_at for result in results]
    if not complete or any(rank is None for rank in complete):
        return None
    return max(rank for rank in complete if rank is not None)


def format_report(results: Sequence[CaseResult]) -> str:
    lines = []
    for number, result in enumerate(results, start=1):
        ranks = ", ".join(
            "missing" if rank is None else str(rank) for rank in result.ranks.values()
        )
        complete = "-" if result.complete_at is None else str(result.complete_at)
        lines.append(
            f"{number:3d}. complete at {complete:>3}  ranks [{ranks}]  {result.case.question}"
        )
        lines.extend(f"       missing label: {heading}" for heading in result.missing)

    found = [result.complete_at for result in results if result.complete_at is not None]
    lines.append("")
    for k in range(1, max(found, default=0) + 1):
        lines.append(f"top {k:2d}: {covered(results, k):3d}/{len(results)} questions complete")
    smallest = smallest_complete_k(results)
    if smallest is None:
        lines.append("Not every question can be completed: fix the missing labels first.")
    else:
        lines.append(f"Smallest k with every question complete: {smallest}")
    return "\n".join(lines)


async def evaluate(settings: Settings, cases: Sequence[EvalCase]) -> list[CaseResult]:
    results = []
    async with database_sessions(settings) as session_factory:
        search = DocumentSearch(create_embedder(settings), session_factory)
        for case in cases:
            matches = await search.search(case.question, limit=None)
            results.append(rank_expected(case, [heading_path(m.content) for m in matches]))
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m openmfp_copilot.evals.retrieval",
        description="Rank every stored chunk for each labeled question and report "
        "how many questions get all their expected chunks within the top k.",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = asyncio.run(evaluate(get_settings(), load_cases(args.questions)))
    print(format_report(results))
    sys.exit(1 if any(result.missing for result in results) else 0)


if __name__ == "__main__":
    main()
