from pathlib import Path

from openmfp_copilot.evals.retrieval import (
    DEFAULT_QUESTIONS,
    CaseResult,
    EvalCase,
    covered,
    format_report,
    heading_path,
    load_cases,
    rank_expected,
    smallest_complete_k,
)
from openmfp_copilot.libraries.repository import KNOWN_LIBRARIES

BACKEND_DIR = Path(__file__).resolve().parent.parent
GENERIC_UI = EvalCase(
    "What is the generic UI feature?",
    ("mesh/ui > Generic UI Feature", "mesh/ui > Generic UI Feature > Angular Configuration"),
)
LICENSE = EvalCase("What license?", ("openmfp/portal > Licensing",))
RANKING = [
    "mesh/ui > Generic UI Feature",
    "openmfp/portal > Licensing",
    "mesh/ui > Contributing",
    "mesh/ui > Generic UI Feature > Angular Configuration",
]


def test_heading_path_is_the_first_line_of_a_chunk() -> None:
    assert heading_path("openmfp/portal > Licensing\n\nApache-2.0") == "openmfp/portal > Licensing"


def test_rank_expected_records_the_position_of_each_expected_chunk() -> None:
    result = rank_expected(GENERIC_UI, RANKING)

    assert result.ranks == {
        "mesh/ui > Generic UI Feature": 1,
        "mesh/ui > Generic UI Feature > Angular Configuration": 4,
    }
    assert result.complete_at == 4
    assert result.missing == []


def test_rank_expected_marks_labels_that_match_no_chunk_as_missing() -> None:
    stale = EvalCase("Stale?", ("mesh/ui > Generic UI Feature", "mesh/ui > Removed section"))

    result = rank_expected(stale, RANKING)

    assert result.missing == ["mesh/ui > Removed section"]
    assert result.complete_at is None


def test_coverage_counts_questions_complete_within_top_k() -> None:
    results = [rank_expected(GENERIC_UI, RANKING), rank_expected(LICENSE, RANKING)]

    assert [covered(results, k) for k in (1, 2, 3, 4)] == [0, 1, 1, 2]
    assert smallest_complete_k(results) == 4


def test_smallest_complete_k_is_none_when_a_label_is_missing() -> None:
    results = [
        rank_expected(LICENSE, RANKING),
        CaseResult(EvalCase("Gone?", ("x",)), {"x": None}),
    ]

    assert smallest_complete_k(results) is None


def test_report_lists_ranks_and_coverage() -> None:
    report = format_report([rank_expected(GENERIC_UI, RANKING), rank_expected(LICENSE, RANKING)])

    assert "  1. complete at   4  ranks [1, 4]  What is the generic UI feature?" in report
    assert "top  2:   1/2 questions complete" in report
    assert report.endswith("Smallest k with every question complete: 4")


def test_labeled_questions_file_is_valid() -> None:
    cases = load_cases(BACKEND_DIR / DEFAULT_QUESTIONS)

    assert len(cases) == 26
    assert all(case.question and case.expected for case in cases)
    assert len({case.question for case in cases}) == len(cases)
    repositories = {f"{library.organization}/{library.name}" for library in KNOWN_LIBRARIES}
    assert all(
        heading.split(" > ", 1)[0] in repositories for case in cases for heading in case.expected
    )
