"""Checks on the sample cases and generator (SPEC section 9, item 7; gold values from section 7)."""
from __future__ import annotations

from pathlib import Path

import pytest

from pcd.data.generate_cases import build_cases
from pcd.data.loader import write_case_dir
from pcd.evals.run_evals import load_cases
from pcd.schemas import Case, ConflictLabel, Route, Status

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data"
MARKER = "SYNTHETIC TEST DOCUMENT"

C = Status.CERTIFIED
S = Status.SCOPED
W = Status.WITHHELD
TC = ConflictLabel.TRUE_CONTRADICTION
SD = ConflictLabel.SCOPE_DIFFERENCE

# case id -> (gold label, gold route, final status by doc letter), from the SPEC section 7 table
EXPECTED = {
    "C01": (TC, Route.HUMAN, {"A": C, "B": S}),
    "C02": (TC, Route.AUTO, {"A": C, "B": W}),
    "C03": (TC, Route.HUMAN, {"A": C, "B": S}),
    "C04": (SD, Route.NONE, {"A": C, "B": C}),
    "C05": (SD, Route.NONE, {"A": C, "B": C}),
    "C06": (SD, Route.NONE, {"A": C, "B": C}),
    "C07": (ConflictLabel.OUTDATED_VERSION, Route.AUTO, {"A": W, "B": C}),
    "C08": (TC, Route.HUMAN, {"A": S, "B": C}),
    "C09": (ConflictLabel.AMBIGUITY_TIE, Route.HUMAN, {"A": W, "B": W}),
    "C10": (ConflictLabel.NO_CONFLICT, Route.NONE, {"A": C, "B": C}),
}


@pytest.fixture(scope="module")
def cases() -> list[Case]:
    return load_cases(SAMPLE)


def test_sample_has_exactly_ten_valid_cases(cases):
    assert [c.case_id for c in cases] == [f"C{n:02d}" for n in range(1, 11)]
    assert all(c.split == "dev" for c in cases)


def test_every_document_carries_the_synthetic_marker(cases):
    for case in cases:
        for doc in case.documents:
            assert MARKER in doc.text, doc.doc_id


def test_claim_ids_are_doc_id_and_topic(cases):
    for case in cases:
        for claim in case.gold.claims:
            assert claim.claim_id == f"{claim.doc_id}:{claim.topic}"


def test_at_least_40_percent_of_documents_state_scope_only_in_prose(cases):
    docs = [doc for case in cases for doc in case.documents]
    prose_only = [doc for doc in docs if "Applies to:" not in doc.text]
    assert len(prose_only) / len(docs) >= 0.4


def test_generator_output_is_deterministic_for_a_fixed_seed(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_case_dir(build_cases(7), first)
    write_case_dir(build_cases(7), second)
    names = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert names == sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    for n in names:
        assert (first / n).read_bytes() == (second / n).read_bytes() == (SAMPLE / n).read_bytes()
    assert load_cases(first) == build_cases(7)


@pytest.mark.parametrize("case_id", list(EXPECTED))
def test_gold_matches_spec_table(cases, case_id):
    label, route, status_by_letter = EXPECTED[case_id]
    case = next(c for c in cases if c.case_id == case_id)
    (pair,) = case.gold.pairs
    assert pair.label == label
    assert pair.route == route
    actual = {claim.doc_id[-1]: case.gold.final_status[claim.claim_id] for claim in case.gold.claims}
    assert actual == status_by_letter
