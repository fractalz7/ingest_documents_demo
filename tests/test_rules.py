"""Unit tests for pcd/rules.py, plus rules-vs-gold on the sample file."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pcd import rules
from pcd.schemas import (
    ANY,
    NONCONFLICT_LABELS,
    Authority,
    Case,
    Checks,
    Claim,
    ConflictLabel,
    Owner,
    Risk,
    Route,
    Scope,
    Status,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data"


def claim(
    doc: str = "A",
    topic: str = "callback_window_hours",
    authority: Authority = Authority.SOP,
    owner: Owner = Owner.PAYER,
    scope: Scope | None = None,
    start: date = date(2026, 1, 1),
    end: date | None = None,
    supersedes: str | None = None,
) -> Claim:
    return Claim(
        claim_id=f"DOC-{doc}:{topic}",
        doc_id=f"DOC-{doc}",
        topic=topic,
        value="24",
        scope=scope or Scope(),
        effective_start=start,
        effective_end=end,
        authority=authority,
        owner=owner,
        supersedes_doc_id=supersedes,
    )


def test_scope_overlap_treats_any_as_wildcard():
    assert rules.scope_overlap(Scope(plan="NW-Gold"), Scope(state="FL"))
    assert rules.scope_overlap(Scope(), Scope(plan="NW-Gold", state="FL", setting="clinic"))
    assert not rules.scope_overlap(Scope(state="FL"), Scope(state="TX"))
    assert not rules.scope_overlap(Scope(plan="NW-Gold"), Scope(plan="NW-Silver"))


def test_date_overlap_is_inclusive_and_open_ended():
    closed = claim(start=date(2025, 1, 1), end=date(2025, 12, 31))
    assert rules.date_overlap(closed, claim(doc="B", start=date(2025, 12, 31)))
    assert not rules.date_overlap(closed, claim(doc="B", start=date(2026, 1, 1)))
    assert rules.date_overlap(claim(start=date(2020, 1, 1)), claim(doc="B", start=date(2030, 1, 1)))


def test_specificity_counts_non_any_dimensions():
    assert rules.specificity(Scope()) == 0
    assert rules.specificity(Scope(plan="NW-Gold", state="FL", setting="clinic")) == 3
    assert rules.specificity(Scope(state=ANY, setting="member")) == 1


def test_scope_covers_only_when_winner_is_at_least_as_wide():
    assert rules.scope_covers(Scope(), Scope(plan="NW-Gold"))
    assert rules.scope_covers(Scope(plan="NW-Gold"), Scope(plan="NW-Gold", state="FL"))
    assert not rules.scope_covers(Scope(plan="NW-Gold"), Scope())
    assert not rules.scope_covers(Scope(plan="NW-Gold"), Scope(plan="NW-Silver"))


def test_r1_supersedes_wins_in_either_argument_order():
    old = claim(doc="A")
    new = claim(doc="B", supersedes="DOC-A", start=date(2026, 2, 1))
    for a, b in ((old, new), (new, old)):
        res = rules.decide_resolution(a, b)
        assert res.rule_fired == "R1_SUPERSEDES"
        assert res.winner_claim_id == new.claim_id
        assert res.loser_claim_id == old.claim_id


def test_r2_higher_authority_wins_and_scoped_loser_is_scoped():
    contract = claim(doc="A", authority=Authority.CONTRACT_EOC, scope=Scope(plan="NW-Gold"))
    sop = claim(doc="B", authority=Authority.SOP)
    res = rules.decide_resolution(contract, sop)
    assert res.rule_fired == "R2_AUTHORITY"
    assert res.winner_claim_id == contract.claim_id
    assert res.loser_disposition == Status.SCOPED


def test_r2_wider_winner_withholds_loser():
    corporate = claim(doc="A", authority=Authority.CORPORATE_POLICY)
    sop = claim(doc="B", authority=Authority.SOP)
    res = rules.decide_resolution(sop, corporate)
    assert res.winner_claim_id == corporate.claim_id
    assert res.loser_disposition == Status.WITHHELD


def test_r3_more_specific_scope_wins_at_equal_authority():
    specific = claim(doc="A", scope=Scope(plan="NW-Gold"))
    general = claim(doc="B")
    res = rules.decide_resolution(general, specific)
    assert res.rule_fired == "R3_SCOPE"
    assert res.winner_claim_id == specific.claim_id
    assert res.loser_disposition == Status.SCOPED


def test_r4_later_start_wins_when_authority_and_scope_tie():
    earlier = claim(doc="A", start=date(2025, 1, 1))
    later = claim(doc="B", start=date(2026, 1, 1))
    res = rules.decide_resolution(earlier, later)
    assert res.rule_fired == "R4_DATE"
    assert res.winner_claim_id == later.claim_id
    assert res.loser_disposition == Status.WITHHELD


def test_tie_returns_none():
    assert rules.decide_resolution(claim(doc="A"), claim(doc="B")) is None


def test_checks_flag_cross_owner_tie_and_high_risk():
    a = claim(doc="A", authority=Authority.CORPORATE_POLICY, owner=Owner.PAYER,
              scope=Scope(plan="NW-Gold"))
    b = claim(doc="B", authority=Authority.CORPORATE_POLICY, owner=Owner.CARE_DELIVERY)
    res = rules.decide_resolution(a, b)
    checks = rules.compute_checks(a, b, res)
    assert checks.rule_applies is True
    assert checks.low_risk is True
    assert checks.no_cross_owner_tie is False

    high_a = claim(doc="A", topic="specialist_referral")
    high_b = claim(doc="B", topic="specialist_referral")
    assert rules.compute_checks(high_a, high_b, None).low_risk is False


def test_unknown_topic_fails_closed_as_high_risk():
    assert rules.risk_of("made_up_topic") == Risk.HIGH
    a = claim(doc="A", topic="made_up_topic", authority=Authority.CORPORATE_POLICY)
    b = claim(doc="B", topic="made_up_topic", authority=Authority.SOP)
    checks = rules.compute_checks(a, b, rules.decide_resolution(a, b))
    assert checks.low_risk is False


def test_route_conflict_is_auto_only_when_everything_passes():
    ok = Checks(rule_applies=True, low_risk=True, no_cross_owner_tie=True)
    assert rules.route_conflict(ok, inconsistent=False, verifier_disputed=False) == Route.AUTO
    assert rules.route_conflict(ok, inconsistent=True, verifier_disputed=False) == Route.HUMAN
    assert rules.route_conflict(ok, inconsistent=False, verifier_disputed=True) == Route.HUMAN
    for failed in (
        ok.model_copy(update={"rule_applies": False}),
        ok.model_copy(update={"low_risk": False}),
        ok.model_copy(update={"no_cross_owner_tie": False}),
    ):
        assert rules.route_conflict(failed, False, False) == Route.HUMAN


def test_label_consistency_rules():
    tc, nc = ConflictLabel.TRUE_CONTRADICTION, ConflictLabel.SCOPE_DIFFERENCE
    assert rules.label_is_consistent(tc, True, True, True)
    assert not rules.label_is_consistent(tc, True, True, False)
    assert not rules.label_is_consistent(nc, True, True, True)
    assert rules.label_is_consistent(nc, False, True, True)


def test_statuses_from_resolution():
    a, b = claim(doc="A"), claim(doc="B")
    assert rules.statuses_from_resolution(a, b, None) == {
        a.claim_id: Status.WITHHELD,
        b.claim_id: Status.WITHHELD,
    }
    contract = claim(doc="A", authority=Authority.CONTRACT_EOC, scope=Scope(plan="NW-Gold"))
    res = rules.decide_resolution(contract, b)
    assert rules.statuses_from_resolution(contract, b, res) == {
        contract.claim_id: Status.CERTIFIED,
        b.claim_id: Status.SCOPED,
    }


def _sample_cases() -> list[Case]:
    if not SAMPLE.exists():
        pytest.skip(f"{SAMPLE.name} not generated yet")
    from pcd.evals.run_evals import load_cases
    return load_cases(SAMPLE)


def test_rules_agree_with_gold_on_sample():
    cases = _sample_cases()
    assert cases, "sample file is empty"
    for case in cases:
        claims = {c.claim_id: c for c in case.gold.claims}
        for gp in case.gold.pairs:
            a, b = claims[gp.claim_a_id], claims[gp.claim_b_id]
            scope_o = rules.scope_overlap(a.scope, b.scope)
            date_o = rules.date_overlap(a, b)
            differ = a.value != b.value
            assert scope_o == gp.scope_overlap, (case.case_id, gp.claim_a_id)
            assert date_o == gp.date_overlap, (case.case_id, gp.claim_a_id)
            assert rules.label_is_consistent(gp.label, scope_o, date_o, differ), (
                case.case_id,
                "gold label contradicts computed facts",
            )

            if gp.label in NONCONFLICT_LABELS:
                # Non-conflicts are routed NONE before any resolution is computed.
                assert gp.route == Route.NONE
                assert gp.resolution is None
                assert all(s == Status.CERTIFIED for s in gp.final_status.values())
                continue

            resolution = rules.decide_resolution(a, b)
            assert resolution == gp.resolution, (case.case_id, gp.claim_a_id)
            checks = rules.compute_checks(a, b, resolution)
            assert rules.route_conflict(checks, inconsistent=False, verifier_disputed=False) == gp.route
            assert rules.statuses_from_resolution(a, b, resolution) == gp.final_status
