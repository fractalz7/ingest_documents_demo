"""Deterministic rules engine: the only code that decides who wins a conflict.

No randomness and no backend calls. Unknown topics count as HIGH risk so that a
missing vocabulary entry fails closed and can never be auto-resolved.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from pcd.schemas import (
    ANY,
    NONCONFLICT_LABELS,
    TOPICS,
    Checks,
    Claim,
    ConflictLabel,
    Resolution,
    Risk,
    Route,
    Scope,
    Status,
)

CONFLICT_LABELS = frozenset(ConflictLabel) - NONCONFLICT_LABELS
_SCOPE_FIELDS = ("plan", "state", "setting")


def risk_of(topic: str) -> Risk:
    """Return the topic's risk; unknown topics are HIGH so they fail closed."""
    return TOPICS.get(topic, Risk.HIGH)


def scope_overlap(a: Scope, b: Scope) -> bool:
    """True when every scope dimension is equal or wildcarded on either side."""
    return all(
        _dim_overlaps(getattr(a, f), getattr(b, f)) for f in _SCOPE_FIELDS
    )


def _dim_overlaps(x: str, y: str) -> bool:
    return x == ANY or y == ANY or x == y


def date_overlap(a: Claim, b: Claim) -> bool:
    """Inclusive interval intersection; a missing end date means open-ended."""
    a_end = a.effective_end or date.max
    b_end = b.effective_end or date.max
    return a.effective_start <= b_end and b.effective_start <= a_end


def specificity(scope: Scope) -> int:
    """Number of scope dimensions that are not wildcarded."""
    return sum(getattr(scope, f) != ANY for f in _SCOPE_FIELDS)


def scope_covers(winner: Scope, loser: Scope) -> bool:
    """True when the winner applies everywhere the loser does."""
    return all(
        getattr(winner, f) == ANY or getattr(winner, f) == getattr(loser, f)
        for f in _SCOPE_FIELDS
    )


def decide_resolution(a: Claim, b: Claim) -> Optional[Resolution]:
    """Apply R1 to R4 in order; the first rule that distinguishes the claims wins.

    Returns None when no rule distinguishes them (a tie).
    """
    if a.supersedes_doc_id == b.doc_id:
        return _resolve(a, b, "R1_SUPERSEDES")
    if b.supersedes_doc_id == a.doc_id:
        return _resolve(b, a, "R1_SUPERSEDES")
    if a.authority != b.authority:
        winner, loser = (a, b) if a.authority > b.authority else (b, a)
        return _resolve(winner, loser, "R2_AUTHORITY")
    if specificity(a.scope) != specificity(b.scope):
        winner, loser = (
            (a, b) if specificity(a.scope) > specificity(b.scope) else (b, a)
        )
        return _resolve(winner, loser, "R3_SCOPE")
    if a.effective_start != b.effective_start:
        winner, loser = (a, b) if a.effective_start > b.effective_start else (b, a)
        return _resolve(winner, loser, "R4_DATE")
    return None


def _resolve(winner: Claim, loser: Claim, rule: str) -> Resolution:
    # A loser is only SCOPED when the winner's scope does not cover it; otherwise
    # the winner replaces it outright and it is WITHHELD.
    disposition = (
        Status.WITHHELD if scope_covers(winner.scope, loser.scope) else Status.SCOPED
    )
    return Resolution(
        winner_claim_id=winner.claim_id,
        loser_claim_id=loser.claim_id,
        rule_fired=rule,
        loser_disposition=disposition,
    )


def label_is_consistent(
    label: ConflictLabel, scope_o: bool, date_o: bool, values_differ: bool
) -> bool:
    """Check the agent's label against the computed facts.

    A conflict label needs all three facts; a non-conflict label must not have them all.
    """
    full = scope_o and date_o and values_differ
    return full if label in CONFLICT_LABELS else not full


def compute_checks(a: Claim, b: Claim, resolution: Optional[Resolution]) -> Checks:
    return Checks(
        rule_applies=resolution is not None,
        low_risk=risk_of(a.topic) == Risk.LOW and risk_of(b.topic) == Risk.LOW,
        no_cross_owner_tie=not (a.authority == b.authority and a.owner != b.owner),
    )


def route_conflict(checks: Checks, inconsistent: bool, verifier_disputed: bool) -> Route:
    """AUTO only when every check passes and nothing flags the pair."""
    ok = (
        checks.rule_applies
        and checks.low_risk
        and checks.no_cross_owner_tie
        and not inconsistent
        and not verifier_disputed
    )
    return Route.AUTO if ok else Route.HUMAN


def statuses_from_resolution(
    a: Claim, b: Claim, resolution: Optional[Resolution]
) -> dict[str, Status]:
    if resolution is None:
        return {a.claim_id: Status.WITHHELD, b.claim_id: Status.WITHHELD}
    return {
        resolution.winner_claim_id: Status.CERTIFIED,
        resolution.loser_claim_id: resolution.loser_disposition,
    }
