"""Deterministic generator for the sample policy-conflict cases (SPEC section 7).

Gold labels are written out per case function, not computed by rules, so the
rules engine can be checked against them. Only the document phrasing is random,
drawn from one seeded random.Random.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from pcd.schemas import (
    ANY,
    TOPICS,
    Authority,
    Case,
    Claim,
    ConflictLabel,
    Document,
    DocumentMetadata,
    Gold,
    GoldPair,
    Owner,
    Resolution,
    Route,
    Scope,
    Status,
)

MARKER = "SYNTHETIC TEST DOCUMENT"
STATE_NAMES = {"FL": "Florida", "TX": "Texas"}
TOPIC_TITLES = {
    "specialist_referral": "Specialist referral rule",
    "copay_primary_care": "Primary care copay",
    "appeal_filing_deadline_days": "Appeal filing deadline",
    "call_summary_retention_days": "Call summary retention",
    "callback_window_hours": "Callback window",
    "document_request_turnaround_days": "Document request turnaround",
    "interpreter_scheduling_lead_hours": "Interpreter scheduling lead time",
}
REFERRAL_TEMPLATES = {
    "required": (
        "Members must get a referral before seeing a specialist.",
        "A referral is required before a member can see a specialist.",
    ),
    "not_required": (
        "A referral is not required to see a specialist.",
        "Members may see specialists without a referral.",
    ),
}
NUMERIC_TEMPLATES = {
    "copay_primary_care": (
        "The copay for a primary care visit is ${n}.",
        "Members pay ${n} per primary care visit.",
    ),
    "appeal_filing_deadline_days": (
        "Members have {n} days to file an appeal.",
        "An appeal must be filed within {n} days of the decision.",
    ),
    "call_summary_retention_days": (
        "Call summaries are kept for {n} days.",
        "Recorded call summaries are retained for {n} days.",
    ),
    "document_request_turnaround_days": (
        "Document requests are completed within {n} days.",
        "Turnaround for a document request is {n} days.",
    ),
    "callback_window_hours": (
        "Callbacks are returned within {n} hours.",
        "A callback window of {n} hours applies.",
    ),
    "interpreter_scheduling_lead_hours": (
        "Interpreters must be booked at least {n} hours ahead.",
        "Interpreter bookings require {n} hours of lead time.",
    ),
}
SCOPE_PROSE = ("This section covers {parts}.", "The rule below applies to {parts}.")
FILLERS = (
    "Questions about this rule go to the policy owner named above.",
    "This text is a made-up example for testing the demo.",
)


@dataclass(frozen=True)
class DocSpec:
    """Inputs for one single-claim document; the text is rendered from these."""

    doc_id: str
    authority: Authority
    owner: Owner
    topic: str
    value: str
    scope: Scope
    start: date
    end: Optional[date] = None
    supersedes: Optional[str] = None
    variant: Optional[int] = None  # fixed phrasing index; None means seeded choice


def claim_id(spec: DocSpec) -> str:
    return f"{spec.doc_id}:{spec.topic}"


def _number(value: str) -> str:
    return value.partition(":")[2] if value.startswith("USD:") else value


def _value_sentence(spec: DocSpec, rng: random.Random) -> str:
    if spec.topic == "specialist_referral":
        options = REFERRAL_TEMPLATES[spec.value]
    else:
        options = NUMERIC_TEMPLATES[spec.topic]
    index = rng.randrange(len(options)) if spec.variant is None else spec.variant
    return options[index].format(n=_number(spec.value))


def _scope_parts(scope: Scope) -> list[str]:
    parts = []
    if scope.plan != ANY:
        parts.append(f"Plan {scope.plan}")
    if scope.state != ANY:
        parts.append(STATE_NAMES[scope.state])
    if scope.setting != ANY:
        parts.append(f"{scope.setting} setting")
    return parts


def _effective(spec: DocSpec) -> str:
    if spec.end is None:
        return spec.start.isoformat()
    return f"{spec.start.isoformat()} to {spec.end.isoformat()}"


def render_document(spec: DocSpec, rng: random.Random) -> tuple[Document, str]:
    """Render the document text; scope goes in an Applies-to line or in the prose, never both."""
    value_sentence = _value_sentence(spec, rng)
    parts = _scope_parts(spec.scope)
    header = [
        MARKER,
        f"Title: {TOPIC_TITLES[spec.topic]}",
        f"Owner: {spec.owner.value} | Type: {spec.authority.name} | "
        f"Effective: {_effective(spec)} | Supersedes: {spec.supersedes or 'none'}",
    ]
    sentences = [value_sentence]
    if parts and rng.random() < 0.5:
        header.append(f"Applies to: {', '.join(parts)}")
    elif parts:
        sentences.append(rng.choice(SCOPE_PROSE).format(parts=", ".join(parts)))
    sentences.append(rng.choice(FILLERS))
    text = "\n".join([*header, "", " ".join(sentences), MARKER])
    return Document(doc_id=spec.doc_id, title=TOPIC_TITLES[spec.topic], text=text), value_sentence


def _claim(spec: DocSpec, source_text: str) -> Claim:
    return Claim(
        claim_id=claim_id(spec),
        doc_id=spec.doc_id,
        topic=spec.topic,
        value=spec.value,
        scope=spec.scope,
        effective_start=spec.start,
        effective_end=spec.end,
        authority=spec.authority,
        owner=spec.owner,
        supersedes_doc_id=spec.supersedes,
        source_text=source_text,
    )


def _metadata(spec: DocSpec) -> DocumentMetadata:
    return DocumentMetadata(
        owner=spec.owner,
        authority=spec.authority,
        effective_start=spec.start,
        effective_end=spec.end,
        supersedes_doc_id=spec.supersedes,
    )


def _resolution(winner: DocSpec, loser: DocSpec, rule: str, disposition: Status) -> Resolution:
    return Resolution(
        winner_claim_id=claim_id(winner),
        loser_claim_id=claim_id(loser),
        rule_fired=rule,
        loser_disposition=disposition,
    )


def _pair(
    a: DocSpec,
    b: DocSpec,
    *,
    label: ConflictLabel,
    route: Route,
    scope_overlap: bool,
    date_overlap: bool,
    statuses: tuple[Status, Status],
    resolution: Optional[Resolution] = None,
) -> GoldPair:
    """Gold pair for doc a and doc b; a must be the doc whose claim id sorts first."""
    id_a, id_b = claim_id(a), claim_id(b)
    return GoldPair(
        claim_a_id=id_a,
        claim_b_id=id_b,
        label=label,
        route=route,
        risk=TOPICS[a.topic],
        scope_overlap=scope_overlap,
        date_overlap=date_overlap,
        resolution=resolution,
        final_status={id_a: statuses[0], id_b: statuses[1]},
    )


def _assemble(
    case_no: int,
    category: str,
    specs: tuple[DocSpec, DocSpec],
    pair: GoldPair,
    rng: random.Random,
) -> Case:
    rendered = [render_document(spec, rng) for spec in specs]
    claims = [_claim(spec, sentence) for spec, (_, sentence) in zip(specs, rendered)]
    gold = Gold(
        metadata={spec.doc_id: _metadata(spec) for spec in specs},
        claims=claims,
        pairs=[pair],
        final_status=dict(pair.final_status),
    )
    return Case(
        case_id=f"C{case_no:02d}",
        split="dev",
        category=category,
        documents=[doc for doc, _ in rendered],
        gold=gold,
    )


def case_01(rng: random.Random) -> Case:
    """High-risk contradiction; the contract wins on authority, and the clinic rule is scoped."""
    a = DocSpec(
        doc_id="DOC-01-A", authority=Authority.CONTRACT_EOC, owner=Owner.PAYER,
        topic="specialist_referral", value="not_required",
        scope=Scope(plan="NW-Gold", state="FL"), start=date(2026, 1, 1),
    )
    b = DocSpec(
        doc_id="DOC-01-B", authority=Authority.CLINIC_PROCEDURE, owner=Owner.CARE_DELIVERY,
        topic="specialist_referral", value="required",
        scope=Scope(state="FL", setting="clinic"), start=date(2026, 3, 1),
    )
    pair = _pair(
        a, b, label=ConflictLabel.TRUE_CONTRADICTION, route=Route.HUMAN,
        scope_overlap=True, date_overlap=True,
        statuses=(Status.CERTIFIED, Status.SCOPED),
        resolution=_resolution(a, b, "R2_AUTHORITY", Status.SCOPED),
    )
    return _assemble(1, "TRUE_CONTRADICTION high-risk", (a, b), pair, rng)


def case_02(rng: random.Random) -> Case:
    """Low-risk contradiction between two payer documents; the rule resolves it."""
    a = DocSpec(
        doc_id="DOC-02-A", authority=Authority.CORPORATE_POLICY, owner=Owner.PAYER,
        topic="call_summary_retention_days", value="90", scope=Scope(), start=date(2025, 7, 1),
    )
    b = DocSpec(
        doc_id="DOC-02-B", authority=Authority.SOP, owner=Owner.PAYER,
        topic="call_summary_retention_days", value="30", scope=Scope(), start=date(2025, 9, 1),
    )
    pair = _pair(
        a, b, label=ConflictLabel.TRUE_CONTRADICTION, route=Route.AUTO,
        scope_overlap=True, date_overlap=True,
        statuses=(Status.CERTIFIED, Status.WITHHELD),
        resolution=_resolution(a, b, "R2_AUTHORITY", Status.WITHHELD),
    )
    return _assemble(2, "TRUE_CONTRADICTION low-risk", (a, b), pair, rng)


def case_03(rng: random.Random) -> Case:
    """High-risk numeric contradiction; the plan-specific contract wins and the general policy is scoped."""
    a = DocSpec(
        doc_id="DOC-03-A", authority=Authority.CONTRACT_EOC, owner=Owner.PAYER,
        topic="copay_primary_care", value="USD:10", scope=Scope(plan="NW-Silver"),
        start=date(2026, 1, 1),
    )
    b = DocSpec(
        doc_id="DOC-03-B", authority=Authority.CORPORATE_POLICY, owner=Owner.PAYER,
        topic="copay_primary_care", value="USD:20", scope=Scope(), start=date(2026, 1, 1),
    )
    pair = _pair(
        a, b, label=ConflictLabel.TRUE_CONTRADICTION, route=Route.HUMAN,
        scope_overlap=True, date_overlap=True,
        statuses=(Status.CERTIFIED, Status.SCOPED),
        resolution=_resolution(a, b, "R2_AUTHORITY", Status.SCOPED),
    )
    return _assemble(3, "TRUE_CONTRADICTION high-risk numeric", (a, b), pair, rng)


def case_04(rng: random.Random) -> Case:
    """Same topic in different states; no scope overlap, so no conflict is raised."""
    a = DocSpec(
        doc_id="DOC-04-A", authority=Authority.SOP, owner=Owner.PAYER,
        topic="callback_window_hours", value="24", scope=Scope(state="FL"),
        start=date(2026, 1, 1),
    )
    b = DocSpec(
        doc_id="DOC-04-B", authority=Authority.SOP, owner=Owner.PAYER,
        topic="callback_window_hours", value="48", scope=Scope(state="TX"),
        start=date(2026, 1, 1),
    )
    pair = _pair(
        a, b, label=ConflictLabel.SCOPE_DIFFERENCE, route=Route.NONE,
        scope_overlap=False, date_overlap=True,
        statuses=(Status.CERTIFIED, Status.CERTIFIED),
    )
    return _assemble(4, "SCOPE_DIFFERENCE state", (a, b), pair, rng)


def case_05(rng: random.Random) -> Case:
    """Same plan, values in non-overlapping plan years; no conflict is raised."""
    a = DocSpec(
        doc_id="DOC-05-A", authority=Authority.CONTRACT_EOC, owner=Owner.PAYER,
        topic="appeal_filing_deadline_days", value="60", scope=Scope(plan="NW-Gold"),
        start=date(2025, 1, 1), end=date(2025, 12, 31),
    )
    b = DocSpec(
        doc_id="DOC-05-B", authority=Authority.CONTRACT_EOC, owner=Owner.PAYER,
        topic="appeal_filing_deadline_days", value="65", scope=Scope(plan="NW-Gold"),
        start=date(2026, 1, 1),
    )
    pair = _pair(
        a, b, label=ConflictLabel.SCOPE_DIFFERENCE, route=Route.NONE,
        scope_overlap=True, date_overlap=False,
        statuses=(Status.CERTIFIED, Status.CERTIFIED),
    )
    return _assemble(5, "SCOPE_DIFFERENCE plan year", (a, b), pair, rng)


def case_06(rng: random.Random) -> Case:
    """Different plans carry different copays; no scope overlap, so no conflict is raised."""
    a = DocSpec(
        doc_id="DOC-06-A", authority=Authority.CONTRACT_EOC, owner=Owner.PAYER,
        topic="copay_primary_care", value="USD:0", scope=Scope(plan="NW-Gold"),
        start=date(2026, 1, 1),
    )
    b = DocSpec(
        doc_id="DOC-06-B", authority=Authority.CONTRACT_EOC, owner=Owner.PAYER,
        topic="copay_primary_care", value="USD:10", scope=Scope(plan="NW-Silver"),
        start=date(2026, 1, 1),
    )
    pair = _pair(
        a, b, label=ConflictLabel.SCOPE_DIFFERENCE, route=Route.NONE,
        scope_overlap=False, date_overlap=True,
        statuses=(Status.CERTIFIED, Status.CERTIFIED),
    )
    return _assemble(6, "SCOPE_DIFFERENCE plan", (a, b), pair, rng)


def case_07(rng: random.Random) -> Case:
    """Newer SOP explicitly supersedes the older one; the rules engine resolves it."""
    a = DocSpec(
        doc_id="DOC-07-A", authority=Authority.SOP, owner=Owner.PAYER,
        topic="document_request_turnaround_days", value="10", scope=Scope(),
        start=date(2025, 1, 1),
    )
    b = DocSpec(
        doc_id="DOC-07-B", authority=Authority.SOP, owner=Owner.PAYER,
        topic="document_request_turnaround_days", value="5", scope=Scope(),
        start=date(2026, 2, 1), supersedes="DOC-07-A",
    )
    pair = _pair(
        a, b, label=ConflictLabel.OUTDATED_VERSION, route=Route.AUTO,
        scope_overlap=True, date_overlap=True,
        statuses=(Status.WITHHELD, Status.CERTIFIED),
        resolution=_resolution(b, a, "R1_SUPERSEDES", Status.WITHHELD),
    )
    return _assemble(7, "OUTDATED_VERSION", (a, b), pair, rng)


def case_08(rng: random.Random) -> Case:
    """Cross-owner tie at equal authority; the more specific scope wins but the route is HUMAN."""
    a = DocSpec(
        doc_id="DOC-08-A", authority=Authority.CORPORATE_POLICY, owner=Owner.PAYER,
        topic="interpreter_scheduling_lead_hours", value="24", scope=Scope(),
        start=date(2026, 1, 1),
    )
    b = DocSpec(
        doc_id="DOC-08-B", authority=Authority.CORPORATE_POLICY, owner=Owner.CARE_DELIVERY,
        topic="interpreter_scheduling_lead_hours", value="48", scope=Scope(plan="NW-Gold"),
        start=date(2026, 1, 1),
    )
    pair = _pair(
        a, b, label=ConflictLabel.TRUE_CONTRADICTION, route=Route.HUMAN,
        scope_overlap=True, date_overlap=True,
        statuses=(Status.SCOPED, Status.CERTIFIED),
        resolution=_resolution(b, a, "R3_SCOPE", Status.SCOPED),
    )
    return _assemble(8, "TRUE_CONTRADICTION cross-owner", (a, b), pair, rng)


def case_09(rng: random.Random) -> Case:
    """Equal authority, scope and date with different values; no rule decides, so both are withheld."""
    a = DocSpec(
        doc_id="DOC-09-A", authority=Authority.SOP, owner=Owner.PAYER,
        topic="call_summary_retention_days", value="30", scope=Scope(), start=date(2026, 1, 1),
    )
    b = DocSpec(
        doc_id="DOC-09-B", authority=Authority.SOP, owner=Owner.PAYER,
        topic="call_summary_retention_days", value="45", scope=Scope(), start=date(2026, 1, 1),
    )
    pair = _pair(
        a, b, label=ConflictLabel.AMBIGUITY_TIE, route=Route.HUMAN,
        scope_overlap=True, date_overlap=True,
        statuses=(Status.WITHHELD, Status.WITHHELD),
    )
    return _assemble(9, "AMBIGUITY_TIE", (a, b), pair, rng)


def case_10(rng: random.Random) -> Case:
    """Same meaning worded differently at a high-risk topic; no conflict is raised."""
    a = DocSpec(
        doc_id="DOC-10-A", authority=Authority.CONTRACT_EOC, owner=Owner.PAYER,
        topic="specialist_referral", value="not_required",
        scope=Scope(plan="NW-Gold", state="FL"), start=date(2026, 1, 1), variant=0,
    )
    b = DocSpec(
        doc_id="DOC-10-B", authority=Authority.CORPORATE_POLICY, owner=Owner.PAYER,
        topic="specialist_referral", value="not_required",
        scope=Scope(state="FL"), start=date(2026, 1, 1), variant=1,
    )
    pair = _pair(
        a, b, label=ConflictLabel.NO_CONFLICT, route=Route.NONE,
        scope_overlap=True, date_overlap=True,
        statuses=(Status.CERTIFIED, Status.CERTIFIED),
    )
    return _assemble(10, "NO_CONFLICT paraphrase", (a, b), pair, rng)


CASE_BUILDERS: tuple[Callable[[random.Random], Case], ...] = (
    case_01, case_02, case_03, case_04, case_05,
    case_06, case_07, case_08, case_09, case_10,
)


def build_cases(seed: int) -> list[Case]:
    rng = random.Random(seed)
    return [build(rng) for build in CASE_BUILDERS]


def write_cases(cases: list[Case], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(case.model_dump_json() + "\n" for case in cases)
    path.write_text(body, encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Write the sample policy-conflict cases (a data folder, or .jsonl if --out ends in .jsonl).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    cases = build_cases(args.seed)
    if args.out.suffix == ".jsonl":
        write_cases(cases, args.out)
    else:
        from pcd.data.loader import write_case_dir
        write_case_dir(cases, args.out)


if __name__ == "__main__":
    main()
