"""Ingestion pipeline: agents propose and verify, deterministic code decides.

Backend output (labels, verdicts, extracted values) only feeds the rules; it never
sets a route directly. Routing goes through rules.route_conflict.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from pydantic import BaseModel

from pcd import rules
from pcd.backends.base import Backend, Reviewer
from pcd.schemas import (
    NONCONFLICT_LABELS,
    Case,
    Claim,
    PairJudgement,
    PairResult,
    PipelineResult,
    Route,
    Status,
    Usage,
)

_STATUS_RANK = {Status.CERTIFIED: 0, Status.SCOPED: 1, Status.WITHHELD: 2}


class PipelineConfig(BaseModel):
    use_verifier: bool = True


def run_case(
    case: Case,
    backend: Backend,
    reviewer: Reviewer,
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """Run one case end to end. The usage field holds only this case's calls."""
    config = config if config is not None else PipelineConfig()
    usage_before = backend.usage

    metadata = {}
    blocked_docs = []
    claims: list[Claim] = []
    for doc in case.documents:
        meta = backend.classify(doc)
        if meta is None:
            blocked_docs.append(doc.doc_id)
            continue
        metadata[doc.doc_id] = meta
        claims.extend(backend.extract(doc, meta))

    pairs = [
        _judge_pair(case.case_id, a, b, backend, reviewer, config)
        for a, b in candidate_pairs(claims)
    ]
    return PipelineResult(
        case_id=case.case_id,
        blocked_docs=blocked_docs,
        metadata=metadata,
        claims=claims,
        pairs=pairs,
        final_status=aggregate_status(claims, pairs),
        usage=_usage_delta(usage_before, backend.usage),
    )


def candidate_pairs(claims: Sequence[Claim]) -> list[tuple[Claim, Claim]]:
    """Cross-document pairs on the same topic, ordered by claim id."""
    pairs = []
    for i, x in enumerate(claims):
        for y in claims[i + 1 :]:
            if x.doc_id != y.doc_id and x.topic == y.topic:
                a, b = sorted((x, y), key=lambda c: c.claim_id)
                pairs.append((a, b))
    return sorted(pairs, key=lambda p: (p[0].claim_id, p[1].claim_id))


def aggregate_status(
    claims: Sequence[Claim], pairs: Sequence[PairResult]
) -> dict[str, Status]:
    """Most restrictive status across a claim's pairs; claims in no pair stay CERTIFIED."""
    final = {c.claim_id: Status.CERTIFIED for c in claims}
    for pair in pairs:
        for claim_id, status in pair.final_status.items():
            if _STATUS_RANK[status] > _STATUS_RANK[final[claim_id]]:
                final[claim_id] = status
    return final


def _judge_pair(
    case_id: str,
    a: Claim,
    b: Claim,
    backend: Backend,
    reviewer: Reviewer,
    config: PipelineConfig,
) -> PairResult:
    judgement: PairJudgement = backend.detect(a, b)
    disputed = False
    if config.use_verifier:
        disputed = not backend.verify(a, b, judgement).agrees

    scope_o = rules.scope_overlap(a.scope, b.scope)
    date_o = rules.date_overlap(a, b)
    differ = a.value != b.value
    consistent = rules.label_is_consistent(judgement.label, scope_o, date_o, differ)
    fields = dict(
        claim_a_id=a.claim_id,
        claim_b_id=b.claim_id,
        predicted_label=judgement.label,
        scope_overlap=scope_o,
        date_overlap=date_o,
        values_differ=differ,
        inconsistent=not consistent,
        verifier_disputed=disputed,
    )

    if judgement.label in NONCONFLICT_LABELS and consistent and not disputed:
        return PairResult(
            **fields,
            route=Route.NONE,
            final_status={a.claim_id: Status.CERTIFIED, b.claim_id: Status.CERTIFIED},
        )

    resolution = rules.decide_resolution(a, b)
    checks = rules.compute_checks(a, b, resolution)
    route = rules.route_conflict(checks, inconsistent=not consistent, verifier_disputed=disputed)
    if route == Route.AUTO:
        final = rules.statuses_from_resolution(a, b, resolution)
        proposal = ""
    else:
        proposal = backend.propose_text(a, b, resolution)
        final = reviewer.decide(case_id, a, b, resolution)
    return PairResult(
        **fields,
        checks=checks,
        route=route,
        resolution=resolution,
        proposal_text=proposal,
        final_status=final,
    )


def _usage_delta(before: Usage, after: Usage) -> Usage:
    return Usage(
        calls=after.calls - before.calls,
        input_tokens=after.input_tokens - before.input_tokens,
        output_tokens=after.output_tokens - before.output_tokens,
    )
