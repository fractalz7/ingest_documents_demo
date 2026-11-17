"""Pure metric functions over gold cases and pipeline results.

`results` maps case_id to PipelineResult. A missing predicted pair counts as
wrong, and a claim missing from the predicted final status counts as not CERTIFIED.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Mapping, Sequence

from pcd import rules
from pcd.schemas import (
    Case,
    Claim,
    ConflictLabel,
    PairResult,
    PipelineResult,
    Risk,
    Route,
    Status,
)

MISSING = "MISSING"
CONFLICT_VALUES = {label.value for label in rules.CONFLICT_LABELS}

_FIELD_GETTERS: dict[str, Callable[[Claim], object]] = {
    "topic": lambda c: c.topic,
    "value": lambda c: c.value,
    "plan": lambda c: c.scope.plan,
    "state": lambda c: c.scope.state,
    "setting": lambda c: c.scope.setting,
    "authority": lambda c: c.authority,
    "owner": lambda c: c.owner,
    "effective_start": lambda c: c.effective_start,
    "effective_end": lambda c: c.effective_end,
}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; (0, 1) when n is 0."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _ratio(k: int, n: int) -> float:
    return k / n if n else 0.0


def _rate(k: int, n: int) -> dict:
    low, high = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": _ratio(k, n), "ci95": [low, high]}


def _pair_key(a_id: str, b_id: str) -> tuple[str, str]:
    return tuple(sorted((a_id, b_id)))  # type: ignore[return-value]


def _predicted_pairs(result: PipelineResult) -> dict[tuple[str, str], PairResult]:
    return {_pair_key(p.claim_a_id, p.claim_b_id): p for p in result.pairs}


def extraction_metrics(cases: Sequence[Case], results: Mapping[str, PipelineResult]) -> dict:
    gold_total = matched = extracted_total = extracted_matched = 0
    field_hits = {name: 0 for name in _FIELD_GETTERS}
    for case in cases:
        extracted = {c.claim_id: c for c in results[case.case_id].claims}
        gold_ids = {g.claim_id for g in case.gold.claims}
        extracted_total += len(extracted)
        extracted_matched += sum(1 for cid in extracted if cid in gold_ids)
        for gold in case.gold.claims:
            gold_total += 1
            got = extracted.get(gold.claim_id)
            if got is None:
                continue
            matched += 1
            for name, get in _FIELD_GETTERS.items():
                field_hits[name] += get(got) == get(gold)
    return {
        "gold_claims": gold_total,
        "extracted_claims": extracted_total,
        "claim_recall": _ratio(matched, gold_total),
        "claim_precision": _ratio(extracted_matched, extracted_total),
        "field_accuracy": {name: _ratio(hits, gold_total) for name, hits in field_hits.items()},
    }


def detection_metrics(cases: Sequence[Case], results: Mapping[str, PipelineResult]) -> dict:
    confusion: dict[str, Counter] = {}
    for case in cases:
        predicted = _predicted_pairs(results[case.case_id])
        for gp in case.gold.pairs:
            pair = predicted.get(_pair_key(gp.claim_a_id, gp.claim_b_id))
            pred = pair.predicted_label.value if pair else MISSING
            confusion.setdefault(gp.label.value, Counter())[pred] += 1

    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(row.get(gold, 0) for gold, row in confusion.items())
    labels = sorted(
        set(confusion) | {p for row in confusion.values() for p in row if p != MISSING}
    )
    per_label = {}
    for label in labels:
        support = sum(confusion.get(label, Counter()).values())
        tp = confusion.get(label, Counter()).get(label, 0)
        fp = sum(row.get(label, 0) for gold, row in confusion.items() if gold != label)
        fn = support - tp
        per_label[label] = {
            "support": support,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, support),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        }

    nonconflict_rows = [
        row for gold, row in confusion.items() if gold not in CONFLICT_VALUES
    ]
    nonconflict_total = sum(sum(row.values()) for row in nonconflict_rows)
    false_conflicts = sum(
        count
        for row in nonconflict_rows
        for pred, count in row.items()
        if pred in CONFLICT_VALUES
    )
    return {
        "pairs": total,
        "accuracy": _ratio(correct, total),
        "per_label": per_label,
        "confusion": {g: dict(sorted(row.items())) for g, row in sorted(confusion.items())},
        "true_contradiction_recall": per_label.get(
            ConflictLabel.TRUE_CONTRADICTION.value, {}
        ).get("recall", 0.0),
        "false_conflict_rate": _ratio(false_conflicts, nonconflict_total),
    }


def routing_metrics(cases: Sequence[Case], results: Mapping[str, PipelineResult]) -> dict:
    gold_total = correct = false_escalations = 0
    predicted_total = predicted_human = risky_auto = 0
    for case in cases:
        result = results[case.case_id]
        topic_of = {c.claim_id: c.topic for c in result.claims}
        for pair in result.pairs:
            predicted_total += 1
            predicted_human += pair.route == Route.HUMAN
            topic = topic_of.get(pair.claim_a_id, "")
            risky_auto += pair.route == Route.AUTO and rules.risk_of(topic) == Risk.HIGH

        predicted = _predicted_pairs(result)
        for gp in case.gold.pairs:
            gold_total += 1
            pair = predicted.get(_pair_key(gp.claim_a_id, gp.claim_b_id))
            pred_route = pair.route if pair else None
            correct += pred_route == gp.route
            false_escalations += gp.route in (Route.AUTO, Route.NONE) and pred_route == Route.HUMAN
    return {
        "pairs": gold_total,
        "routing_accuracy": _ratio(correct, gold_total),
        "risky_auto_resolved": risky_auto,
        "escalation_rate": _ratio(predicted_human, predicted_total),
        "false_escalations": false_escalations,
    }


def end_to_end_metrics(cases: Sequence[Case], results: Mapping[str, PipelineResult]) -> dict:
    certified = wrong_certified = gold_certified = over_withheld = 0
    for case in cases:
        predicted = results[case.case_id].final_status
        for claim_id, gold_status in case.gold.final_status.items():
            pred_status = predicted.get(claim_id)
            if pred_status == Status.CERTIFIED:
                certified += 1
                wrong_certified += gold_status != Status.CERTIFIED
            if gold_status == Status.CERTIFIED:
                gold_certified += 1
                over_withheld += pred_status != Status.CERTIFIED
    return {
        "wrong_certified": _rate(wrong_certified, certified),
        "over_withholding": _rate(over_withheld, gold_certified),
    }


def cost_metrics(results: Mapping[str, PipelineResult]) -> dict:
    n = len(results)
    calls = sum(r.usage.calls for r in results.values())
    input_tokens = sum(r.usage.input_tokens for r in results.values())
    output_tokens = sum(r.usage.output_tokens for r in results.values())
    return {
        "cases": n,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "per_case": {
            "calls": _ratio(calls, n),
            "input_tokens": _ratio(input_tokens, n),
            "output_tokens": _ratio(output_tokens, n),
        },
    }


def summarize(cases: Sequence[Case], results: Mapping[str, PipelineResult]) -> dict:
    return {
        "cases": len(cases),
        "extraction": extraction_metrics(cases, results),
        "detection": detection_metrics(cases, results),
        "routing": routing_metrics(cases, results),
        "end_to_end": end_to_end_metrics(cases, results),
        "cost": cost_metrics({c.case_id: results[c.case_id] for c in cases}),
    }
