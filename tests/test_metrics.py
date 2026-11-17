"""Metric unit tests with hand-built gold cases and predictions (no pipeline run)."""
from __future__ import annotations

from datetime import date

import pytest

from pcd import rules
from pcd.evals import metrics
from pcd.schemas import (
    Authority,
    Case,
    Claim,
    ConflictLabel,
    Gold,
    GoldPair,
    Owner,
    PairResult,
    PipelineResult,
    Route,
    Scope,
    Status,
    Usage,
)

TC, SD, OUT = ConflictLabel.TRUE_CONTRADICTION, ConflictLabel.SCOPE_DIFFERENCE, ConflictLabel.OUTDATED_VERSION
AMB, NO = ConflictLabel.AMBIGUITY_TIE, ConflictLabel.NO_CONFLICT


def _claim(doc_id, topic="callback_window_hours", value="24"):
    return Claim(claim_id=f"{doc_id}:{topic}", doc_id=doc_id, topic=topic, value=value,
                 scope=Scope(), effective_start=date(2026, 1, 1),
                 authority=Authority.SOP, owner=Owner.PAYER)


def _pair_result(a, b, label, route, status=None):
    return PairResult(
        claim_a_id=a.claim_id, claim_b_id=b.claim_id, predicted_label=label,
        scope_overlap=True, date_overlap=True, values_differ=True, route=route,
        final_status=status or {a.claim_id: Status.CERTIFIED, b.claim_id: Status.CERTIFIED},
    )


def build(case_id, rows):
    """rows: (gold_label, gold_route, pred_label or None, pred_route, topic). One row is one pair."""
    claims, gold_pairs, preds = [], [], []
    for i, (gl, gr, pl, pr, topic) in enumerate(rows):
        a = _claim(f"{case_id}-{i}-A", topic)
        b = _claim(f"{case_id}-{i}-B", topic)
        claims += [a, b]
        status = {a.claim_id: Status.CERTIFIED, b.claim_id: Status.CERTIFIED}
        gold_pairs.append(GoldPair(
            claim_a_id=a.claim_id, claim_b_id=b.claim_id, label=gl, route=gr,
            risk=rules.risk_of(topic), scope_overlap=True, date_overlap=True,
            final_status=status,
        ))
        if pl is not None:
            preds.append(_pair_result(a, b, pl, pr, status))
    case = Case(case_id=case_id, split="dev", category="unit", documents=[],
                gold=Gold(metadata={}, claims=claims, pairs=gold_pairs,
                          final_status={c.claim_id: Status.CERTIFIED for c in claims}))
    result = PipelineResult(case_id=case_id, claims=claims, pairs=preds,
                            final_status={c.claim_id: Status.CERTIFIED for c in claims})
    return case, result


LOW = "callback_window_hours"
HIGH = "specialist_referral"

# gold label, gold route, predicted label, predicted route
DETECTION_ROWS = [
    (TC, Route.HUMAN, TC, Route.AUTO, LOW),
    (TC, Route.AUTO, TC, Route.AUTO, LOW),
    (TC, Route.HUMAN, SD, Route.NONE, LOW),
    (TC, Route.HUMAN, AMB, Route.HUMAN, LOW),
    (SD, Route.NONE, SD, Route.NONE, LOW),
    (SD, Route.NONE, TC, Route.HUMAN, LOW),
    (NO, Route.NONE, NO, Route.NONE, LOW),
]


def test_wilson_interval_matches_reference_value():
    low, high = metrics.wilson_interval(8, 10)
    assert abs(low - 0.490) < 0.001
    assert abs(high - 0.943) < 0.001


def test_wilson_interval_with_no_trials_is_full_range():
    assert metrics.wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_stays_inside_unit_range():
    low, high = metrics.wilson_interval(0, 5)
    assert low == 0.0 and 0.0 < high < 1.0
    low, high = metrics.wilson_interval(5, 5)
    assert 0.0 < low < 1.0 and high == 1.0


def test_detection_precision_recall_f1_from_known_confusion():
    case, result = build("C-DET", DETECTION_ROWS)
    det = metrics.detection_metrics([case], {case.case_id: result})

    # TRUE_CONTRADICTION: support 4, tp 2, fp 1, fn 2
    tc = det["per_label"][TC.value]
    assert tc["support"] == 4
    assert tc["precision"] == pytest.approx(2 / 3)
    assert tc["recall"] == pytest.approx(0.5)
    assert tc["f1"] == pytest.approx(4 / 7)

    # SCOPE_DIFFERENCE: support 2, tp 1, fp 1, fn 1
    sd = det["per_label"][SD.value]
    assert sd["precision"] == pytest.approx(0.5)
    assert sd["recall"] == pytest.approx(0.5)
    assert sd["f1"] == pytest.approx(0.5)

    # AMBIGUITY_TIE is predicted but never in gold: precision, recall and F1 are all 0
    amb = det["per_label"][AMB.value]
    assert (amb["support"], amb["precision"], amb["recall"], amb["f1"]) == (0, 0.0, 0.0, 0.0)

    assert det["per_label"][NO.value]["f1"] == pytest.approx(1.0)
    assert det["accuracy"] == pytest.approx(4 / 7)
    assert det["true_contradiction_recall"] == pytest.approx(0.5)
    # 1 of the 3 gold non-conflict pairs was predicted as a conflict
    assert det["false_conflict_rate"] == pytest.approx(1 / 3)
    assert det["confusion"][TC.value][SD.value] == 1


def test_missing_predicted_pair_counts_as_wrong():
    case, result = build("C-MISS", [(TC, Route.HUMAN, None, None, LOW),
                                    (SD, Route.NONE, SD, Route.NONE, LOW)])
    det = metrics.detection_metrics([case], {case.case_id: result})
    assert det["accuracy"] == pytest.approx(0.5)
    assert det["confusion"][TC.value] == {metrics.MISSING: 1}


def test_routing_accuracy_escalations_and_false_escalations():
    case, result = build("C-ROUTE", DETECTION_ROWS)
    rt = metrics.routing_metrics([case], {case.case_id: result})
    # gold routes: HUMAN, AUTO, HUMAN, HUMAN, NONE, NONE, NONE
    # predicted:   AUTO, AUTO, NONE, HUMAN, NONE, HUMAN, NONE
    assert rt["routing_accuracy"] == pytest.approx(4 / 7)
    assert rt["false_escalations"] == 1  # gold NONE, predicted HUMAN (row 5)
    assert rt["escalation_rate"] == pytest.approx(2 / 7)
    assert rt["risky_auto_resolved"] == 0


def test_risky_auto_counts_only_high_risk_topics_routed_auto():
    case, result = build("C-RISK", [
        (TC, Route.HUMAN, TC, Route.AUTO, HIGH),
        (TC, Route.AUTO, TC, Route.AUTO, LOW),
        (TC, Route.HUMAN, TC, Route.HUMAN, HIGH),
    ])
    rt = metrics.routing_metrics([case], {case.case_id: result})
    assert rt["risky_auto_resolved"] == 1


def test_end_to_end_rates_and_intervals():
    claims = [_claim(f"D{i}") for i in range(4)]
    gold_status = [Status.CERTIFIED, Status.CERTIFIED, Status.WITHHELD, Status.CERTIFIED]
    pred_status = [Status.CERTIFIED, Status.WITHHELD, Status.CERTIFIED, Status.CERTIFIED]
    gold = Gold(metadata={}, claims=claims, pairs=[],
                final_status={c.claim_id: s for c, s in zip(claims, gold_status)})
    case = Case(case_id="C-E2E", split="dev", category="unit", documents=[], gold=gold)
    result = PipelineResult(case_id="C-E2E", claims=claims,
                            final_status={c.claim_id: s for c, s in zip(claims, pred_status)})
    e2e = metrics.end_to_end_metrics([case], {"C-E2E": result})
    # predicted CERTIFIED: claims 0, 2, 3; wrong when gold is not CERTIFIED: claim 2
    assert (e2e["wrong_certified"]["k"], e2e["wrong_certified"]["n"]) == (1, 3)
    # gold CERTIFIED: claims 0, 1, 3; withheld in prediction: claim 1
    assert (e2e["over_withholding"]["k"], e2e["over_withholding"]["n"]) == (1, 3)
    low, high = metrics.wilson_interval(1, 3)
    assert e2e["wrong_certified"]["ci95"] == pytest.approx([low, high])


def test_extraction_metrics_on_wrong_value_and_missing_and_extra_claims():
    gold_x = _claim("X", value="24")
    gold_y = _claim("Y", value="30")
    extra = _claim("Z", value="5")
    wrong_x = gold_x.model_copy(update={"value": "99"})
    gold = Gold(metadata={}, claims=[gold_x, gold_y], pairs=[], final_status={})
    case = Case(case_id="C-EXT", split="dev", category="unit", documents=[], gold=gold)
    result = PipelineResult(case_id="C-EXT", claims=[wrong_x, extra])
    ext = metrics.extraction_metrics([case], {"C-EXT": result})
    assert ext["claim_recall"] == pytest.approx(0.5)
    assert ext["claim_precision"] == pytest.approx(0.5)
    assert ext["field_accuracy"]["value"] == 0.0
    assert ext["field_accuracy"]["topic"] == pytest.approx(0.5)


def test_cost_metrics_totals_and_per_case_averages():
    def res(cid, calls):
        return PipelineResult(case_id=cid, usage=Usage(calls=calls, input_tokens=10 * calls,
                                                        output_tokens=calls))
    cost = metrics.cost_metrics({"a": res("a", 4), "b": res("b", 8)})
    assert cost["calls"] == 12
    assert cost["input_tokens"] == 120
    assert cost["output_tokens"] == 12
    assert cost["per_case"]["calls"] == pytest.approx(6.0)
