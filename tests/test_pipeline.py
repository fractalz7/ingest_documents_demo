"""Pipeline tests with hand-built cases, plus the sample-file acceptance checks (spec section 9)."""
from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from pcd import pipeline
from pcd.backends.base import UsageMeter
from pcd.backends.stub import NoisyBackend, OracleBackend, OracleReviewer
from pcd.evals import metrics
from pcd.evals.run_evals import load_cases, main, run_all
from pcd.pipeline import PipelineConfig, run_case
from pcd.schemas import (
    Authority,
    Case,
    ConflictLabel,
    Document,
    DocumentMetadata,
    Gold,
    Owner,
    PairJudgement,
    PairResult,
    Route,
    Scope,
    Status,
    Verdict,
    Claim,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data"


class ScriptedBackend:
    """Fixed classify/extract outputs, and a fixed detect label and verdict."""

    name = "scripted"

    def __init__(self, claims_by_doc, meta_by_doc, label, agrees=True):
        self.claims_by_doc = claims_by_doc
        self.meta_by_doc = meta_by_doc
        self.label = label
        self.agrees = agrees
        self._meter = UsageMeter()

    @property
    def usage(self):
        return self._meter.usage

    def classify(self, doc):
        self._meter.add()
        return self.meta_by_doc.get(doc.doc_id)

    def extract(self, doc, meta):
        self._meter.add()
        return [c.model_copy() for c in self.claims_by_doc.get(doc.doc_id, [])]

    def detect(self, a, b):
        self._meter.add()
        return PairJudgement(label=self.label)

    def verify(self, a, b, judgement):
        self._meter.add()
        return Verdict(agrees=self.agrees)

    def propose_text(self, a, b, resolution):
        self._meter.add()
        return "proposal"


class RecordingReviewer:
    def __init__(self):
        self.calls = []

    def decide(self, case_id, a, b, resolution):
        self.calls.append((a.claim_id, b.claim_id))
        return {a.claim_id: Status.WITHHELD, b.claim_id: Status.WITHHELD}


def claim(doc, topic="callback_window_hours", authority=Authority.SOP, owner=Owner.PAYER,
          scope=None, start=date(2026, 1, 1), supersedes=None, value="24"):
    return Claim(
        claim_id=f"DOC-{doc}:{topic}", doc_id=f"DOC-{doc}", topic=topic, value=value,
        scope=scope or Scope(), effective_start=start, authority=authority, owner=owner,
        supersedes_doc_id=supersedes,
    )


def meta(authority=Authority.SOP, owner=Owner.PAYER, start=date(2026, 1, 1), supersedes=None):
    return DocumentMetadata(owner=owner, authority=authority, effective_start=start,
                            supersedes_doc_id=supersedes)


def empty_case(docs, case_id="C-T"):
    return Case(case_id=case_id, split="dev", category="unit", documents=docs,
                gold=Gold(metadata={}, claims=[], pairs=[], final_status={}))


def two_doc_run(a, b, meta_a, meta_b, label=ConflictLabel.TRUE_CONTRADICTION,
                agrees=True, use_verifier=True):
    docs = [Document(doc_id="DOC-A", title="A", text="SYNTHETIC TEST DOCUMENT"),
            Document(doc_id="DOC-B", title="B", text="SYNTHETIC TEST DOCUMENT")]
    backend = ScriptedBackend({"DOC-A": [a], "DOC-B": [b]},
                              {"DOC-A": meta_a, "DOC-B": meta_b}, label, agrees)
    reviewer = RecordingReviewer()
    result = run_case(empty_case(docs), backend, reviewer, PipelineConfig(use_verifier=use_verifier))
    return result, reviewer


def superseding_pair(topic):
    old = claim("A", topic=topic, start=date(2025, 1, 1))
    new = claim("B", topic=topic, start=date(2026, 2, 1), supersedes="DOC-A", value="30")
    return old, new, meta(start=date(2025, 1, 1)), meta(start=date(2026, 2, 1), supersedes="DOC-A")


def test_low_risk_superseded_pair_is_auto_resolved():
    old, new, ma, mb = superseding_pair("callback_window_hours")
    result, reviewer = two_doc_run(old, new, ma, mb, label=ConflictLabel.OUTDATED_VERSION)
    (pair,) = result.pairs
    assert pair.route == Route.AUTO
    assert pair.resolution.rule_fired == "R1_SUPERSEDES"
    assert result.final_status == {old.claim_id: Status.WITHHELD, new.claim_id: Status.CERTIFIED}
    assert reviewer.calls == []


def test_high_risk_pair_never_routes_auto_even_when_rule_applies():
    old, new, ma, mb = superseding_pair("specialist_referral")
    result, reviewer = two_doc_run(old, new, ma, mb, label=ConflictLabel.OUTDATED_VERSION)
    assert result.pairs[0].route == Route.HUMAN
    assert result.pairs[0].resolution is not None
    assert len(reviewer.calls) == 1


@pytest.mark.parametrize("label", list(ConflictLabel))
@pytest.mark.parametrize("topic", ["specialist_referral", "copay_primary_care",
                                   "appeal_filing_deadline_days", "made_up_topic"])
@pytest.mark.parametrize("agrees", [True, False])
@pytest.mark.parametrize("use_verifier", [True, False])
@pytest.mark.parametrize("values", [("24", "30"), ("24", "24")])
def test_invariant_no_high_risk_auto_for_any_backend_output(label, topic, agrees,
                                                            use_verifier, values):
    a = claim("A", topic=topic, value=values[0], authority=Authority.CONTRACT_EOC,
              scope=Scope(plan="NW-Gold"))
    b = claim("B", topic=topic, value=values[1], authority=Authority.SOP)
    result, _ = two_doc_run(a, b, meta(Authority.CONTRACT_EOC), meta(Authority.SOP),
                            label=label, agrees=agrees, use_verifier=use_verifier)
    for pair in result.pairs:
        assert pair.route != Route.AUTO


def test_cross_owner_tie_goes_to_human():
    a = claim("A", authority=Authority.CORPORATE_POLICY, owner=Owner.PAYER)
    b = claim("B", authority=Authority.CORPORATE_POLICY, owner=Owner.CARE_DELIVERY,
              scope=Scope(plan="NW-Gold"))
    result, reviewer = two_doc_run(a, b, meta(Authority.CORPORATE_POLICY),
                                   meta(Authority.CORPORATE_POLICY, owner=Owner.CARE_DELIVERY))
    assert result.pairs[0].route == Route.HUMAN
    assert result.pairs[0].checks.no_cross_owner_tie is False
    assert len(reviewer.calls) == 1


def test_consistent_nonconflict_routes_none_and_certifies_both():
    a = claim("A", scope=Scope(state="FL"))
    b = claim("B", scope=Scope(state="TX"), value="48")
    result, reviewer = two_doc_run(a, b, meta(), meta(), label=ConflictLabel.SCOPE_DIFFERENCE)
    pair = result.pairs[0]
    assert pair.route == Route.NONE
    assert pair.checks is None
    assert result.final_status == {a.claim_id: Status.CERTIFIED, b.claim_id: Status.CERTIFIED}
    assert reviewer.calls == []


def test_nonconflict_label_on_fully_overlapping_facts_is_inconsistent_and_escalated():
    a = claim("A", scope=Scope(plan="NW-Gold"))
    b = claim("B", scope=Scope(plan="NW-Gold"), value="48")
    result, _ = two_doc_run(a, b, meta(), meta(), label=ConflictLabel.NO_CONFLICT)
    assert result.pairs[0].inconsistent is True
    assert result.pairs[0].route == Route.HUMAN


def test_conflict_label_with_equal_values_is_inconsistent_and_escalated():
    a = claim("A", scope=Scope(plan="NW-Gold"), authority=Authority.CONTRACT_EOC)
    b = claim("B", scope=Scope(plan="NW-Gold"), value="24")
    result, reviewer = two_doc_run(a, b, meta(Authority.CONTRACT_EOC), meta(),
                                   label=ConflictLabel.TRUE_CONTRADICTION)
    assert result.pairs[0].inconsistent is True
    assert result.pairs[0].route == Route.HUMAN
    assert len(reviewer.calls) == 1


def test_verifier_dispute_escalates_low_risk_pair_and_ablation_removes_it():
    old, new, ma, mb = superseding_pair("callback_window_hours")
    disputed, _ = two_doc_run(old, new, ma, mb, label=ConflictLabel.OUTDATED_VERSION,
                              agrees=False, use_verifier=True)
    assert disputed.pairs[0].verifier_disputed is True
    assert disputed.pairs[0].route == Route.HUMAN

    ablated, _ = two_doc_run(old, new, ma, mb, label=ConflictLabel.OUTDATED_VERSION,
                             agrees=False, use_verifier=False)
    assert ablated.pairs[0].verifier_disputed is False
    assert ablated.pairs[0].route == Route.AUTO


def test_blocked_document_is_excluded_entirely():
    a = claim("A")
    b = claim("B")
    docs = [Document(doc_id="DOC-A", title="A", text="x"),
            Document(doc_id="DOC-B", title="B", text="x")]
    backend = ScriptedBackend({"DOC-A": [a], "DOC-B": [b]},
                              {"DOC-A": meta()}, ConflictLabel.TRUE_CONTRADICTION)
    result = run_case(empty_case(docs), backend, RecordingReviewer())
    assert result.blocked_docs == ["DOC-B"]
    assert result.pairs == []
    assert [c.claim_id for c in result.claims] == [a.claim_id]
    assert result.final_status == {a.claim_id: Status.CERTIFIED}
    assert "DOC-B" not in result.metadata


def test_candidate_pairs_same_topic_cross_doc_sorted():
    c1 = claim("A")
    c2 = claim("C")
    c3 = claim("B")
    other = claim("B", topic="call_summary_retention_days")
    same_doc = Claim(claim_id="DOC-A:copay_primary_care", doc_id="DOC-A",
                     topic="copay_primary_care", value="USD:10", scope=Scope(),
                     effective_start=date(2026, 1, 1), authority=Authority.SOP,
                     owner=Owner.PAYER)
    pairs = pipeline.candidate_pairs([c1, c2, c3, other, same_doc])
    assert [(a.claim_id, b.claim_id) for a, b in pairs] == [
        ("DOC-A:callback_window_hours", "DOC-B:callback_window_hours"),
        ("DOC-A:callback_window_hours", "DOC-C:callback_window_hours"),
        ("DOC-B:callback_window_hours", "DOC-C:callback_window_hours"),
    ]


def test_aggregate_status_is_most_restrictive_across_pairs():
    c = [claim("A"), claim("B"), claim("C")]
    pairs = [
        PairResult(claim_a_id=c[0].claim_id, claim_b_id=c[1].claim_id,
                   predicted_label=ConflictLabel.NO_CONFLICT, scope_overlap=True,
                   date_overlap=True, values_differ=False, route=Route.NONE,
                   final_status={c[0].claim_id: Status.CERTIFIED, c[1].claim_id: Status.SCOPED}),
        PairResult(claim_a_id=c[0].claim_id, claim_b_id=c[2].claim_id,
                   predicted_label=ConflictLabel.NO_CONFLICT, scope_overlap=True,
                   date_overlap=True, values_differ=False, route=Route.NONE,
                   final_status={c[0].claim_id: Status.WITHHELD, c[2].claim_id: Status.CERTIFIED}),
    ]
    final = pipeline.aggregate_status(c, pairs)
    assert final == {c[0].claim_id: Status.WITHHELD, c[1].claim_id: Status.SCOPED,
                     c[2].claim_id: Status.CERTIFIED}


def test_usage_is_reported_per_case_as_a_delta():
    old, new, ma, mb = superseding_pair("callback_window_hours")
    docs = [Document(doc_id="DOC-A", title="A", text="x"),
            Document(doc_id="DOC-B", title="B", text="x")]
    backend = ScriptedBackend({"DOC-A": [old], "DOC-B": [new]},
                              {"DOC-A": ma, "DOC-B": mb}, ConflictLabel.OUTDATED_VERSION)
    reviewer = RecordingReviewer()
    first = run_case(empty_case(docs, "C1"), backend, reviewer)
    second = run_case(empty_case(docs, "C2"), backend, reviewer)
    # 2 classify + 2 extract + 1 detect + 1 verify, and no propose_text for an AUTO pair
    assert first.usage.calls == 6
    assert second.usage.calls == 6
    assert backend.usage.calls == 12


# Sample-file checks (spec section 9, items 2 to 5, 8 and 9). These skip until
# pcd/data/cases_sample10.jsonl exists.

def _sample_cases():
    if not SAMPLE.exists():
        pytest.skip(f"{SAMPLE.name} not generated yet")
    return load_cases(SAMPLE)


def test_oracle_run_is_perfect():
    cases = _sample_cases()
    results = run_all(cases, OracleBackend(cases), OracleReviewer(cases))
    summary = metrics.summarize(cases, results)
    assert summary["extraction"]["claim_recall"] == 1.0
    assert summary["extraction"]["claim_precision"] == 1.0
    assert all(v == 1.0 for v in summary["extraction"]["field_accuracy"].values())
    for label, row in summary["detection"]["per_label"].items():
        assert row["f1"] == 1.0, label
    assert summary["routing"]["routing_accuracy"] == 1.0
    assert summary["routing"]["risky_auto_resolved"] == 0
    assert summary["end_to_end"]["wrong_certified"]["k"] == 0
    assert summary["end_to_end"]["over_withholding"]["k"] == 0


@pytest.mark.parametrize("rate", [None, 0.9])
def test_zero_risky_auto_resolves_for_seeds_0_to_49(rate):
    cases = _sample_cases()
    kwargs = {} if rate is None else dict(
        classify_error=rate, extract_scope_error=rate, extract_value_error=rate,
        detect_label_error=rate, verifier_catch_rate=rate, verifier_false_dispute=rate,
    )
    for seed in range(50):
        backend = NoisyBackend(cases, seed=seed, **kwargs)
        results = run_all(cases, backend, OracleReviewer(cases))
        assert metrics.routing_metrics(cases, results)["risky_auto_resolved"] == 0, seed


def test_same_seed_gives_identical_results_json(tmp_path, capsys):
    _sample_cases()
    outputs = {}
    for name, seed in (("a", 3), ("b", 3), ("c", 4)):
        out = tmp_path / name
        assert main(["--cases", str(SAMPLE), "--backend", "noisy",
                     "--seed", str(seed), "--out", str(out)]) == 0
        outputs[name] = (out / "results.json").read_bytes()
    assert outputs["a"] == outputs["b"]
    assert outputs["a"] != outputs["c"]


def test_detection_accuracy_is_reported_honestly_under_total_label_error():
    cases = _sample_cases()
    backend = NoisyBackend(cases, seed=0, detect_label_error=1.0)
    results = run_all(cases, backend, OracleReviewer(cases))
    assert metrics.detection_metrics(cases, results)["accuracy"] == 0.0


def test_verifier_does_not_increase_wrong_certified_rate_on_average():
    cases = _sample_cases()

    def mean_wrong_rate(use_verifier):
        rates = []
        for seed in range(50):
            backend = NoisyBackend(cases, seed=seed, detect_label_error=0.9)
            results = run_all(cases, backend, OracleReviewer(cases),
                              PipelineConfig(use_verifier=use_verifier))
            rates.append(metrics.end_to_end_metrics(cases, results)["wrong_certified"]["rate"])
        return sum(rates) / len(rates)

    assert mean_wrong_rate(True) <= mean_wrong_rate(False)


def test_cli_runs_with_oracle_backend(tmp_path):
    _sample_cases()
    proc = subprocess.run(
        [sys.executable, "-m", "pcd.evals.run_evals", "--cases", str(SAMPLE),
         "--backend", "oracle", "--out", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "|" in proc.stdout
    assert (tmp_path / "results.json").exists()
