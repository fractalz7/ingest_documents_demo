"""Offline backends for the demo and the tests: an oracle, a noisy wrapper, and an oracle reviewer.

Nothing here touches the network. Noise is keyed by seed, stage and item id, so a
run does not depend on call order.
"""
from __future__ import annotations

import random
from typing import Iterable, Optional

from pcd.backends.base import UsageMeter
from pcd.schemas import (
    ANY,
    Authority,
    Case,
    Claim,
    ConflictLabel,
    Document,
    DocumentMetadata,
    GoldPair,
    PairJudgement,
    Resolution,
    Status,
    Verdict,
)

SCOPE_VALUES = {
    "plan": [ANY, "NW-Gold", "NW-Silver"],
    "state": [ANY, "FL", "TX"],
    "setting": [ANY, "member", "clinic"],
}


def _pair_key(a_id: str, b_id: str) -> tuple[str, str]:
    return tuple(sorted((a_id, b_id)))  # type: ignore[return-value]


def _index_pairs(cases: Iterable[Case]) -> dict[tuple[str, str], GoldPair]:
    return {
        _pair_key(gp.claim_a_id, gp.claim_b_id): gp
        for case in cases
        for gp in case.gold.pairs
    }


def _proposal_text(a: Claim, b: Claim, resolution: Optional[Resolution]) -> str:
    if resolution is None:
        return f"{a.claim_id} and {b.claim_id} conflict on {a.topic}; no rule decides, owner to rule."
    return (
        f"{a.claim_id} vs {b.claim_id} on {a.topic}: {resolution.rule_fired} selects "
        f"{resolution.winner_claim_id}; {resolution.loser_claim_id} is "
        f"{resolution.loser_disposition.value}."
    )


class OracleBackend:
    """Returns gold labels exactly. The verifier always agrees."""

    name = "oracle"

    def __init__(self, cases: Iterable[Case]) -> None:
        cases = list(cases)
        self._metadata: dict[str, DocumentMetadata] = {}
        self._claims_by_doc: dict[str, list[Claim]] = {}
        for case in cases:
            self._metadata.update(case.gold.metadata)
            for claim in case.gold.claims:
                self._claims_by_doc.setdefault(claim.doc_id, []).append(claim)
        self._pairs = _index_pairs(cases)
        self._meter = UsageMeter()

    @property
    def usage(self):
        return self._meter.usage

    def gold_pair(self, a_id: str, b_id: str) -> GoldPair:
        try:
            return self._pairs[_pair_key(a_id, b_id)]
        except KeyError:
            raise KeyError(f"no gold pair for {a_id} and {b_id}") from None

    def classify(self, doc: Document) -> Optional[DocumentMetadata]:
        self._meter.add()
        return self._metadata.get(doc.doc_id)

    def extract(self, doc: Document, meta: DocumentMetadata) -> list[Claim]:
        self._meter.add()
        return [c.model_copy(deep=True) for c in self._claims_by_doc.get(doc.doc_id, [])]

    def detect(self, a: Claim, b: Claim) -> PairJudgement:
        self._meter.add()
        return PairJudgement(label=self.gold_pair(a.claim_id, b.claim_id).label, rationale="oracle")

    def verify(self, a: Claim, b: Claim, judgement: PairJudgement) -> Verdict:
        self._meter.add()
        return Verdict(agrees=True, rationale="oracle")

    def propose_text(self, a: Claim, b: Claim, resolution: Optional[Resolution]) -> str:
        self._meter.add()
        return _proposal_text(a, b, resolution)


class NoisyBackend(OracleBackend):
    """The oracle with injected errors. Each error type has its own keyed RNG stream."""

    name = "noisy"

    def __init__(
        self,
        cases: Iterable[Case],
        seed: int = 0,
        classify_error: float = 0.05,
        extract_scope_error: float = 0.10,
        extract_value_error: float = 0.05,
        detect_label_error: float = 0.15,
        verifier_catch_rate: float = 0.60,
        verifier_false_dispute: float = 0.05,
    ) -> None:
        super().__init__(cases)
        self.seed = seed
        self.classify_error = classify_error
        self.extract_scope_error = extract_scope_error
        self.extract_value_error = extract_value_error
        self.detect_label_error = detect_label_error
        self.verifier_catch_rate = verifier_catch_rate
        self.verifier_false_dispute = verifier_false_dispute

    def _rng(self, stage: str, key: str) -> random.Random:
        return random.Random(f"{self.seed}:{stage}:{key}")

    def classify(self, doc: Document) -> Optional[DocumentMetadata]:
        meta = super().classify(doc)
        if meta is None:
            return None
        rng = self._rng("classify", doc.doc_id)
        if rng.random() < self.classify_error:
            others = [a for a in Authority if a != meta.authority]
            return meta.model_copy(update={"authority": rng.choice(others)})
        return meta

    def extract(self, doc: Document, meta: DocumentMetadata) -> list[Claim]:
        return [self._corrupt_claim(c) for c in super().extract(doc, meta)]

    def _corrupt_claim(self, claim: Claim) -> Claim:
        rng = self._rng("extract", claim.claim_id)
        updates = {}
        if rng.random() < self.extract_scope_error:
            dim = rng.choice(list(SCOPE_VALUES))
            current = getattr(claim.scope, dim)
            new = rng.choice([v for v in SCOPE_VALUES[dim] if v != current])
            updates["scope"] = claim.scope.model_copy(update={dim: new})
        if rng.random() < self.extract_value_error:
            updates["value"] = _perturb_value(claim.value, rng)
        return claim.model_copy(update=updates) if updates else claim

    def detect(self, a: Claim, b: Claim) -> PairJudgement:
        judgement = super().detect(a, b)
        rng = self._rng("detect", f"{a.claim_id}|{b.claim_id}")
        if rng.random() < self.detect_label_error:
            others = [label for label in ConflictLabel if label != judgement.label]
            return PairJudgement(label=rng.choice(others), rationale="noisy label")
        return judgement

    def verify(self, a: Claim, b: Claim, judgement: PairJudgement) -> Verdict:
        super().verify(a, b, judgement)
        wrong = judgement.label != self.gold_pair(a.claim_id, b.claim_id).label
        rate = self.verifier_catch_rate if wrong else self.verifier_false_dispute
        rng = self._rng("verify", f"{a.claim_id}|{b.claim_id}")
        disputed = rng.random() < rate
        return Verdict(agrees=not disputed, rationale="noisy verifier")


def _perturb_value(value: str, rng: random.Random) -> str:
    """Change a normalized value to another value of the same format."""
    if value in ("required", "not_required"):
        return "not_required" if value == "required" else "required"
    if value.startswith("USD:"):
        return f"USD:{int(value[4:]) + rng.choice([1, 5, 10])}"
    return str(int(value) + rng.choice([1, 5, 10]))


class OracleReviewer:
    """A policy owner who is always right: returns the gold final status for the pair."""

    def __init__(self, cases: Iterable[Case]) -> None:
        self._pairs = _index_pairs(cases)

    def decide(
        self, case_id: str, a: Claim, b: Claim, resolution: Optional[Resolution]
    ) -> dict[str, Status]:
        gp = self._pairs[_pair_key(a.claim_id, b.claim_id)]
        return dict(gp.final_status)
