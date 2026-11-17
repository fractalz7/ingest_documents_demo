"""Reproduce the 50-seed sweep: verifier on/off, label noise, and topic-extraction noise.

Run from the project root:  python scripts/seed_sweep.py
Prints one row per setting, pooled over 50 seeds x the sample cases.

TopicNoisy is defined here, not in the package: it adds an extraction topic error that
NoisyBackend does not model yet, to show how a mislabeled topic silently hides a conflict.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcd.backends.stub import NoisyBackend, OracleReviewer  # noqa: E402
from pcd.evals.metrics import summarize  # noqa: E402
from pcd.evals.run_evals import load_cases, run_all  # noqa: E402
from pcd.pipeline import PipelineConfig  # noqa: E402
from pcd.schemas import TOPICS, Risk, Route  # noqa: E402

SEEDS = range(50)
CASES = load_cases(Path(__file__).resolve().parent.parent / "data")
GOLD_TOPIC = {c.claim_id: c.topic for case in CASES for c in case.gold.claims}


class TopicNoisy(NoisyBackend):
    """NoisyBackend plus an independent per-claim topic error."""

    def __init__(self, *args, topic_error: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.topic_error = topic_error

    def _corrupt_claim(self, claim):
        out = super()._corrupt_claim(claim)
        rng = random.Random(f"{self.seed}:topic:{claim.claim_id}")
        if rng.random() < self.topic_error:
            others = [t for t in TOPICS if t != out.topic]
            return out.model_copy(update={"topic": rng.choice(others)})
        return out


def sweep(label: str, verifier: bool, topic_error: float = 0.0, **noise) -> None:
    wc_k = wc_n = ow_k = ow_n = calls = pairs = escalated = risky = 0
    for seed in SEEDS:
        backend = TopicNoisy(CASES, seed=seed, topic_error=topic_error, **noise)
        results = run_all(CASES, backend, OracleReviewer(CASES), PipelineConfig(use_verifier=verifier))
        summary = summarize(CASES, results)
        end = summary["end_to_end"]
        wc_k += end["wrong_certified"]["k"]
        wc_n += end["wrong_certified"]["n"]
        ow_k += end["over_withholding"]["k"]
        ow_n += end["over_withholding"]["n"]
        calls += summary["cost"]["calls"]
        for case in CASES:
            for pair in results[case.case_id].pairs:
                pairs += 1
                escalated += pair.route == Route.HUMAN
                # Risk judged by the TRUE topic, not the extracted one.
                if pair.route == Route.AUTO and TOPICS[GOLD_TOPIC[pair.claim_a_id]] == Risk.HIGH:
                    risky += 1
    runs = len(SEEDS) * len(CASES)
    print(
        f"{label:<44} wrong_certified {wc_k / wc_n:6.1%}  over_withhold {ow_k / ow_n:6.1%}  "
        f"escalated {escalated / max(pairs, 1):6.1%}  calls/case {calls / runs:5.1f}  "
        f"risky_auto(true topic) {risky}"
    )


def main() -> None:
    print(f"{len(SEEDS)} seeds x {len(CASES)} cases = {len(SEEDS) * len(CASES)} case-runs per row")
    sweep("default noise, no verifier", False)
    sweep("default noise, verifier", True)
    sweep("label error 40%, no verifier", False, detect_label_error=0.40)
    sweep("label error 40%, verifier", True, detect_label_error=0.40)
    sweep("topic error 15%, verifier", True, topic_error=0.15)
    sweep("topic error 15%, verifier + label err 40%", True, topic_error=0.15, detect_label_error=0.40)


if __name__ == "__main__":
    main()
