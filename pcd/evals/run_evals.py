"""Command-line eval runner.

    python -m pcd.evals.run_evals --cases data --backend oracle
    python -m pcd.evals.run_evals --cases ... --backend noisy --seed 0 [--no-verifier]
    python -m pcd.evals.run_evals --cases ... --compare

Prints a markdown table and writes results.json under --out. The file has no
timestamps, so the same inputs give byte-identical output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from pcd.backends.base import Backend, Reviewer
from pcd.backends.stub import NoisyBackend, OracleBackend, OracleReviewer
from pcd.evals.metrics import summarize, wilson_interval
from pcd.pipeline import PipelineConfig, run_case
from pcd.schemas import Case, PipelineResult

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "data"


def load_cases(path: Path) -> list[Case]:
    """Load cases from a data folder (documents/*.txt + gold.json) or a legacy .jsonl file."""
    path = Path(path)
    if path.is_dir():
        from pcd.data.loader import load_case_dir
        return load_case_dir(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Case.model_validate_json(line) for line in lines if line.strip()]


def make_backend(name: str, cases: Sequence[Case], seed: int) -> tuple[Backend, Reviewer]:
    if name == "oracle":
        return OracleBackend(cases), OracleReviewer(cases)
    if name == "noisy":
        return NoisyBackend(cases, seed=seed), OracleReviewer(cases)
    raise ValueError(f"unknown backend: {name}")


def run_all(
    cases: Sequence[Case],
    backend: Backend,
    reviewer: Reviewer,
    config: Optional[PipelineConfig] = None,
) -> dict[str, PipelineResult]:
    return {case.case_id: run_case(case, backend, reviewer, config) for case in cases}


def evaluate(
    cases: Sequence[Case], backend_name: str, seed: int = 0, use_verifier: bool = True
) -> dict:
    backend, reviewer = make_backend(backend_name, cases, seed)
    results = run_all(cases, backend, reviewer, PipelineConfig(use_verifier=use_verifier))
    return summarize(cases, results)


def _fmt_rate(rate: dict) -> str:
    low, high = rate["ci95"]
    return f"{rate['rate']:.3f} [{low:.3f}, {high:.3f}] ({rate['k']}/{rate['n']})"


def _flatten(summary: Mapping) -> dict[str, str]:
    ext, det = summary["extraction"], summary["detection"]
    rout, e2e, cost = summary["routing"], summary["end_to_end"], summary["cost"]
    rows = {
        "claim recall": f"{ext['claim_recall']:.3f}",
        "claim precision": f"{ext['claim_precision']:.3f}",
    }
    rows.update({f"field accuracy: {f}": f"{v:.3f}" for f, v in ext["field_accuracy"].items()})
    rows.update({
        "detection accuracy": f"{det['accuracy']:.3f}",
        "true_contradiction_recall": f"{det['true_contradiction_recall']:.3f}",
        "false_conflict_rate": f"{det['false_conflict_rate']:.3f}",
        "routing accuracy": f"{rout['routing_accuracy']:.3f}",
        "risky_auto_resolved": str(rout["risky_auto_resolved"]),
        "escalation rate": f"{rout['escalation_rate']:.3f}",
        "false escalations": str(rout["false_escalations"]),
        "wrong_certified_rate": _fmt_rate(e2e["wrong_certified"]),
        "over_withholding_rate": _fmt_rate(e2e["over_withholding"]),
        "calls (total)": str(cost["calls"]),
        "input tokens (total)": str(cost["input_tokens"]),
        "output tokens (total)": str(cost["output_tokens"]),
        "calls per case": f"{cost['per_case']['calls']:.1f}",
    })
    return rows


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _caveat(n_cases: int) -> str:
    return (
        f"Caveat: {n_cases} cases gives very wide intervals. "
        "Treat these numbers as a smoke test, not an estimate."
    )


def format_single(summary: Mapping) -> str:
    rows = [[k, v] for k, v in _flatten(summary).items()]
    return _table(["Metric", "Value"], rows) + "\n\n" + _caveat(summary["cases"])


def format_compare(with_verifier: Mapping, without_verifier: Mapping) -> str:
    left, right = _flatten(with_verifier), _flatten(without_verifier)
    rows = [[k, left[k], right[k]] for k in left]
    return (
        _table(["Metric", "With verifier", "Without verifier"], rows)
        + "\n\n"
        + _caveat(with_verifier["cases"])
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the policy-conflict pipeline evals.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="data folder (documents/*.txt + gold.json)")
    parser.add_argument("--backend", choices=["oracle", "noisy"], default="oracle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verifier", action="store_true")
    parser.add_argument("--out", default="results", help="directory for results.json")
    parser.add_argument(
        "--compare", action="store_true",
        help="run noisy with and without the verifier, side by side",
    )
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.cases))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        with_v = evaluate(cases, "noisy", args.seed, use_verifier=True)
        without_v = evaluate(cases, "noisy", args.seed, use_verifier=False)
        payload = {
            "backend": "noisy",
            "seed": args.seed,
            "with_verifier": with_v,
            "without_verifier": without_v,
        }
        print(format_compare(with_v, without_v))
    else:
        use_verifier = not args.no_verifier
        summary = evaluate(cases, args.backend, args.seed, use_verifier=use_verifier)
        payload = {
            "backend": args.backend,
            "seed": args.seed,
            "use_verifier": use_verifier,
            "summary": summary,
        }
        print(format_single(summary))

    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
