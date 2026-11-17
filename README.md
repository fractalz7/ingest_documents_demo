# Policy-conflict ingestion demo

A small, offline demo of the ingestion design. Agents propose and verify; deterministic
code (`pcd/rules.py`, `pcd/pipeline.py`) decides. An eval harness scores the pipeline
against synthetic cases with known gold labels. All data is synthetic.

See `SPEC.md` for the design and acceptance criteria.

## Setup

    pip install -r requirements.txt
    python3 -m pytest -q

Tests need no network and no CrewAI.

## Run the evals

    python -m pcd.evals.run_evals --cases data --backend oracle
    python -m pcd.evals.run_evals --cases data --backend noisy --seed 0
    python -m pcd.evals.run_evals --cases data --compare

Each run prints a markdown table and writes `results/results.json` (change with `--out`).
Pass `--no-verifier` to disable the verifier stage.

## Generate the sample cases

    python -m pcd.data.generate_cases --out data --seed 7

## Layout

- `pcd/schemas.py`: data contract (read-only)
- `pcd/rules.py`: rules engine (R1 to R4, checks, route)
- `pcd/pipeline.py`: `run_case` for one case
- `pcd/backends/`: protocols, oracle and noisy stubs, optional CrewAI adapter (UNTESTED)
- `pcd/evals/`: metrics and CLI
