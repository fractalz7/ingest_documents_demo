"""Read and write the sample cases as plain files.

Layout of a data folder:
    documents/<DOC_ID>.txt   the policy text, exactly as the pipeline ingests it
    gold.json                answer key: one entry per case (id, split, category, doc ids, gold)
    gold.csv                 flat, human-readable view of the same answer key (written, never read)
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from pcd.schemas import Case, Document

_TITLE = re.compile(r"^Title: (.*)$", re.MULTILINE)


def load_case_dir(root: Path) -> list[Case]:
    root = Path(root)
    entries = json.loads((root / "gold.json").read_text(encoding="utf-8"))
    cases = []
    for e in entries:
        docs = []
        for doc_id in e["doc_ids"]:
            text = (root / "documents" / f"{doc_id}.txt").read_text(encoding="utf-8")
            if text.endswith("\n"):
                text = text[:-1]
            m = _TITLE.search(text)
            docs.append(Document(doc_id=doc_id, title=m.group(1) if m else doc_id, text=text))
        cases.append(Case(case_id=e["case_id"], split=e["split"], category=e["category"],
                          documents=docs, gold=e["gold"]))
    return cases


def write_case_dir(cases: list[Case], root: Path) -> None:
    root = Path(root)
    (root / "documents").mkdir(parents=True, exist_ok=True)
    entries = []
    for c in cases:
        for d in c.documents:
            (root / "documents" / f"{d.doc_id}.txt").write_text(d.text + "\n", encoding="utf-8")
        entries.append({"case_id": c.case_id, "split": c.split, "category": c.category,
                        "doc_ids": [d.doc_id for d in c.documents],
                        "gold": c.gold.model_dump(mode="json")})
    (root / "gold.json").write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with open(root / "gold.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "split", "category", "claim_a", "claim_b", "true_label", "risk",
                    "expected_route", "scope_overlap", "date_overlap", "rule",
                    "final_status_a", "final_status_b"])
        for c in cases:
            for g in c.gold.pairs:
                fs = list(g.final_status.values())
                w.writerow([c.case_id, c.split, c.category, g.claim_a_id, g.claim_b_id, g.label.value,
                            g.risk.value, g.route.value, g.scope_overlap, g.date_overlap,
                            g.resolution.rule_fired if g.resolution else "", fs[0].value, fs[1].value])
