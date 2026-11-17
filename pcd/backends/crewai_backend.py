"""CrewAI adapter for the Backend protocol.

UNTESTED: not exercised by the test suite. Written from the CrewAI docs
(https://docs.crewai.com) without a live run. Each protocol method is one agent
task with a Pydantic output model. Routing stays in pcd/pipeline.py, so a bad
model output can at worst send a pair to a human.

Environment: DEMO_LLM_MODEL names the model. The provider SDK reads its own key
from the environment. No key is set or read in this file.
"""
from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel

from pcd.backends.base import UsageMeter
from pcd.schemas import (
    TOPICS,
    Claim,
    Document,
    DocumentMetadata,
    PairJudgement,
    Resolution,
    Usage,
    Verdict,
)

try:  # optional dependency; the module still imports without it
    from crewai import Agent, Crew, Task  # TODO(verify): import path for the pinned release
except ImportError:
    Agent = Crew = Task = None


class _ExtractOut(BaseModel):
    claims: list[Claim]


class _TextOut(BaseModel):
    text: str


class CrewAIBackend:
    name = "crewai"

    def __init__(self, model: Optional[str] = None) -> None:
        if Agent is None:
            raise ImportError("crewai is not installed; it is an optional dependency")
        self.model = model or os.environ.get("DEMO_LLM_MODEL")
        if not self.model:
            raise RuntimeError("Set DEMO_LLM_MODEL to the model name to use")
        self._meter = UsageMeter()

    @property
    def usage(self) -> Usage:
        return self._meter.usage

    def _ask(self, role: str, instruction: str, output_model: type[BaseModel]):
        # TODO(verify): Agent(llm=...) accepts a model string in the pinned release
        agent = Agent(
            role=role,
            goal=instruction,
            backstory="A careful policy analyst. Use only the text you are given.",
            llm=self.model,
            allow_delegation=False,
        )
        # TODO(verify): output_pydantic and the result's structured-output attribute
        task = Task(
            description=instruction,
            expected_output="Output matching the required schema.",
            agent=agent,
            output_pydantic=output_model,
        )
        result = Crew(agents=[agent], tasks=[task]).kickoff()
        self._meter.add()  # TODO(verify): read real token counts from the result instead
        return result.pydantic

    def classify(self, doc: Document) -> Optional[DocumentMetadata]:
        # TODO(verify): block non-policy documents, e.g. with a separate output model
        return self._ask(
            "Document classifier",
            "Classify this document's owner, authority, effective dates and "
            f"supersession.\n\n{doc.text}",
            DocumentMetadata,
        )

    def extract(self, doc: Document, meta: DocumentMetadata) -> list[Claim]:
        instruction = (
            f"Extract claims using only these topic ids: {', '.join(TOPICS)}. "
            "Booleans are required or not_required, integers are decimal strings, "
            "money is USD:<int>.\n\n" + doc.text
        )
        out = self._ask("Claim extractor", instruction, _ExtractOut)
        # Ids are set here so the adapter cannot emit a claim id that breaks the contract.
        return [
            c.model_copy(update={"doc_id": doc.doc_id, "claim_id": f"{doc.doc_id}:{c.topic}"})
            for c in out.claims
        ]

    def detect(self, a: Claim, b: Claim) -> PairJudgement:
        instruction = (
            "Label this pair of claims with one conflict label.\n"
            f"A: {a.model_dump_json()}\nB: {b.model_dump_json()}"
        )
        return self._ask("Conflict detector", instruction, PairJudgement)

    def verify(self, a: Claim, b: Claim, judgement: PairJudgement) -> Verdict:
        instruction = (
            f"Check whether the label {judgement.label.value} fits this pair. "
            f"A: {a.model_dump_json()}\nB: {b.model_dump_json()}"
        )
        return self._ask("Label verifier", instruction, Verdict)

    def propose_text(self, a: Claim, b: Claim, resolution: Optional[Resolution]) -> str:
        instruction = (
            "Write a two-sentence proposal for a policy owner. "
            f"A: {a.model_dump_json()}\nB: {b.model_dump_json()}\n"
            f"Deterministic resolution: {resolution.model_dump_json() if resolution else 'none'}"
        )
        return self._ask("Proposal writer", instruction, _TextOut).text
