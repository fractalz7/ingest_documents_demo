"""Backend and Reviewer protocols, and the usage counter that backends share."""
from __future__ import annotations

from typing import Optional, Protocol

from pcd.schemas import (
    Claim,
    Document,
    DocumentMetadata,
    PairJudgement,
    Resolution,
    Status,
    Usage,
    Verdict,
)

# Fixed fake token counts per call, so cost numbers are reproducible offline.
FAKE_INPUT_TOKENS = 300
FAKE_OUTPUT_TOKENS = 60


class Backend(Protocol):
    name: str

    @property
    def usage(self) -> Usage: ...

    def classify(self, doc: Document) -> Optional[DocumentMetadata]:
        """Return metadata, or None to block the document."""

    def extract(self, doc: Document, meta: DocumentMetadata) -> list[Claim]: ...

    def detect(self, a: Claim, b: Claim) -> PairJudgement: ...

    def verify(self, a: Claim, b: Claim, judgement: PairJudgement) -> Verdict: ...

    def propose_text(self, a: Claim, b: Claim, resolution: Optional[Resolution]) -> str: ...


class Reviewer(Protocol):
    def decide(
        self, case_id: str, a: Claim, b: Claim, resolution: Optional[Resolution]
    ) -> dict[str, Status]:
        """Return the final status for each claim in the pair."""


class UsageMeter:
    """Counts calls and adds token counts; the default counts are fixed fakes."""

    def __init__(self) -> None:
        self._usage = Usage()

    def add(
        self,
        input_tokens: int = FAKE_INPUT_TOKENS,
        output_tokens: int = FAKE_OUTPUT_TOKENS,
    ) -> None:
        self._usage.calls += 1
        self._usage.input_tokens += input_tokens
        self._usage.output_tokens += output_tokens

    @property
    def usage(self) -> Usage:
        return self._usage.model_copy()
