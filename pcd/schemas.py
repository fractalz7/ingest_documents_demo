"""Shared data contract for the policy-conflict demo. AUTHORITATIVE: do not edit.

All content in this project is synthetic. Owners are generic (PAYER, CARE_DELIVERY);
nothing here describes real Humana or CenterWell policy.
"""
from __future__ import annotations

from datetime import date
from enum import Enum, IntEnum
from typing import Optional

from pydantic import BaseModel, Field

ANY = "ALL"  # wildcard for a scope dimension


class Owner(str, Enum):
    PAYER = "PAYER"
    CARE_DELIVERY = "CARE_DELIVERY"


class Authority(IntEnum):
    CLINIC_PROCEDURE = 1
    SOP = 2
    CORPORATE_POLICY = 3
    CONTRACT_EOC = 4
    REGULATION = 5


class Risk(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class ConflictLabel(str, Enum):
    TRUE_CONTRADICTION = "TRUE_CONTRADICTION"
    SCOPE_DIFFERENCE = "SCOPE_DIFFERENCE"
    OUTDATED_VERSION = "OUTDATED_VERSION"
    AMBIGUITY_TIE = "AMBIGUITY_TIE"
    NO_CONFLICT = "NO_CONFLICT"


NONCONFLICT_LABELS = {ConflictLabel.SCOPE_DIFFERENCE, ConflictLabel.NO_CONFLICT}


class Route(str, Enum):
    AUTO = "AUTO"    # rules engine resolved it
    HUMAN = "HUMAN"  # sent to a policy owner
    NONE = "NONE"    # no conflict raised


class Status(str, Enum):
    CERTIFIED = "CERTIFIED"
    SCOPED = "SCOPED"
    WITHHELD = "WITHHELD"


# Controlled topic vocabulary. Extractors must use these exact topic ids.
TOPICS: dict[str, Risk] = {
    "specialist_referral": Risk.HIGH,
    "copay_primary_care": Risk.HIGH,
    "appeal_filing_deadline_days": Risk.HIGH,
    "call_summary_retention_days": Risk.LOW,
    "callback_window_hours": Risk.LOW,
    "document_request_turnaround_days": Risk.LOW,
    "interpreter_scheduling_lead_hours": Risk.LOW,
}

# Normalized value formats (the extractor must emit these, never free text):
#   booleans -> "required" | "not_required"
#   integers -> decimal string, e.g. "30"
#   money    -> "USD:<int>", e.g. "USD:10"


class Scope(BaseModel):
    plan: str = ANY      # e.g. "NW-Gold", "NW-Silver" or ANY
    state: str = ANY     # two-letter code or ANY
    setting: str = ANY   # "member", "clinic" or ANY


class DocumentMetadata(BaseModel):
    owner: Owner
    authority: Authority
    effective_start: date
    effective_end: Optional[date] = None
    supersedes_doc_id: Optional[str] = None


class Document(BaseModel):
    doc_id: str
    title: str
    text: str  # raw text; first lines are a header carrying the metadata


class Claim(BaseModel):
    claim_id: str  # f"{doc_id}:{topic}"
    doc_id: str
    topic: str     # key of TOPICS
    value: str     # normalized, see formats above
    scope: Scope
    effective_start: date
    effective_end: Optional[date] = None
    authority: Authority
    owner: Owner
    supersedes_doc_id: Optional[str] = None
    source_text: str = ""


class PairJudgement(BaseModel):
    label: ConflictLabel
    rationale: str = ""


class Verdict(BaseModel):
    agrees: bool
    corrected_label: Optional[ConflictLabel] = None
    rationale: str = ""


class Resolution(BaseModel):
    """Produced by the deterministic rules engine only. Never by an agent."""
    winner_claim_id: str
    loser_claim_id: str
    rule_fired: str  # R1_SUPERSEDES | R2_AUTHORITY | R3_SCOPE | R4_DATE
    loser_disposition: Status  # SCOPED or WITHHELD


class Checks(BaseModel):
    rule_applies: bool
    low_risk: bool
    no_cross_owner_tie: bool


class PairResult(BaseModel):
    claim_a_id: str
    claim_b_id: str
    predicted_label: ConflictLabel
    scope_overlap: bool
    date_overlap: bool
    values_differ: bool
    inconsistent: bool = False       # agent label contradicts the computed facts
    verifier_disputed: bool = False
    checks: Optional[Checks] = None
    route: Route
    resolution: Optional[Resolution] = None
    proposal_text: str = ""
    final_status: dict[str, Status] = Field(default_factory=dict)  # claim_id -> status for this pair


class Usage(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class PipelineResult(BaseModel):
    case_id: str
    blocked_docs: list[str] = Field(default_factory=list)
    metadata: dict[str, DocumentMetadata] = Field(default_factory=dict)  # doc_id -> metadata
    claims: list[Claim] = Field(default_factory=list)
    pairs: list[PairResult] = Field(default_factory=list)
    final_status: dict[str, Status] = Field(default_factory=dict)  # claim_id -> most restrictive status
    usage: Usage = Field(default_factory=Usage)


class GoldPair(BaseModel):
    claim_a_id: str
    claim_b_id: str
    label: ConflictLabel
    route: Route
    risk: Risk
    scope_overlap: bool
    date_overlap: bool
    resolution: Optional[Resolution] = None  # None for non-conflicts and for ties
    final_status: dict[str, Status]


class Gold(BaseModel):
    metadata: dict[str, DocumentMetadata]
    claims: list[Claim]
    pairs: list[GoldPair]
    final_status: dict[str, Status]


class Case(BaseModel):
    case_id: str
    split: str  # "dev" or "test"
    category: str  # human-readable, e.g. "TRUE_CONTRADICTION high-risk"
    documents: list[Document]
    gold: Gold
