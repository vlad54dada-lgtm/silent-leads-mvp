"""Domain models: leads, communication events, data issues, decisions."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Decision(StrEnum):
    """Exactly the four outcomes from the sprint spec."""

    CONTACT_CANDIDATE = "contact_candidate"
    SKIP_NOT_DUE = "skip_not_due"
    DO_NOT_CONTACT = "do_not_contact"
    MANUAL_REVIEW = "manual_review"


# Higher number wins when several rules fire for one case.
DECISION_PRIORITY: dict[Decision, int] = {
    Decision.DO_NOT_CONTACT: 3,
    Decision.MANUAL_REVIEW: 2,
    Decision.SKIP_NOT_DUE: 1,
    Decision.CONTACT_CANDIDATE: 0,
}


class TemplateStatus(StrEnum):
    TEMPLATE_MISSING = "template_missing"
    TEST_TEMPLATE = "test_template"
    APPROVED_TEMPLATE = "approved_template"
    NOT_APPLICABLE = "not_applicable"


class ContentState(StrEnum):
    TEXT_AVAILABLE = "text_available"
    WITHHELD_HIDDEN = "withheld_hidden"
    CALL_WITHOUT_TEXT = "call_without_text"
    EMPTY_TEXT = "empty_text"


class Direction(StrEnum):
    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"


class Role(StrEnum):
    CLIENT = "client"
    SALES = "sales"


class IssueLevel(StrEnum):
    ERROR = "error"  # decision for the case cannot be trusted -> manual_review
    WARNING = "warning"  # recorded, does not change the decision by itself


class DataIssue(BaseModel):
    """One explicit data problem. ``case_id`` is None for file-level issues."""

    model_config = ConfigDict(frozen=True)

    level: IssueLevel
    code: str
    detail: str
    case_id: str | None = None
    message_id: str | None = None


class Lead(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    team_alias: str | None
    sales_alias: str | None
    status: str | None
    created_at_crm: datetime | None
    silent_since_crm: datetime | None
    has_quote: bool | None
    quote_at_crm: datetime | None
    last_inbound_at_crm: datetime | None
    last_outbound_at_crm: datetime | None
    events_count: int | None
    text_events: int | None
    hidden_events: int | None
    calls_without_text: int | None
    client_timezone: str | None


class Message(BaseModel):
    """One communication event. Multi-part rows with the same message_id are merged."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    message_id: str
    occurred_at_crm: datetime | None
    event_type: str | None
    direction: Direction | None
    role: Role | None
    call_result: str | None
    call_seconds: int | None
    content_state: ContentState | None
    text: str | None
    parts_expected: int = 1
    parts_found: int = 1

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())


class CaseResult(BaseModel):
    """One output row — column names match the «Результат» sheet exactly."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    decision: Decision
    reason: str
    evidence_message_ids: tuple[str, ...]
    draft_sms: str
    template_status: TemplateStatus
    reviewer_note: str
