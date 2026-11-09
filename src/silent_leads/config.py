"""Run configuration loaded from YAML. Every tunable number or phrase lives here."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from silent_leads.models import TemplateStatus


class SheetNames(BaseModel):
    model_config = ConfigDict(frozen=True)

    leads: str = "Ліди"
    messages: str = "Повідомлення"
    instruction: str = "Інструкція"
    control: str = "Контроль"


class Template(BaseModel):
    """A message template. ``status`` says whether the text may be treated as approved."""

    model_config = ConfigDict(frozen=True)

    text: str = ""
    status: TemplateStatus = TemplateStatus.TEMPLATE_MISSING

    @field_validator("status")
    @classmethod
    def _status_matches_text(cls, status: TemplateStatus) -> TemplateStatus:
        if status == TemplateStatus.NOT_APPLICABLE:
            msg = "template status 'not_applicable' is only produced by the tool, not configured"
            raise ValueError(msg)
        return status


class Templates(BaseModel):
    model_config = ConfigDict(frozen=True)

    with_quote: Template = Template()
    without_quote: Template = Template()


class Phrases(BaseModel):
    """Lower-cased phrase lists matched against client utterances."""

    model_config = ConfigDict(frozen=True)

    opt_out: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()


class ReviewTriggers(BaseModel):
    """Which situations send a case to manual_review. All default to the safe side."""

    model_config = ConfigDict(frozen=True)

    contact_after_silent: bool = True
    hidden_inbound: bool = True
    no_client_text: bool = True
    unknown_speaker_phrase: bool = True


class AiReview(BaseModel):
    """Optional second-opinion pass over contact candidates by a Claude model.

    Off by default. Requires coordinator approval (recorded in docs/limitations.md).
    The model sees only the client's own utterances and can only move a case
    from contact_candidate to manual_review — never the other way.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    model: str = "claude-opus-5-5"
    max_input_chars: int = Field(default=20000, ge=500)


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    first_touch_delay_days: int = Field(default=7, ge=0)
    crm_timezone: str = "America/Los_Angeles"
    snapshot_utc: datetime | None = None
    sheets: SheetNames = SheetNames()
    templates: Templates = Templates()
    phrases: Phrases = Phrases()
    review: ReviewTriggers = ReviewTriggers()
    ai_review: AiReview = AiReview()

    @field_validator("crm_timezone")
    @classmethod
    def _tz_exists(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            msg = f"unknown timezone: {value}"
            raise ValueError(msg) from exc
        return value

    @field_validator("snapshot_utc")
    @classmethod
    def _snapshot_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            msg = "snapshot_utc must carry a timezone, e.g. 2026-11-05T14:53:41Z"
            raise ValueError(msg)
        return value

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.crm_timezone)


def load_config(path: Path) -> Config:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        msg = f"config root must be a mapping: {path}"
        raise ValueError(msg)
    return Config.model_validate(raw)
