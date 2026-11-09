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
    """Phrase lists applied to what the client wrote (case-insensitive).

    ``exact_opt_out`` is a RULE: a client message whose *whole* text is one of these words
    (the US carrier opt-out keywords) blocks contact regardless of due date or context.
    ``hints`` are NOT a rule: a hit only says the context must be judged — by the AI review
    when it is enabled, otherwise by a human (``manual_review``). 'stop by our office' is a
    hint that a human or the model will clear; the bare word 'STOP' is the rule.
    """

    model_config = ConfigDict(frozen=True)

    exact_opt_out: tuple[str, ...] = ("stop", "unsubscribe")
    hints: tuple[str, ...] = ()


class ReviewTriggers(BaseModel):
    """Which situations send a case to manual_review. All default to the safe side."""

    model_config = ConfigDict(frozen=True)

    contact_after_silent: bool = True
    hidden_inbound: bool = True
    no_client_text: bool = True
    # An inbound event with no readable text that is not marked withheld (a call without a
    # transcript, an empty text, an unknown content_state): the client said something we cannot read.
    inbound_without_text: bool = True
    # A transcript line whose author could not be determined (no (C)/(S) marker): it may be the
    # client refusing — never a positive decision without a human.
    unknown_speaker: bool = True
    # A phrase hint in a client/unknown-speaker utterance: manual_review when the AI review is
    # off; with the AI review on, the model judges the context and the hint is only noted.
    phrase_hint: bool = True


class AiReview(BaseModel):
    """Context analysis of the dialogue by a Claude model.

    Off by default. Requires coordinator approval (recorded in docs/limitations.md).
    With it on, every case that has readable client text (and no exact STOP) is sent to the
    model, which labels the client's intent with a confidence. Only a high-confidence refusal
    with evidence blocks contact; anything less certain goes to manual_review. The model never
    makes a case a candidate by itself — due date and the other triggers still apply.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    model: str = "claude-sonnet-5-5"
    max_input_chars: int = Field(default=20000, ge=500)
    # None -> api.anthropic.com. An Anthropic-compatible gateway (e.g. https://openrouter.ai/api)
    # adds a third party that sees client texts; record its approval in docs/limitations.md.
    base_url: str | None = None
    # Name of the environment variable holding the API key, never the key itself.
    api_key_env: str = "ANTHROPIC_API_KEY"

    @field_validator("base_url")
    @classmethod
    def _base_url_is_https(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().rstrip("/")
        if not value.startswith("https://"):
            msg = "ai_review.base_url must start with https://"
            raise ValueError(msg)
        return value

    @field_validator("api_key_env")
    @classmethod
    def _api_key_env_is_a_name(cls, value: str) -> str:
        value = value.strip()
        if not value or not all(ch.isalnum() or ch == "_" for ch in value):
            msg = "ai_review.api_key_env must be an environment variable NAME (letters, digits, _)"
            raise ValueError(msg)
        if value.lower().startswith("sk-") or len(value) > 64:
            msg = "ai_review.api_key_env looks like a key value; put the key in the environment, not here"
            raise ValueError(msg)
        return value


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
