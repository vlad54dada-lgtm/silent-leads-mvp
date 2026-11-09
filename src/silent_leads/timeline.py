"""Per-case chronological history and speaker attribution inside transcripts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from silent_leads.models import Direction, Lead, Message

# "Kathleen (C): Hello?"  /  "[СПІВРОБІТНИК] (S): Hello, ..."
_SPEAKER_LINE = re.compile(r"^\s*(?P<name>[^\n:]*?)\s*\((?P<tag>[CS])\)\s*:\s*(?P<text>.*)$")


class Speaker(StrEnum):
    CLIENT = "client"
    SALES = "sales"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Utterance:
    message_id: str
    speaker: Speaker
    text: str
    occurred_at: datetime | None


def split_utterances(message: Message) -> list[Utterance]:
    """Attribute each line of a message to a speaker.

    Call transcripts carry ``(C)`` / ``(S)`` tags per line; those win over the channel
    ``direction``. A line without a tag inside a transcript that otherwise has tags is
    ``UNKNOWN`` — it may be a quote, a system line or a mis-split fragment.
    SMS/MMS text has no tags: the whole text belongs to the channel direction.
    """
    if not message.has_text:
        return []
    assert message.text is not None  # for the type checker; has_text guarantees it
    lines = [line for line in message.text.splitlines() if line.strip()]
    tagged = [(_SPEAKER_LINE.match(line), line) for line in lines]
    has_tags = any(m is not None for m, _ in tagged)

    if not has_tags:
        speaker = _speaker_from_direction(message.direction)
        return [Utterance(message.message_id, speaker, message.text.strip(), message.occurred_at_crm)]

    result: list[Utterance] = []
    for match, line in tagged:
        if match is None:
            result.append(
                Utterance(message.message_id, Speaker.UNKNOWN, line.strip(), message.occurred_at_crm)
            )
            continue
        speaker = Speaker.CLIENT if match.group("tag") == "C" else Speaker.SALES
        result.append(
            Utterance(message.message_id, speaker, match.group("text").strip(), message.occurred_at_crm)
        )
    return result


def _speaker_from_direction(direction: Direction | None) -> Speaker:
    if direction == Direction.INBOUND:
        return Speaker.CLIENT
    if direction == Direction.OUTBOUND:
        return Speaker.SALES
    return Speaker.UNKNOWN


@dataclass(frozen=True)
class CaseTimeline:
    lead: Lead
    messages: tuple[Message, ...]  # chronological; undated messages last, by id

    @property
    def after_silent(self) -> tuple[Message, ...]:
        """Events strictly after the CRM silent timestamp (same CRM-local clock)."""
        since = self.lead.silent_since_crm
        if since is None:
            return ()
        return tuple(m for m in self.messages if m.occurred_at_crm is not None and m.occurred_at_crm > since)

    @property
    def undated(self) -> tuple[Message, ...]:
        return tuple(m for m in self.messages if m.occurred_at_crm is None)

    def utterances(self) -> list[Utterance]:
        return [u for m in self.messages for u in split_utterances(m)]


def _sort_key(message: Message) -> tuple[int, datetime, str]:
    if message.occurred_at_crm is None:
        return (1, datetime.max, message.message_id)
    return (0, message.occurred_at_crm, message.message_id)


def build_timelines(leads: list[Lead], messages: list[Message]) -> dict[str, CaseTimeline]:
    """One timeline per lead, keyed by case_id, in deterministic order."""
    by_case: dict[str, list[Message]] = {lead.case_id: [] for lead in leads}
    for message in messages:
        if message.case_id in by_case:
            by_case[message.case_id].append(message)
    return {
        lead.case_id: CaseTimeline(lead=lead, messages=tuple(sorted(by_case[lead.case_id], key=_sort_key)))
        for lead in sorted(leads, key=lambda lead: lead.case_id)
    }
