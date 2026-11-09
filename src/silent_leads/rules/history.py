"""Dialogue-history findings: opt-outs, negatives, post-silent contact, missing context.

Every finding carries the message ids that prove it; those ids always belong to the
case being analysed because they come from its own timeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from silent_leads.config import Config
from silent_leads.models import ContentState, Direction
from silent_leads.timeline import CaseTimeline, Speaker, Utterance


class FindingKind(StrEnum):
    OPT_OUT = "opt_out"  # client asked not to be contacted -> do_not_contact
    NEGATIVE = "negative"  # unambiguous client negative -> do_not_contact
    PHRASE_UNKNOWN_SPEAKER = "phrase_unknown_speaker"  # key phrase, speaker unclear -> review
    CONTACT_AFTER_SILENT = "contact_after_silent"  # any event after silent_since -> review
    HIDDEN_INBOUND = "hidden_inbound"  # client message with withheld text -> review
    NO_CLIENT_TEXT = "no_client_text"  # nothing the client said is readable -> review
    HIDDEN_OUTBOUND = "hidden_outbound"  # informational only


@dataclass(frozen=True)
class Finding:
    kind: FindingKind
    message_ids: tuple[str, ...]
    detail: str


_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_APOSTROPHES).lower()).strip()


def _compile(phrases: tuple[str, ...]) -> list[tuple[str, re.Pattern[str]]]:
    """Whole-phrase match with word boundaries, so 'stop' does not hit 'nonstop'."""
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for phrase in phrases:
        norm = _normalize(phrase)
        if not norm:
            continue
        compiled.append((phrase, re.compile(r"(?<!\w)" + re.escape(norm) + r"(?!\w)")))
    return compiled


def _match_phrases(
    utterances: list[Utterance], patterns: list[tuple[str, re.Pattern[str]]]
) -> dict[Speaker, list[tuple[str, str]]]:
    """Return {speaker: [(message_id, phrase), ...]} for every phrase hit."""
    hits: dict[Speaker, list[tuple[str, str]]] = {Speaker.CLIENT: [], Speaker.SALES: [], Speaker.UNKNOWN: []}
    for utterance in utterances:
        text = _normalize(utterance.text)
        for phrase, pattern in patterns:
            if pattern.search(text):
                hits[utterance.speaker].append((utterance.message_id, phrase))
    return hits


def _unique_ids(pairs: list[tuple[str, str]]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for message_id, _ in pairs:
        seen.setdefault(message_id, None)
    return tuple(seen)


def _phrases_text(pairs: list[tuple[str, str]]) -> str:
    phrases = sorted({phrase for _, phrase in pairs})
    return ", ".join(f"'{p}'" for p in phrases)


def analyse_history(timeline: CaseTimeline, config: Config) -> list[Finding]:
    findings: list[Finding] = []
    utterances = timeline.utterances()
    opt_out = _compile(config.phrases.opt_out)
    negative = _compile(config.phrases.negative)

    opt_hits = _match_phrases(utterances, opt_out)
    neg_hits = _match_phrases(utterances, negative)

    if opt_hits[Speaker.CLIENT]:
        findings.append(
            Finding(
                FindingKind.OPT_OUT,
                _unique_ids(opt_hits[Speaker.CLIENT]),
                f"клієнт просив припинити контакт: {_phrases_text(opt_hits[Speaker.CLIENT])}",
            )
        )
    if neg_hits[Speaker.CLIENT]:
        findings.append(
            Finding(
                FindingKind.NEGATIVE,
                _unique_ids(neg_hits[Speaker.CLIENT]),
                f"однозначний негатив від клієнта: {_phrases_text(neg_hits[Speaker.CLIENT])}",
            )
        )
    unknown_pairs = opt_hits[Speaker.UNKNOWN] + neg_hits[Speaker.UNKNOWN]
    if unknown_pairs and config.review.unknown_speaker_phrase:
        findings.append(
            Finding(
                FindingKind.PHRASE_UNKNOWN_SPEAKER,
                _unique_ids(unknown_pairs),
                f"ключова фраза в репліці з невизначеним автором: {_phrases_text(unknown_pairs)}",
            )
        )

    after = timeline.after_silent
    if after and config.review.contact_after_silent:
        who = sorted({m.direction.value if m.direction else "?" for m in after})
        findings.append(
            Finding(
                FindingKind.CONTACT_AFTER_SILENT,
                tuple(m.message_id for m in after),
                f"{len(after)} под. після переходу в silent ({', '.join(who)}) — "
                "статус silent може бути неактуальним",
            )
        )

    hidden_in = [
        m
        for m in timeline.messages
        if m.content_state == ContentState.WITHHELD_HIDDEN and m.direction == Direction.INBOUND
    ]
    if hidden_in and config.review.hidden_inbound:
        findings.append(
            Finding(
                FindingKind.HIDDEN_INBOUND,
                tuple(m.message_id for m in hidden_in),
                f"{len(hidden_in)} вхідн. повідомлень клієнта з прихованим текстом — контекст неповний",
            )
        )
    hidden_out = [
        m
        for m in timeline.messages
        if m.content_state == ContentState.WITHHELD_HIDDEN and m.direction == Direction.OUTBOUND
    ]
    if hidden_out:
        findings.append(
            Finding(
                FindingKind.HIDDEN_OUTBOUND,
                tuple(m.message_id for m in hidden_out),
                f"{len(hidden_out)} вихідн. повідомлень з прихованим текстом",
            )
        )

    client_said_something = any(u.speaker == Speaker.CLIENT and u.text for u in utterances)
    if not client_said_something and config.review.no_client_text:
        inbound_ids = tuple(m.message_id for m in timeline.messages if m.direction == Direction.INBOUND)
        findings.append(
            Finding(
                FindingKind.NO_CLIENT_TEXT,
                inbound_ids,
                "жодного читабельного тексту від клієнта — відсутній контекст не є дозволом",
            )
        )
    return findings
