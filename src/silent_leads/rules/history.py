"""Dialogue-history findings: exact opt-out keyword, phrase hints, post-silent contact, missing context.

Every finding carries the message ids that prove it; those ids always belong to the
case being analysed because they come from its own timeline.

Only one text rule decides on its own: a client message that *is* an opt-out keyword
(STOP / UNSUBSCRIBE — the US carrier standard). Every other wording is context, judged by
the AI review or by a human; here it is only flagged as a hint.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from silent_leads.config import Config
from silent_leads.models import ContentState, Direction, Message
from silent_leads.timeline import CaseTimeline, Speaker, Utterance


class FindingKind(StrEnum):
    EXACT_OPT_OUT = "exact_opt_out"  # whole client message is STOP/UNSUBSCRIBE -> do_not_contact
    PHRASE_HINT = "phrase_hint"  # key phrase in client/unknown text -> context must be judged
    CONTACT_AFTER_SILENT = "contact_after_silent"  # any event after silent_since -> review
    HIDDEN_INBOUND = "hidden_inbound"  # client message with withheld text -> review
    NO_CLIENT_TEXT = "no_client_text"  # nothing the client said is readable -> review
    INBOUND_WITHOUT_TEXT = "inbound_without_text"  # client event with no readable text -> review
    UNKNOWN_SPEAKER = "unknown_speaker"  # transcript line with undetermined author -> review
    HIDDEN_OUTBOUND = "hidden_outbound"  # informational only


@dataclass(frozen=True)
class Finding:
    kind: FindingKind
    message_ids: tuple[str, ...]
    detail: str


_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})
_EDGE_PUNCTUATION = " \t\r\n.!?,;:\"'()[]«»"


def _normalize(text: str) -> str:
    """NFKC (full-width 'ＳＴＯＰ' -> 'STOP'), drop invisible format characters, lower, collapse spaces."""
    folded = unicodedata.normalize("NFKC", text)
    visible = "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")
    return re.sub(r"\s+", " ", visible.translate(_APOSTROPHES).lower()).strip()


def normalize_whole_message(text: str) -> str:
    """Lower-case the whole text and drop surrounding punctuation: 'STOP.' -> 'stop'."""
    return _normalize(text).strip(_EDGE_PUNCTUATION)


def is_exact_opt_out(message: Message, keywords: tuple[str, ...]) -> bool:
    """The rule: an inbound message whose entire text is one opt-out keyword."""
    if message.direction != Direction.INBOUND or not message.has_text:
        return False
    assert message.text is not None
    normalized = {normalize_whole_message(k) for k in keywords} - {""}
    return normalize_whole_message(message.text) in normalized


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


def has_client_text(timeline: CaseTimeline) -> bool:
    return any(u.speaker == Speaker.CLIENT and u.text.strip() for u in timeline.utterances())


def analyse_history(timeline: CaseTimeline, config: Config) -> list[Finding]:
    findings: list[Finding] = []
    utterances = timeline.utterances()

    exact = [m for m in timeline.messages if is_exact_opt_out(m, config.phrases.exact_opt_out)]
    if exact:
        words = sorted({normalize_whole_message(m.text or "").upper() for m in exact})
        findings.append(
            Finding(
                FindingKind.EXACT_OPT_OUT,
                tuple(m.message_id for m in exact),
                f"офіційна відписка: повідомлення клієнта складається з ключового слова {', '.join(words)}",
            )
        )

    hint_hits = _match_phrases(utterances, _compile(config.phrases.hints))
    client_pairs = hint_hits[Speaker.CLIENT]
    unknown_pairs = hint_hits[Speaker.UNKNOWN]
    if (client_pairs or unknown_pairs) and config.review.phrase_hint:
        who = []
        if client_pairs:
            who.append("клієнт")
        if unknown_pairs:
            who.append("невизначений автор")
        findings.append(
            Finding(
                FindingKind.PHRASE_HINT,
                _unique_ids(client_pairs + unknown_pairs),
                f"фраза потребує оцінки контексту: {_phrases_text(client_pairs + unknown_pairs)} "
                f"({', '.join(who)})",
            )
        )

    after = timeline.after_silent
    if after and config.review.contact_after_silent:
        who_dir = sorted({m.direction.value if m.direction else "?" for m in after})
        findings.append(
            Finding(
                FindingKind.CONTACT_AFTER_SILENT,
                tuple(m.message_id for m in after),
                f"{len(after)} под. після переходу в silent ({', '.join(who_dir)}) — "
                "статус silent може бути неактуальним",
            )
        )

    # Anything that is not clearly outbound counts as the client's side: an unknown direction
    # must not make a hidden message disappear from the review.
    hidden_in = [
        m
        for m in timeline.messages
        if m.content_state == ContentState.WITHHELD_HIDDEN and m.direction != Direction.OUTBOUND
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

    no_text_in = [
        m
        for m in timeline.messages
        if not m.has_text
        and m.content_state != ContentState.WITHHELD_HIDDEN
        and m.direction != Direction.OUTBOUND
    ]
    if no_text_in and config.review.inbound_without_text:
        findings.append(
            Finding(
                FindingKind.INBOUND_WITHOUT_TEXT,
                tuple(m.message_id for m in no_text_in),
                f"{len(no_text_in)} вхідн. подій клієнта без читабельного тексту (дзвінок без транскрипту, "
                "порожній текст) — контекст неповний",
            )
        )

    unknown = [u for u in utterances if u.speaker == Speaker.UNKNOWN and u.text.strip()]
    if unknown and config.review.unknown_speaker:
        findings.append(
            Finding(
                FindingKind.UNKNOWN_SPEAKER,
                _unique_ids([(u.message_id, "") for u in unknown]),
                f"{len(unknown)} реплік з невизначеним автором — не можна відрізнити клієнта від сейлза",
            )
        )

    if not has_client_text(timeline) and config.review.no_client_text:
        inbound_ids = tuple(m.message_id for m in timeline.messages if m.direction == Direction.INBOUND)
        findings.append(
            Finding(
                FindingKind.NO_CLIENT_TEXT,
                inbound_ids,
                "жодного читабельного тексту від клієнта — відсутній контекст не є дозволом",
            )
        )
    return findings
