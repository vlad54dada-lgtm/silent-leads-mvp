from __future__ import annotations

from datetime import date, datetime

import pytest

from silent_leads.config import Config, Phrases, ReviewTriggers
from silent_leads.models import ContentState, DataIssue, Decision, Direction, IssueLevel, Message, Role
from silent_leads.rules.ai_review import AiConfidence, AiLabel, AiOutcome
from silent_leads.rules.decide import EvidenceLeakError, Verdict, decide
from silent_leads.rules.due_date import check_due
from silent_leads.rules.history import FindingKind, analyse_history, is_exact_opt_out, normalize_whole_message
from silent_leads.timeline import CaseTimeline, build_timelines
from tests.test_timeline import _lead, _msg

CONTROL = date(2026, 11, 5)
CONFIG = Config(
    phrases=Phrases(
        exact_opt_out=("stop", "unsubscribe"),
        hints=("stop", "don't call", "do not text", "remove me", "not interested", "wrong number"),
    )
)
DUE_SILENT = "2026-10-20 10:00:00"  # 16 days before control -> due
NOT_DUE_SILENT = "2026-11-03 10:00:00"  # 2 days -> not due


def _hidden(message_id: str, occurred: str, direction: Direction) -> Message:
    return Message(
        case_id="L1",
        message_id=message_id,
        occurred_at_crm=datetime.fromisoformat(occurred),
        event_type="SMS",
        direction=direction,
        role=Role.CLIENT if direction == Direction.INBOUND else Role.SALES,
        call_result=None,
        call_seconds=None,
        content_state=ContentState.WITHHELD_HIDDEN,
        text=None,
    )


def _timeline(messages: list[Message], silent: str | None = DUE_SILENT) -> CaseTimeline:
    return build_timelines([_lead(silent=silent)], messages)["L1"]


def _verdict(
    timeline: CaseTimeline,
    issues: list[DataIssue] | None = None,
    config: Config = CONFIG,
    ai: AiOutcome | None = None,
) -> Verdict:
    due = check_due(timeline.lead, CONTROL, config.first_touch_delay_days)
    return decide(timeline, due, analyse_history(timeline, config), issues or [], ai)


def _ai(
    label: AiLabel,
    confidence: AiConfidence = AiConfidence.HIGH,
    evidence: tuple[str, ...] = (),
    error: str | None = None,
) -> AiOutcome:
    return AiOutcome(
        model="test-model",
        label=None if error else label,
        confidence=None if error else confidence,
        evidence_message_ids=evidence,
        explanation="" if error else "пояснення",
        error=error,
    )


CLIENT_OK = _msg("M1", "2026-10-02 09:00:00", direction=Direction.INBOUND, text="Yes, send me the photos")
SALES_HELLO = _msg("M0", "2026-10-01 09:00:00", text="Hello, following up")


def test_clean_due_case_is_contact_candidate() -> None:
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK]))
    assert verdict.decision == Decision.CONTACT_CANDIDATE
    assert verdict.evidence_message_ids == ()
    assert "настав" in verdict.reason


def test_clean_not_due_case_is_skip_not_due() -> None:
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK], silent=NOT_DUE_SILENT))
    assert verdict.decision == Decision.SKIP_NOT_DUE
    assert "2026-11-10" in verdict.reason  # earliest contact date


# --- the one text rule: the whole client message is STOP / UNSUBSCRIBE ---


@pytest.mark.parametrize("text", ["STOP", "stop", "Stop.", " STOP! ", "Unsubscribe", "'stop'"])
def test_exact_opt_out_keyword_blocks_regardless_of_due_date(text: str) -> None:
    keyword = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text=text)
    for silent in (DUE_SILENT, NOT_DUE_SILENT):
        verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK, keyword], silent=silent))
        assert verdict.decision == Decision.DO_NOT_CONTACT
        assert verdict.evidence_message_ids == ("M2",)
        assert "офіційна відписка" in verdict.reason


def test_exact_opt_out_found_in_early_message_not_only_last() -> None:
    early = _msg("M2", "2026-09-15 09:00:00", direction=Direction.INBOUND, text="STOP")
    later = _msg("M3", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="ok")
    verdict = _verdict(_timeline([early, SALES_HELLO, later]))
    assert verdict.decision == Decision.DO_NOT_CONTACT
    assert verdict.evidence_message_ids == ("M2",)


@pytest.mark.parametrize(
    "text",
    ["Please STOP texting me.", "stop by our office", "Reply STOP to opt out", "nonstop", "STOP STOP"],
)
def test_keyword_inside_a_phrase_is_not_the_rule(text: str) -> None:
    message = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text=text)
    assert is_exact_opt_out(message, ("stop", "unsubscribe")) is False


def test_exact_rule_ignores_sales_and_transcripts() -> None:
    sales_stop = _msg("M2", "2026-10-03 09:00:00", text="STOP")  # outbound
    assert is_exact_opt_out(sales_stop, ("stop",)) is False
    transcript = _msg("M3", "2026-10-03 09:00:00", text="Jane (C): Stop.", event_type="CALL")
    assert is_exact_opt_out(transcript, ("stop",)) is False


def test_normalize_whole_message_strips_edges_only() -> None:
    assert normalize_whole_message('  "STOP." ') == "stop"
    assert normalize_whole_message("Stop   now") == "stop now"


# --- phrase hints: without the AI review a human must look; with it the model decides ---


def test_phrase_hint_without_ai_goes_to_manual_review_not_do_not_contact() -> None:
    hint = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="Please STOP texting me.")
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK, hint]))
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert verdict.evidence_message_ids == ("M2",)
    assert "контексту" in verdict.reason


def test_phrase_hint_beats_not_due_without_ai() -> None:
    hint = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="Sorry, wrong number")
    verdict = _verdict(_timeline([SALES_HELLO, hint], silent=NOT_DUE_SILENT))
    assert verdict.decision == Decision.MANUAL_REVIEW


def test_phrase_said_by_sales_is_not_a_hint() -> None:
    sales = _msg("M2", "2026-10-03 09:00:00", text="Reply STOP if you are not interested")
    verdict = _verdict(_timeline([sales, CLIENT_OK]))
    assert verdict.decision == Decision.CONTACT_CANDIDATE


def test_transcript_client_line_is_a_hint_by_speaker_tag() -> None:
    transcript = "[СПІВРОБІТНИК] (S): Should I stop calling?\nJane (C): Yes, don't call me again."
    call = _msg("M2", "2026-10-03 09:00:00", text=transcript, event_type="CALL")  # OUTBOUND channel
    findings = analyse_history(_timeline([SALES_HELLO, call]), CONFIG)
    hints = [f for f in findings if f.kind == FindingKind.PHRASE_HINT]
    assert hints and hints[0].message_ids == ("M2",) and "клієнт" in hints[0].detail


def test_phrase_with_unknown_speaker_is_a_hint_too() -> None:
    transcript = "Jane (C): Hello?\n[СПІВРОБІТНИК] (S): Hi.\nplease stop\n"
    call = _msg("M2", "2026-10-03 09:00:00", text=transcript, event_type="CALL")
    verdict = _verdict(_timeline([SALES_HELLO, call]))
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "невизначений автор" in verdict.reason


def test_word_boundaries_avoid_substring_hits() -> None:
    client = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="We have a nonstop schedule")
    findings = analyse_history(_timeline([SALES_HELLO, client]), CONFIG)
    assert all(f.kind != FindingKind.PHRASE_HINT for f in findings)


def test_phrase_hint_can_be_switched_off() -> None:
    relaxed = Config(phrases=CONFIG.phrases, review=ReviewTriggers(phrase_hint=False))
    hint = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="stop by anytime")
    verdict = _verdict(_timeline([SALES_HELLO, hint]), config=relaxed)
    assert verdict.decision == Decision.CONTACT_CANDIDATE


# --- the AI context verdict ---

STOP_BY = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="Sure, stop by Thursday")


def test_ai_clears_a_phrase_hint_in_context() -> None:
    verdict = _verdict(_timeline([SALES_HELLO, STOP_BY]), ai=_ai(AiLabel.NONE))
    assert verdict.decision == Decision.CONTACT_CANDIDATE
    assert "AI не знайшла" in verdict.reason
    assert "[AI test-model]" in verdict.reviewer_note
    assert "[інфо] фраза потребує оцінки контексту" in verdict.reviewer_note


def test_ai_high_confidence_opt_out_with_evidence_blocks() -> None:
    for label in (AiLabel.OPT_OUT, AiLabel.NEGATIVE):
        verdict = _verdict(_timeline([SALES_HELLO, STOP_BY]), ai=_ai(label, evidence=("M2",)))
        assert verdict.decision == Decision.DO_NOT_CONTACT
        assert verdict.evidence_message_ids == ("M2",)
        assert "AI:" in verdict.reason


def test_ai_opt_out_without_evidence_is_manual_review() -> None:
    verdict = _verdict(_timeline([SALES_HELLO, STOP_BY]), ai=_ai(AiLabel.OPT_OUT, evidence=()))
    assert verdict.decision == Decision.MANUAL_REVIEW


@pytest.mark.parametrize("confidence", [AiConfidence.MEDIUM, AiConfidence.LOW])
def test_ai_less_than_high_confidence_is_manual_review(confidence: AiConfidence) -> None:
    for label in (AiLabel.OPT_OUT, AiLabel.NEGATIVE, AiLabel.NONE):
        verdict = _verdict(_timeline([SALES_HELLO, STOP_BY]), ai=_ai(label, confidence, evidence=("M2",)))
        assert verdict.decision == Decision.MANUAL_REVIEW, label
        assert "впевненість" in verdict.reason


def test_ai_unclear_is_manual_review() -> None:
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK]), ai=_ai(AiLabel.UNCLEAR))
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "неоднозначний" in verdict.reason


def test_ai_error_is_manual_review_with_note() -> None:
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK]), ai=_ai(AiLabel.NONE, error="ConnectionError: x"))
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "не виконано" in verdict.reason
    assert "ConnectionError" in verdict.reviewer_note


def test_ai_cannot_override_exact_stop_or_due_date() -> None:
    keyword = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="STOP")
    verdict = _verdict(_timeline([SALES_HELLO, keyword]), ai=_ai(AiLabel.NONE))
    assert verdict.decision == Decision.DO_NOT_CONTACT
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK], silent=NOT_DUE_SILENT), ai=_ai(AiLabel.NONE))
    assert verdict.decision == Decision.SKIP_NOT_DUE


def test_ai_block_also_wins_over_data_errors() -> None:
    issue = DataIssue(level=IssueLevel.ERROR, code="lead_events_count_mismatch", detail="x", case_id="L1")
    verdict = _verdict(
        _timeline([SALES_HELLO, CLIENT_OK]), issues=[issue], ai=_ai(AiLabel.OPT_OUT, evidence=("M1",))
    )
    assert verdict.decision == Decision.DO_NOT_CONTACT
    assert "lead_events_count_mismatch" in verdict.reviewer_note


# --- the remaining triggers are unchanged ---


def test_contact_after_silent_requires_review_even_when_due() -> None:
    late = _msg("M9", "2026-10-25 09:00:00", text="checking in")  # after silent 2026-10-20
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK, late]))
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert verdict.evidence_message_ids == ("M9",)


def test_contact_after_silent_beats_not_due() -> None:
    late = _msg("M9", "2026-11-04 09:00:00", direction=Direction.INBOUND, text="hey")
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK, late], silent=NOT_DUE_SILENT))
    assert verdict.decision == Decision.MANUAL_REVIEW


def test_hidden_inbound_text_is_not_permission() -> None:
    hidden = _hidden("M2", "2026-10-03 09:00:00", Direction.INBOUND)
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK, hidden]))
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "прихованим" in verdict.reason
    assert verdict.evidence_message_ids == ("M2",)


def test_hidden_outbound_only_is_informational() -> None:
    hidden = _hidden("M2", "2026-10-03 09:00:00", Direction.OUTBOUND)
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK, hidden]))
    assert verdict.decision == Decision.CONTACT_CANDIDATE
    assert "прихованим" in verdict.reviewer_note


def test_no_client_text_goes_to_manual_review() -> None:
    hangup = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text=None, event_type="CALL")
    verdict = _verdict(_timeline([SALES_HELLO, hangup]))
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "контекст" in verdict.reason
    assert verdict.evidence_message_ids == ("M2",)


def test_review_triggers_can_be_switched_off_in_config() -> None:
    relaxed = Config(phrases=CONFIG.phrases, review=ReviewTriggers(no_client_text=False))
    verdict = _verdict(_timeline([SALES_HELLO]), config=relaxed)
    assert verdict.decision == Decision.CONTACT_CANDIDATE


def test_data_error_forces_manual_review_but_exact_stop_still_wins() -> None:
    issue = DataIssue(level=IssueLevel.ERROR, code="lead_events_count_mismatch", detail="x", case_id="L1")
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK]), issues=[issue])
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "lead_events_count_mismatch" in verdict.reason

    keyword = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="UNSUBSCRIBE")
    verdict = _verdict(_timeline([SALES_HELLO, keyword]), issues=[issue])
    assert verdict.decision == Decision.DO_NOT_CONTACT
    assert "lead_events_count_mismatch" in verdict.reviewer_note


def test_data_warning_is_noted_but_does_not_change_decision() -> None:
    issue = DataIssue(level=IssueLevel.WARNING, code="lead_quote_without_date", detail="y", case_id="L1")
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK]), issues=[issue])
    assert verdict.decision == Decision.CONTACT_CANDIDATE
    assert "lead_quote_without_date" in verdict.reviewer_note


def test_missing_silent_date_is_manual_review() -> None:
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK], silent=None))
    assert verdict.decision == Decision.MANUAL_REVIEW


def test_evidence_from_another_case_is_a_hard_error() -> None:
    from silent_leads.rules.history import Finding

    timeline = _timeline([SALES_HELLO, CLIENT_OK])
    foreign = Finding(FindingKind.EXACT_OPT_OUT, ("L2-M1",), "leaked")
    with pytest.raises(EvidenceLeakError):
        decide(timeline, check_due(timeline.lead, CONTROL, 7), [foreign], [])
