from __future__ import annotations

from datetime import date, datetime

import pytest

from silent_leads.config import Config, Phrases, ReviewTriggers
from silent_leads.models import ContentState, DataIssue, Decision, Direction, IssueLevel, Message, Role
from silent_leads.rules.decide import EvidenceLeakError, Verdict, decide
from silent_leads.rules.due_date import check_due
from silent_leads.rules.history import FindingKind, analyse_history
from silent_leads.timeline import CaseTimeline, build_timelines
from tests.test_timeline import _lead, _msg

CONTROL = date(2026, 11, 5)
CONFIG = Config(
    phrases=Phrases(
        opt_out=("stop", "don't call", "do not text", "remove me"),
        negative=("not interested", "wrong number"),
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
    timeline: CaseTimeline, issues: list[DataIssue] | None = None, config: Config = CONFIG
) -> Verdict:
    due = check_due(timeline.lead, CONTROL, config.first_touch_delay_days)
    return decide(timeline, due, analyse_history(timeline, config), issues or [])


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


def test_client_opt_out_blocks_regardless_of_due_date() -> None:
    opt_out = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="Please STOP texting me.")
    for silent in (DUE_SILENT, NOT_DUE_SILENT):
        verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK, opt_out], silent=silent))
        assert verdict.decision == Decision.DO_NOT_CONTACT
        assert verdict.evidence_message_ids == ("M2",)
        assert "stop" in verdict.reason


def test_opt_out_found_in_early_message_not_only_last() -> None:
    early = _msg("M2", "2026-09-15 09:00:00", direction=Direction.INBOUND, text="remove me from your list")
    later = _msg("M3", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="ok")
    verdict = _verdict(_timeline([early, SALES_HELLO, later]))
    assert verdict.decision == Decision.DO_NOT_CONTACT
    assert verdict.evidence_message_ids == ("M2",)


def test_client_negative_is_do_not_contact() -> None:
    negative = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="Sorry, wrong number")
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK, negative]))
    assert verdict.decision == Decision.DO_NOT_CONTACT
    assert "негатив" in verdict.reason


def test_phrase_said_by_sales_does_not_block() -> None:
    sales = _msg("M2", "2026-10-03 09:00:00", text="Reply STOP if you are not interested")
    verdict = _verdict(_timeline([sales, CLIENT_OK]))
    assert verdict.decision == Decision.CONTACT_CANDIDATE


def test_transcript_speaker_tags_decide_authorship() -> None:
    transcript = "[СПІВРОБІТНИК] (S): Should I stop calling?\nJane (C): Yes, don't call me again."
    call = _msg("M2", "2026-10-03 09:00:00", text=transcript, event_type="CALL")  # OUTBOUND channel
    verdict = _verdict(_timeline([SALES_HELLO, call]))
    assert verdict.decision == Decision.DO_NOT_CONTACT
    assert verdict.evidence_message_ids == ("M2",)


def test_phrase_with_unknown_speaker_goes_to_manual_review() -> None:
    transcript = "Jane (C): Hello?\n[СПІВРОБІТНИК] (S): Hi.\nplease stop\n"
    call = _msg("M2", "2026-10-03 09:00:00", text=transcript, event_type="CALL")
    verdict = _verdict(_timeline([SALES_HELLO, call]))
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert verdict.evidence_message_ids == ("M2",)
    assert "невизначеним автором" in verdict.reason


def test_word_boundaries_avoid_substring_hits() -> None:
    client = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="We have a nonstop schedule")
    findings = analyse_history(_timeline([SALES_HELLO, client]), CONFIG)
    assert all(f.kind != FindingKind.OPT_OUT for f in findings)


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


def test_data_error_forces_manual_review_but_opt_out_still_wins() -> None:
    issue = DataIssue(level=IssueLevel.ERROR, code="lead_events_count_mismatch", detail="x", case_id="L1")
    verdict = _verdict(_timeline([SALES_HELLO, CLIENT_OK]), issues=[issue])
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "lead_events_count_mismatch" in verdict.reason

    opt_out = _msg("M2", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="do not text me")
    verdict = _verdict(_timeline([SALES_HELLO, opt_out]), issues=[issue])
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
    foreign = Finding(FindingKind.OPT_OUT, ("L2-M1",), "leaked")
    with pytest.raises(EvidenceLeakError):
        decide(timeline, check_due(timeline.lead, CONTROL, 7), [foreign], [])
