"""New contact after silent: the brief's literal rule (default) and the narrowed rule with a grace window."""

from __future__ import annotations

from silent_leads.config import ContactPolicy
from silent_leads.models import ContentState, Decision, Direction, PolicyLabel
from tests.test_contact_policy import CLIENT, CONFIG, SALES, _assessment, _event, _run, _timeline
from tests.test_timeline import _msg

# The lead in these tests went silent at 2026-10-20 10:00.
GRACE = CONFIG.model_copy(update={"contact_policy": ContactPolicy(sales_followup_grace_hours=24)})


def test_by_default_any_event_after_silent_is_manual_review() -> None:
    sales_after = _msg("M3", "2026-10-20 11:00:00", text="Last check-in on your quote!")
    verdict = _run(_timeline([SALES, CLIENT, sales_after]), _assessment())
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert verdict.new_contact_after_silent


def test_unanswered_sales_attempt_inside_the_window_is_not_a_new_contact() -> None:
    sales_after = _msg("M3", "2026-10-20 11:00:00", text="Last check-in on your quote!")
    verdict = _run(_timeline([SALES, CLIENT, sales_after]), _assessment(), config=GRACE)
    assert verdict.decision == Decision.CONTACT_CANDIDATE
    assert verdict.new_contact_after_silent  # the fact is still recorded for the «Розмітка» column
    assert "не новий контакт" in verdict.reviewer_note


def test_voicemail_greeting_reached_after_silent_is_not_a_new_contact() -> None:
    greeting = _event(
        "M3",
        "2026-10-20 10:00:30",
        call_result="SUCCESS",
        content_state=ContentState.TEXT_AVAILABLE,
        text="[КЛІЄНТ] (C): Your call has been forwarded to voicemail.",
    )
    verdict = _run(_timeline([SALES, CLIENT, greeting]), _assessment(position="M2"), config=GRACE)
    assert verdict.decision == Decision.CONTACT_CANDIDATE


def test_client_event_after_silent_stays_manual_review() -> None:
    client_after = _msg(
        "M3", "2026-10-20 10:30:00", direction=Direction.INBOUND, text="Are you still available?"
    )
    verdict = _run(_timeline([SALES, CLIENT, client_after]), _assessment(position="M2"), config=GRACE)
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "з боку клієнта" in verdict.reason
    assert "M3" in verdict.evidence_message_ids


def test_client_position_spoken_after_silent_stays_manual_review() -> None:
    call = _event(
        "M3",
        "2026-10-20 10:20:00",
        call_result="SUCCESS",
        content_state=ContentState.TEXT_AVAILABLE,
        text="[СПІВРОБІТНИК] (S): Any update?\n[КЛІЄНТ] (C): Still thinking about it, call me next month.",
    )
    verdict = _run(_timeline([SALES, CLIENT, call]), _assessment(position="M3"), config=GRACE)
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "після silent" in verdict.reason


def test_late_sales_touch_stays_manual_review() -> None:
    late = _msg("M3", "2026-10-25 10:00:00", text="Following up again!")
    verdict = _run(_timeline([SALES, CLIENT, late]), _assessment(), config=GRACE)
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "пізніше" in verdict.reason


def test_hidden_conversation_inside_the_window_still_needs_a_human() -> None:
    hidden = _event("M3", "2026-10-20 10:10:00", hidden_kind="conversation", call_result="SUCCESS")
    verdict = _run(_timeline([SALES, CLIENT, hidden]), _assessment(), config=GRACE)
    assert verdict.label == PolicyLabel.NEED_HUMAN  # step 4 is independent of the grace window
    assert verdict.decision == Decision.MANUAL_REVIEW
