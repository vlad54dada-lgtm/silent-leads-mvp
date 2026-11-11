"""New contact after silent: the brief's literal rule (default) and the narrowed rule with a grace window.

The narrowed rule must not lean on the model: whatever could hide a client's reaction after
silent (a conversation, an unclear author, a wrong role) is caught by code.
"""

from __future__ import annotations

import pytest

from silent_leads.config import Config, ContactPolicy
from silent_leads.models import ContentState, Decision, Direction, Message, PolicyLabel, Role
from silent_leads.rules.contact_policy import AiReason
from tests.test_contact_policy import CLIENT, CONFIG, SALES, _assessment, _event, _run, _timeline
from tests.test_timeline import _msg

# The lead in these tests went silent at 2026-10-20 10:00:00.
WINDOW_HOURS = 3


def _config(**policy: object) -> Config:
    settings = {"sales_followup_grace_hours": WINDOW_HOURS, **policy}
    return CONFIG.model_copy(update={"contact_policy": ContactPolicy(**settings)})  # type: ignore[arg-type]


GRACE = _config()


def _call(message_id: str, occurred: str, text: str) -> Message:
    return _event(
        message_id, occurred, call_result="SUCCESS", content_state=ContentState.TEXT_AVAILABLE, text=text
    )


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


def test_answering_machine_reached_after_silent_is_not_a_new_contact() -> None:
    greeting = _call("M3", "2026-10-20 10:00:30", "[КЛІЄНТ] (C): Your call has been forwarded to voicemail.")
    verdict = _run(_timeline([SALES, CLIENT, greeting]), _assessment(position="M2"), config=GRACE)
    assert verdict.decision == Decision.CONTACT_CANDIDATE


def test_read_receipt_inside_the_window_is_not_a_new_contact() -> None:
    read = _event(
        "M3",
        "2026-10-20 10:12:00",
        event_type="READ",
        call_result="DELIVERED",
        content_state=ContentState.EMPTY_TEXT,
    )
    assert (
        _run(_timeline([SALES, CLIENT, read]), _assessment(), config=GRACE).decision
        == Decision.CONTACT_CANDIDATE
    )


def test_client_event_after_silent_stays_manual_review() -> None:
    client_after = _msg(
        "M3", "2026-10-20 10:30:00", direction=Direction.INBOUND, text="Are you still available?"
    )
    verdict = _run(_timeline([SALES, CLIENT, client_after]), _assessment(position="M2"), config=GRACE)
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "не є дією продавця" in verdict.reason
    assert "M3" in verdict.evidence_message_ids


@pytest.mark.parametrize("position", ["M2", None])
def test_conversation_after_silent_is_caught_by_code_even_if_the_model_misses_it(
    position: str | None,
) -> None:
    """The client refuses in an outbound call 20 minutes after silent; the model points elsewhere."""
    call = _call(
        "M3",
        "2026-10-20 10:20:00",
        "[СПІВРОБІТНИК] (S): Hi, any update on the quote?\n"
        "[КЛІЄНТ] (C): Oh, hi.\n"
        "[КЛІЄНТ] (C): We went with another company, sorry.",
    )
    verdict = _run(_timeline([SALES, CLIENT, call]), _assessment(position=position), config=GRACE)
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "розмова з клієнтом після silent" in verdict.reason
    assert "M3" in verdict.evidence_message_ids


def test_one_client_line_is_a_conversation_when_the_limit_is_zero() -> None:
    greeting = _call("M3", "2026-10-20 10:00:30", "[КЛІЄНТ] (C): Please leave a message.")
    strict = _config(followup_max_client_lines=0)
    assert (
        _run(_timeline([SALES, CLIENT, greeting]), _assessment(), config=strict).decision
        == Decision.MANUAL_REVIEW
    )


def test_client_position_spoken_after_silent_stays_manual_review() -> None:
    call = _call("M3", "2026-10-20 10:20:00", "[КЛІЄНТ] (C): Still thinking about it, call me next month.")
    verdict = _run(_timeline([SALES, CLIENT, call]), _assessment(position="M3"), config=GRACE)
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "після silent" in verdict.reason


def test_one_live_client_line_after_silent_is_judged_by_the_model_like_any_client_line() -> None:
    """Code cannot tell an answering machine from a one-line live reply, so it lets one line through.

    The line stays in the dialogue the model reads: a refusal in it blocks the lead, and a position
    spoken there sends it to a human (the test above). Setting the limit to 0 removes this reliance.
    """
    call = _call("M3", "2026-10-20 10:20:00", "[КЛІЄНТ] (C): We hired someone else, please take us off.")
    refusal = _assessment(
        PolicyLabel.DO_NOT_CONTACT, AiReason.HIRED_OTHER_OR_DONE, position="M3", evidence=["M3"]
    )
    verdict = _run(_timeline([SALES, CLIENT, call]), refusal, config=GRACE)
    assert verdict.decision == Decision.DO_NOT_CONTACT
    assert "M3" in verdict.evidence_message_ids


def test_line_with_an_undetermined_speaker_after_silent_stays_manual_review() -> None:
    untagged = _call("M3", "2026-10-20 10:05:00", "Hello? Who is this?")  # a transcript without (C)/(S) tags
    verdict = _run(_timeline([SALES, CLIENT, untagged]), _assessment(), config=GRACE)
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "невизначеним автором" in verdict.reason


@pytest.mark.parametrize("role", [Role.CLIENT, None])
def test_outbound_event_that_is_not_marked_as_sales_stays_manual_review(role: Role | None) -> None:
    odd = _msg("M3", "2026-10-20 10:30:00", text="Thanks!").model_copy(update={"role": role})
    verdict = _run(_timeline([SALES, CLIENT, odd]), _assessment(), config=GRACE)
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "не є дією продавця" in verdict.reason


@pytest.mark.parametrize(
    ("occurred", "decision"),
    [
        ("2026-10-20 13:00:00", Decision.CONTACT_CANDIDATE),  # exactly the window: still the closing attempt
        ("2026-10-20 13:00:01", Decision.MANUAL_REVIEW),  # one second later: a re-touch was already made
        ("2026-10-25 10:00:00", Decision.MANUAL_REVIEW),
    ],
)
def test_window_boundary(occurred: str, decision: Decision) -> None:
    verdict = _run(
        _timeline([SALES, CLIENT, _msg("M3", occurred, text="Following up!")]), _assessment(), config=GRACE
    )
    assert verdict.decision == decision
    if decision == Decision.MANUAL_REVIEW:
        assert "пізніше" in verdict.reason


def test_zero_window_keeps_everything_after_the_silent_second_in_manual_review() -> None:
    zero = _config(sales_followup_grace_hours=0)
    just_after = _msg("M3", "2026-10-20 10:00:01", text="Following up!")
    assert (
        _run(_timeline([SALES, CLIENT, just_after]), _assessment(), config=zero).decision
        == Decision.MANUAL_REVIEW
    )


def test_event_in_the_same_second_as_silent_is_not_after_silent() -> None:
    same_second = _msg("M3", "2026-10-20 10:00:00", text="Closing message")
    verdict = _run(_timeline([SALES, CLIENT, same_second]), _assessment())  # the literal rule
    assert not verdict.new_contact_after_silent
    assert verdict.decision == Decision.CONTACT_CANDIDATE


def test_one_late_event_among_several_is_enough_for_manual_review() -> None:
    inside = _msg("M3", "2026-10-20 10:10:00", text="Thanks, talk soon!")
    late = _msg("M4", "2026-10-22 09:00:00", text="Hi again, any news?")
    verdict = _run(_timeline([SALES, CLIENT, inside, late]), _assessment(), config=GRACE)
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "M4" in verdict.evidence_message_ids and "M3" not in verdict.evidence_message_ids


def test_hidden_conversation_inside_the_window_still_needs_a_human() -> None:
    hidden = _event("M3", "2026-10-20 10:10:00", hidden_kind="conversation", call_result="SUCCESS")
    verdict = _run(_timeline([SALES, CLIENT, hidden]), _assessment(), config=GRACE)
    assert verdict.label == PolicyLabel.NEED_HUMAN  # step 4 is independent of the window
    assert verdict.decision == Decision.MANUAL_REVIEW
