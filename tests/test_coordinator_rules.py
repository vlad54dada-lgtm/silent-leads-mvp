"""Rules the coordinator set on 2026-11-12 that code, not the model, applies.

A step 3 refusal followed by a wordless inbound event from the client (a missed call two days after
"we will do it ourselves") goes to a human: nobody knows what the client wanted.
"""

from __future__ import annotations

import pytest

from silent_leads.config import ContactPolicy
from silent_leads.models import ContentState, Decision, Direction, Message, PolicyLabel
from silent_leads.rules.ai_review import AiConfidence
from silent_leads.rules.contact_policy import (
    REASON_CLIENT_EVENT_AFTER_REFUSAL,
    AiReason,
    wordless_client_events_after,
)
from tests.test_contact_policy import CONFIG, SALES, _assessment, _event, _run, _timeline
from tests.test_timeline import _msg

REFUSAL = _msg(
    "M2", "2026-10-02 09:00:00", direction=Direction.INBOUND, text="Thanks, we will hang it ourselves."
)


def _missed_call(message_id: str = "M3", occurred: str = "2026-10-04 09:00:00") -> Message:
    return _event(
        message_id,
        occurred,
        direction=Direction.INBOUND,
        call_result="HANGUP",
        content_state=ContentState.CALL_WITHOUT_TEXT,
    )


def _refusal(reason: AiReason = AiReason.HIRED_OTHER_OR_DONE, **kwargs: object) -> object:
    return _assessment(PolicyLabel.DO_NOT_CONTACT, reason, evidence=["M2"], **kwargs)  # type: ignore[arg-type]


def test_missed_client_call_after_a_refusal_needs_a_human() -> None:
    verdict = _run(_timeline([SALES, REFUSAL, _missed_call()]), _refusal())
    assert verdict.label == PolicyLabel.NEED_HUMAN
    assert verdict.label_reason == REASON_CLIENT_EVENT_AFTER_REFUSAL
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert verdict.evidence_message_ids == ("M2", "M3")  # the refusal and the event that reopened it
    assert "M3" in verdict.reviewer_note


@pytest.mark.parametrize("reason", [AiReason.POLITE_NO, AiReason.WRONG_CONTACT_OR_SERVICE])
def test_every_step_three_refusal_is_reopened_by_a_wordless_client_event(reason: AiReason) -> None:
    verdict = _run(_timeline([SALES, REFUSAL, _missed_call()]), _refusal(reason))
    assert verdict.label == PolicyLabel.NEED_HUMAN
    assert verdict.decision == Decision.MANUAL_REVIEW


@pytest.mark.parametrize("reason", [AiReason.OPTOUT, AiReason.NEGATIVE_OR_COMPLAINT])
def test_hard_stop_is_not_reopened_by_a_missed_call(reason: AiReason) -> None:
    verdict = _run(_timeline([SALES, REFUSAL, _missed_call()]), _refusal(reason))
    assert verdict.label == PolicyLabel.DO_NOT_CONTACT
    assert verdict.decision == Decision.DO_NOT_CONTACT


def test_hidden_client_voicemail_after_a_refusal_needs_a_human() -> None:
    voicemail = _event(
        "M3",
        "2026-10-04 09:00:00",
        direction=Direction.INBOUND,
        hidden_kind="client_voicemail",
        call_result="VM",
    )
    verdict = _run(_timeline([SALES, REFUSAL, voicemail]), _refusal())
    assert verdict.label == PolicyLabel.NEED_HUMAN


@pytest.mark.parametrize(
    "later",
    [
        _event(
            "M3", "2026-10-04 09:00:00", call_result="HANGUP", content_state=ContentState.CALL_WITHOUT_TEXT
        ),
        _msg("M3", "2026-10-04 09:00:00", text="Just checking in once more!"),
    ],
    ids=["sales call without text", "sales text"],
)
def test_sales_events_after_a_refusal_change_nothing(later: Message) -> None:
    verdict = _run(_timeline([SALES, REFUSAL, later]), _refusal())
    assert verdict.decision == Decision.DO_NOT_CONTACT


def test_client_words_after_a_refusal_are_left_to_the_model() -> None:
    """Text is something the model has read; only an event with nothing to read triggers the rule."""
    thanks = _msg("M3", "2026-10-04 09:00:00", direction=Direction.INBOUND, text="Thanks again.")
    verdict = _run(_timeline([SALES, REFUSAL, thanks]), _refusal())
    assert verdict.decision == Decision.DO_NOT_CONTACT


def test_missed_call_before_the_refusal_changes_nothing() -> None:
    verdict = _run(_timeline([SALES, _missed_call("M0", "2026-10-01 12:00:00"), REFUSAL]), _refusal())
    assert verdict.decision == Decision.DO_NOT_CONTACT


def test_last_evidence_is_the_anchor_when_the_model_named_no_position() -> None:
    verdict = _run(_timeline([SALES, REFUSAL, _missed_call()]), _refusal(position=None))
    assert verdict.label == PolicyLabel.NEED_HUMAN


def test_refusal_without_certainty_and_a_missed_call_still_goes_to_a_human() -> None:
    verdict = _run(_timeline([SALES, REFUSAL, _missed_call()]), _refusal(confidence=AiConfidence.MEDIUM))
    assert verdict.decision == Decision.MANUAL_REVIEW


def test_rule_can_be_switched_off_in_the_config() -> None:
    off = CONFIG.model_copy(
        update={"contact_policy": ContactPolicy(client_event_after_refusal_needs_human=False)}
    )
    verdict = _run(_timeline([SALES, REFUSAL, _missed_call()]), _refusal(), config=off)
    assert verdict.decision == Decision.DO_NOT_CONTACT


def test_wordless_client_events_are_looked_for_only_after_the_anchor() -> None:
    timeline = _timeline([_missed_call("M0", "2026-10-01 08:00:00"), SALES, REFUSAL, _missed_call()])
    assert wordless_client_events_after(timeline, "M2") == ("M3",)
    assert wordless_client_events_after(timeline, "M3") == ()
    assert wordless_client_events_after(timeline, "no-such-id") == ()
