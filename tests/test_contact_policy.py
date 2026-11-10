"""v2 policy («Кому пишемо»): deterministic parts, step 4, every branch of the judge's answer."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from silent_leads.config import AiReview, Config, Phrases
from silent_leads.models import (
    ContentState,
    DataIssue,
    Decision,
    Direction,
    IssueLevel,
    Message,
    PolicyLabel,
    Role,
)
from silent_leads.rules.ai_review import AiConfidence
from silent_leads.rules.contact_policy import (
    REASON_AI_ERROR,
    REASON_HIDDEN_CONVERSATION,
    REASON_NO_AI,
    AiReason,
    PolicyAssessment,
    PolicyOutcome,
    PolicyVerdict,
    decide_policy,
    judge_case,
    needs_judge,
    policy_items,
    step4_risks,
)
from silent_leads.rules.due_date import check_due
from silent_leads.timeline import CaseTimeline, build_timelines
from tests.test_timeline import _lead, _msg

CONTROL = date(2026, 11, 5)
CONFIG = Config(policy="v2", phrases=Phrases(exact_opt_out=("stop", "unsubscribe"), hints=()))
AI = AiReview(enabled=True, model="test-model")
DUE_SILENT = "2026-10-20 10:00:00"  # 16 days before control -> due
NOT_DUE_SILENT = "2026-11-03 10:00:00"  # 2 days -> not due

SALES = _msg("M1", "2026-10-01 09:00:00", text="Here is your quote for the bedroom wallpaper.")
CLIENT = _msg(
    "M2", "2026-10-02 09:00:00", direction=Direction.INBOUND, text="A bit over our budget, let me think."
)


def _event(
    message_id: str,
    occurred: str,
    *,
    direction: Direction = Direction.OUTBOUND,
    event_type: str = "CALL",
    call_result: str | None = None,
    content_state: ContentState = ContentState.WITHHELD_HIDDEN,
    hidden_kind: str | None = None,
    text: str | None = None,
) -> Message:
    return Message(
        case_id="L1",
        message_id=message_id,
        occurred_at_crm=datetime.fromisoformat(occurred),
        event_type=event_type,
        direction=direction,
        role=Role.CLIENT if direction == Direction.INBOUND else Role.SALES,
        call_result=call_result,
        call_seconds=None,
        content_state=content_state,
        text=text,
        hidden_kind=hidden_kind,
    )


def _timeline(messages: list[Message], silent: str | None = DUE_SILENT) -> CaseTimeline:
    return build_timelines([_lead(silent=silent)], messages)["L1"]


def _assessment(
    label: PolicyLabel = PolicyLabel.CONTACT_OK,
    reason: AiReason = AiReason.PRICE,
    confidence: AiConfidence = AiConfidence.HIGH,
    position: str | None = "M2",
    evidence: list[str] | None = None,
    question: str = "",
    stop_then_resumed: bool = False,
) -> PolicyAssessment:
    return PolicyAssessment(
        label=label,
        reason=reason,
        confidence=confidence,
        last_client_position_message_id=position,
        evidence_message_ids=evidence if evidence is not None else ([position] if position else []),
        stop_then_resumed=stop_then_resumed,
        question_for_owner=question,
        explanation="пояснення",
    )


class FakeJudge:
    def __init__(self, answer: object) -> None:
        self.answer = answer
        self.payloads: list[str] = []

    def judge(self, case_id: str, payload: str) -> PolicyAssessment:
        self.payloads.append(payload)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer  # type: ignore[return-value]


def _run(
    timeline: CaseTimeline,
    answer: object | None,
    issues: list[DataIssue] | None = None,
    config: Config = CONFIG,
    ai: AiReview = AI,
) -> PolicyVerdict:
    outcome: PolicyOutcome | None = None
    if answer is not None and needs_judge(timeline, config):
        outcome = judge_case(timeline, FakeJudge(answer), ai, config.contact_policy)
    due = check_due(timeline.lead, CONTROL, config.first_touch_delay_days)
    return decide_policy(timeline, due, issues or [], outcome, config)


# --------------------------------------------------------------------------- deterministic parts


def test_final_exact_stop_blocks_without_asking_the_model() -> None:
    stop = _msg("M3", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="STOP")
    later_sales = _msg("M4", "2026-10-05 09:00:00", text="Hi again!")
    timeline = _timeline([SALES, CLIENT, stop, later_sales])
    assert not needs_judge(timeline, CONFIG)
    verdict = _run(timeline, None)
    assert verdict.label == PolicyLabel.DO_NOT_CONTACT
    assert verdict.label_reason == AiReason.OPTOUT
    assert verdict.decision == Decision.DO_NOT_CONTACT
    assert verdict.evidence_message_ids == ("M3",)
    assert verdict.confidence == "rule"


def test_stop_then_client_resumed_goes_to_the_model_and_never_straight_to_candidate() -> None:
    stop = _msg("M3", "2026-10-03 09:00:00", direction=Direction.INBOUND, text="Stop")
    back = _msg(
        "M4", "2026-10-06 09:00:00", direction=Direction.INBOUND, text="Actually, how is your schedule?"
    )
    timeline = _timeline([SALES, CLIENT, stop, back])
    assert needs_judge(timeline, CONFIG)
    verdict = _run(timeline, _assessment(reason=AiReason.POSTPONE_OR_COMPARING, position="M4"))
    assert verdict.label == PolicyLabel.CONTACT_OK
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "STOP" in verdict.reason
    assert verdict.question_for_owner  # default question filled in by code


def test_client_who_never_said_anything_is_contact_ok_silent_without_the_model() -> None:
    timeline = _timeline([SALES, _msg("M2", "2026-10-03 09:00:00", text="Just checking in!")])
    assert not needs_judge(timeline, CONFIG)
    verdict = _run(timeline, None)
    assert (verdict.label, verdict.label_reason, verdict.decision) == (
        PolicyLabel.CONTACT_OK,
        AiReason.SILENT,
        Decision.CONTACT_CANDIDATE,
    )


def test_silent_client_inside_the_seven_days_is_skip_not_due() -> None:
    timeline = _timeline([SALES], silent=NOT_DUE_SILENT)
    assert _run(timeline, None).decision == Decision.SKIP_NOT_DUE


def test_without_ai_client_words_need_a_human() -> None:
    verdict = _run(_timeline([SALES, CLIENT]), None)
    assert verdict.label == PolicyLabel.NEED_HUMAN
    assert verdict.label_reason == REASON_NO_AI
    assert verdict.decision == Decision.MANUAL_REVIEW


# --------------------------------------------------------------------------- the judge's answers


def test_high_confidence_contact_ok_with_due_date_is_a_candidate_with_the_position_as_evidence() -> None:
    verdict = _run(_timeline([SALES, CLIENT]), _assessment())
    assert verdict.decision == Decision.CONTACT_CANDIDATE
    assert verdict.label == PolicyLabel.CONTACT_OK
    assert verdict.evidence_message_ids == ("M2",)
    assert verdict.position_id == "M2"


def test_contact_ok_not_due_is_skip() -> None:
    verdict = _run(_timeline([SALES, CLIENT], silent=NOT_DUE_SILENT), _assessment())
    assert verdict.decision == Decision.SKIP_NOT_DUE


@pytest.mark.parametrize("confidence", [AiConfidence.MEDIUM, AiConfidence.LOW])
def test_contact_ok_without_high_confidence_is_manual_review(confidence: AiConfidence) -> None:
    verdict = _run(_timeline([SALES, CLIENT]), _assessment(confidence=confidence))
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert verdict.label == PolicyLabel.CONTACT_OK  # the label is kept for evaluation


def test_label_inconsistent_with_reason_is_manual_review() -> None:
    verdict = _run(_timeline([SALES, CLIENT]), _assessment(reason=AiReason.POLITE_NO))
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "не відповідає" in verdict.reviewer_note


def test_high_confidence_refusal_with_evidence_blocks() -> None:
    answer = _assessment(PolicyLabel.DO_NOT_CONTACT, AiReason.HIRED_OTHER_OR_DONE, evidence=["M2"])
    verdict = _run(_timeline([SALES, CLIENT]), answer)
    assert verdict.decision == Decision.DO_NOT_CONTACT
    assert verdict.evidence_message_ids == ("M2",)


@pytest.mark.parametrize(
    ("confidence", "evidence"),
    [(AiConfidence.MEDIUM, ["M2"]), (AiConfidence.HIGH, []), (AiConfidence.HIGH, ["OTHER-M9"])],
)
def test_refusal_without_certainty_or_own_evidence_is_manual_review(
    confidence: AiConfidence, evidence: list[str]
) -> None:
    answer = _assessment(
        PolicyLabel.DO_NOT_CONTACT, AiReason.HIRED_OTHER_OR_DONE, confidence=confidence, evidence=evidence
    )
    verdict = _run(_timeline([SALES, CLIENT]), answer)
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert verdict.label == PolicyLabel.DO_NOT_CONTACT
    assert all(mid.startswith("M") for mid in verdict.evidence_message_ids)


def test_polite_no_blocks_and_always_carries_a_question_for_the_owner() -> None:
    answer = _assessment(PolicyLabel.DO_NOT_CONTACT, AiReason.POLITE_NO, evidence=["M2"])
    verdict = _run(_timeline([SALES, CLIENT]), answer)
    assert verdict.decision == Decision.DO_NOT_CONTACT
    assert verdict.question_for_owner


def test_model_need_human_is_manual_review() -> None:
    answer = _assessment(PolicyLabel.NEED_HUMAN, AiReason.UNCLEAR, position=None)
    verdict = _run(_timeline([SALES, CLIENT]), answer)
    assert (verdict.label, verdict.decision) == (PolicyLabel.NEED_HUMAN, Decision.MANUAL_REVIEW)


@pytest.mark.parametrize("answer", [RuntimeError("network down"), "not an assessment"])
def test_judge_failure_or_wrong_type_is_manual_review_not_a_crash(answer: object) -> None:
    verdict = _run(_timeline([SALES, CLIENT]), answer)
    assert verdict.label_reason == REASON_AI_ERROR
    assert verdict.decision == Decision.MANUAL_REVIEW


def test_truncated_dialogue_never_clears_a_case() -> None:
    long_sales = _msg("M0", "2026-09-01 09:00:00", text="x" * 2000)
    ai = AiReview(enabled=True, model="test-model", max_input_chars=500)
    verdict = _run(_timeline([long_sales, SALES, CLIENT]), _assessment(), ai=ai)
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "обрізано" in verdict.reviewer_note


def test_new_contact_after_silent_is_manual_review_and_flagged() -> None:
    after = _msg("M3", "2026-10-25 09:00:00", text="Following up!")
    verdict = _run(_timeline([SALES, CLIENT, after]), _assessment())
    assert verdict.new_contact_after_silent
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "M3" in verdict.evidence_message_ids


def test_data_error_stops_a_candidate_but_not_a_block() -> None:
    error = [DataIssue(level=IssueLevel.ERROR, code="bad", detail="x", case_id="L1")]
    timeline = _timeline([SALES, CLIENT])
    assert _run(timeline, _assessment(), issues=error).decision == Decision.MANUAL_REVIEW
    block = _assessment(PolicyLabel.DO_NOT_CONTACT, AiReason.OPTOUT, evidence=["M2"])
    assert _run(timeline, block, issues=error).decision == Decision.DO_NOT_CONTACT


# --------------------------------------------------------------------------- step 4


def test_hidden_conversation_after_the_last_position_needs_a_human() -> None:
    hidden = _event("M3", "2026-10-04 09:00:00", hidden_kind="conversation", call_result="SUCCESS")
    verdict = _run(_timeline([SALES, CLIENT, hidden]), _assessment())
    assert verdict.label == PolicyLabel.NEED_HUMAN
    assert verdict.label_reason == REASON_HIDDEN_CONVERSATION
    assert verdict.evidence_message_ids == ("M3",)
    assert verdict.decision == Decision.MANUAL_REVIEW


def test_hidden_conversation_before_the_last_position_is_not_a_reason() -> None:
    hidden = _event("M1H", "2026-10-01 12:00:00", hidden_kind="conversation", call_result="SUCCESS")
    verdict = _run(_timeline([SALES, hidden, CLIENT]), _assessment())
    assert verdict.decision == Decision.CONTACT_CANDIDATE


@pytest.mark.parametrize(
    "event",
    [
        _event("M3", "2026-10-04 09:00:00", hidden_kind="voicemail_greeting", call_result="SUCCESS"),
        _event("M3", "2026-10-04 09:00:00", hidden_kind="contact_details", event_type="SMS"),
        _event("M3", "2026-10-04 09:00:00", hidden_kind="platform_notification", event_type="SMS"),
        _event(
            "M3", "2026-10-04 09:00:00", call_result="HANGUP", content_state=ContentState.CALL_WITHOUT_TEXT
        ),
        _event("M3", "2026-10-04 09:00:00", call_result="VM", content_state=ContentState.CALL_WITHOUT_TEXT),
    ],
)
def test_harmless_hidden_events_after_the_position_are_not_a_reason(event: Message) -> None:
    assert _run(_timeline([SALES, CLIENT, event]), _assessment()).decision == Decision.CONTACT_CANDIDATE


@pytest.mark.parametrize(
    "event",
    [
        _event(
            "M3", "2026-10-04 09:00:00", call_result="SUCCESS", content_state=ContentState.CALL_WITHOUT_TEXT
        ),
        _event("M3", "2026-10-04 09:00:00", hidden_kind=None, event_type="SMS"),  # unknown kind = risk
    ],
)
def test_connected_call_without_transcript_or_unknown_hidden_kind_needs_a_human(event: Message) -> None:
    assert _run(_timeline([SALES, CLIENT, event]), _assessment()).label == PolicyLabel.NEED_HUMAN


def test_position_that_is_not_a_client_message_falls_back_to_the_whole_history() -> None:
    hidden = _event("M1H", "2026-10-01 12:00:00", hidden_kind="conversation", call_result="SUCCESS")
    timeline = _timeline([SALES, hidden, CLIENT])
    verdict = _run(timeline, _assessment(position="M1"))  # M1 is the sales quote
    assert verdict.label == PolicyLabel.NEED_HUMAN
    assert "не репліка клієнта" in verdict.reviewer_note
    assert step4_risks(timeline, None, CONFIG.contact_policy) == ("M1H",)


def test_silent_client_with_a_hidden_conversation_anywhere_needs_a_human() -> None:
    hidden = _event("M2", "2026-10-02 09:00:00", hidden_kind="conversation", call_result="SUCCESS")
    verdict = _run(_timeline([SALES, hidden]), None)
    assert verdict.label == PolicyLabel.NEED_HUMAN


# --------------------------------------------------------------------------- what the model sees


def test_policy_items_attribute_transcript_lines_and_skip_hangups() -> None:
    call = _event(
        "M3",
        "2026-10-03 09:00:00",
        call_result="SUCCESS",
        content_state=ContentState.TEXT_AVAILABLE,
        text="[СПІВРОБІТНИК] (S): Hi, any questions?\n[КЛІЄНТ] (C): We will hold off for now.",
    )
    hangup = _event(
        "M4", "2026-10-04 09:00:00", call_result="HANGUP", content_state=ContentState.CALL_WITHOUT_TEXT
    )
    greeting = _event("M5", "2026-10-05 09:00:00", hidden_kind="voicemail_greeting", call_result="VM")
    items = policy_items(_timeline([SALES, CLIENT, call, hangup, greeting]), CONFIG.contact_policy)
    speakers = [(i["message_id"], i.get("speaker")) for i in items]
    assert ("M3", "sales") in speakers and ("M3", "client") in speakers
    assert all(i["message_id"] != "M4" for i in items)
    assert {
        "message_id": "M5",
        "event": "CALL OUTBOUND VM",
        "hidden": True,
        "hidden_kind": "voicemail_greeting",
    } in items
