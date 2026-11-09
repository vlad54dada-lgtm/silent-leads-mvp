"""AI context analysis: dialogue payload, truncation, outcome mapping, pipeline wiring."""

from __future__ import annotations

import json
from pathlib import Path

from silent_leads.config import AiReview, Config, Phrases
from silent_leads.io.reader import read_export
from silent_leads.models import ContentState, Decision, Direction, Message, Role
from silent_leads.pipeline import run_pipeline
from silent_leads.rules.ai_review import (
    AiAssessment,
    AiConfidence,
    AiLabel,
    AiOutcome,
    build_payload,
    dialogue_items,
    review_case,
)
from silent_leads.timeline import build_timelines
from tests.conftest import build_workbook, lead, message
from tests.test_timeline import _lead, _msg

SETTINGS = AiReview(enabled=True, model="test-model", max_input_chars=500)


class FakeReviewer:
    def __init__(self, assessment: AiAssessment | Exception | dict[str, AiAssessment]) -> None:
        self.assessment = assessment
        self.calls: list[tuple[str, str]] = []

    def assess(self, case_id: str, payload: str) -> AiAssessment:
        self.calls.append((case_id, payload))
        if isinstance(self.assessment, Exception):
            raise self.assessment
        if isinstance(self.assessment, dict):
            return self.assessment[case_id]
        return self.assessment


def _assessment(
    label: AiLabel, confidence: AiConfidence = AiConfidence.HIGH, evidence: list[str] | None = None
) -> AiAssessment:
    return AiAssessment(
        label=label, confidence=confidence, evidence_message_ids=evidence or [], explanation="x"
    )


def _hidden(message_id: str, occurred: str) -> Message:
    return Message(
        case_id="L1",
        message_id=message_id,
        occurred_at_crm=__import__("datetime").datetime.fromisoformat(occurred),
        event_type="SMS",
        direction=Direction.INBOUND,
        role=Role.CLIENT,
        call_result=None,
        call_seconds=None,
        content_state=ContentState.WITHHELD_HIDDEN,
        text=None,
    )


def _timeline():  # type: ignore[no-untyped-def]
    transcript = (
        "Jane (C): Hi, I was waiting for a quote.\n[СПІВРОБІТНИК] (S): We will send it today.\nok bye"
    )
    return build_timelines(
        [_lead()],
        [
            _msg("M1", "2026-10-01 09:00:00", text="Hello from sales"),
            _msg("M2", "2026-10-02 09:00:00", direction=Direction.INBOUND, text="sure, thanks"),
            _hidden("M3", "2026-10-02 10:00:00"),
            _msg("M4", "2026-10-03 09:00:00", text=transcript, event_type="CALL"),
        ],
    )["L1"]


def test_payload_is_the_dialogue_with_speakers_and_ids_but_no_names_or_dates() -> None:
    items = dialogue_items(_timeline())
    assert items == [
        {"message_id": "M1", "speaker": "sales", "text": "Hello from sales"},
        {"message_id": "M2", "speaker": "client", "text": "sure, thanks"},
        {"message_id": "M3", "speaker": "client", "hidden": True},
        {"message_id": "M4", "speaker": "client", "text": "Hi, I was waiting for a quote."},
        {"message_id": "M4", "speaker": "sales", "text": "We will send it today."},
        {"message_id": "M4", "speaker": "unknown", "text": "ok bye"},
    ]
    payload, truncated = build_payload(_timeline(), 500)
    assert "СПІВРОБІТНИК" not in payload and "Jane" not in payload and "2026" not in payload
    assert truncated is False


def test_truncation_keeps_the_most_recent_items() -> None:
    payload, truncated = build_payload(_timeline(), 70)
    kept = json.loads(payload)
    assert truncated is True
    assert [i["message_id"] for i in kept] == ["M2", "M3", "M4", "M4", "M4"]
    assert kept[1] == {"message_id": "M3", "speaker": "client", "hidden": True}


def test_review_case_filters_evidence_to_own_ids_and_keeps_confidence() -> None:
    reviewer = FakeReviewer(_assessment(AiLabel.OPT_OUT, AiConfidence.MEDIUM, ["M2", "L9-M1", "M2"]))
    outcome = review_case(_timeline(), reviewer, SETTINGS)
    assert outcome.label == AiLabel.OPT_OUT
    assert outcome.confidence == AiConfidence.MEDIUM
    assert outcome.evidence_message_ids == ("M2",)
    assert outcome.needs_review and not outcome.blocks
    assert reviewer.calls[0][0] == "L1"


def test_review_case_never_raises() -> None:
    outcome = review_case(_timeline(), FakeReviewer(ConnectionError("network down")), SETTINGS)
    assert outcome.error is not None and "network down" in outcome.error
    assert outcome.label is None and outcome.needs_review


def test_outcome_flags() -> None:
    def outcome(label: AiLabel, confidence: AiConfidence, evidence: tuple[str, ...] = ("M1",)) -> AiOutcome:
        return AiOutcome("m", label, confidence, evidence, "")

    assert outcome(AiLabel.OPT_OUT, AiConfidence.HIGH).blocks
    assert not outcome(AiLabel.OPT_OUT, AiConfidence.HIGH, ()).blocks
    assert not outcome(AiLabel.OPT_OUT, AiConfidence.MEDIUM).blocks
    assert outcome(AiLabel.NONE, AiConfidence.HIGH).clears
    assert not outcome(AiLabel.NONE, AiConfidence.MEDIUM).clears
    assert outcome(AiLabel.UNCLEAR, AiConfidence.HIGH).needs_review


def test_pipeline_reviews_every_case_with_client_text_except_exact_stop(workbook_path: Path) -> None:
    build_workbook(
        workbook_path,
        [
            lead("L1"),  # due, "stop by" -> AI clears -> candidate
            lead("L2"),  # exact STOP -> rule, never sent to the model
            lead("L3", silent_since_crm="2026-11-03 10:00:00"),  # not due, still analysed for validation
            lead("L4"),  # no client text -> nothing to analyse
            lead("L5"),  # AI: negative, high -> do_not_contact
            lead("L6"),  # AI: opt_out, medium -> manual_review
        ],
        [
            message("L1", "L1-M1"),
            message("L1", "L1-M2", direction="INBOUND", text="sure, stop by on Thursday"),
            message("L2", "L2-M1"),
            message("L2", "L2-M2", direction="INBOUND", text="STOP"),
            message("L3", "L3-M1"),
            message("L3", "L3-M2", direction="INBOUND", text="later"),
            message("L4", "L4-M1"),
            message("L5", "L5-M1"),
            message("L5", "L5-M2", direction="INBOUND", text="we went with someone else"),
            message("L6", "L6-M1"),
            message("L6", "L6-M2", direction="INBOUND", text="maybe don't text me"),
        ],
    )
    config = Config(
        phrases=Phrases(hints=("stop", "don't text")),
        ai_review=AiReview(enabled=True, model="test-model"),
    )
    reviewer = FakeReviewer(
        {
            "L1": _assessment(AiLabel.NONE),
            "L3": _assessment(AiLabel.NONE),
            "L5": _assessment(AiLabel.NEGATIVE, evidence=["L5-M2"]),
            "L6": _assessment(AiLabel.OPT_OUT, AiConfidence.MEDIUM, ["L6-M2"]),
        }
    )
    run = run_pipeline(config, read_export(workbook_path), reviewer=reviewer)

    assert [c for c, _ in reviewer.calls] == ["L1", "L3", "L5", "L6"]
    decisions = {r.case_id: r.decision for r in run.results}
    assert decisions == {
        "L1": Decision.CONTACT_CANDIDATE,
        "L2": Decision.DO_NOT_CONTACT,
        "L3": Decision.SKIP_NOT_DUE,
        "L4": Decision.MANUAL_REVIEW,
        "L5": Decision.DO_NOT_CONTACT,
        "L6": Decision.MANUAL_REVIEW,
    }
    assert set(run.ai_outcomes) == {"L1", "L3", "L5", "L6"}
    by_id = {r.case_id: r for r in run.results}
    assert by_id["L5"].evidence_message_ids == ("L5-M2",)
    assert by_id["L6"].evidence_message_ids == ("L6-M2",)
    assert by_id["L5"].draft_sms == "" and by_id["L6"].draft_sms == ""


def test_pipeline_without_ai_sends_hints_to_manual_review(workbook_path: Path) -> None:
    build_workbook(
        workbook_path,
        [lead("L1"), lead("L2")],
        [
            message("L1", "L1-M1", direction="INBOUND", text="stop by anytime"),
            message("L2", "L2-M1", direction="INBOUND", text="ok"),
        ],
    )
    reviewer = FakeReviewer(_assessment(AiLabel.OPT_OUT))
    run = run_pipeline(
        Config(phrases=Phrases(hints=("stop",))), read_export(workbook_path), reviewer=reviewer
    )
    assert reviewer.calls == []
    decisions = {r.case_id: r.decision for r in run.results}
    assert decisions == {"L1": Decision.MANUAL_REVIEW, "L2": Decision.CONTACT_CANDIDATE}
    assert run.ai_outcomes == {}


def test_reviewer_returning_a_wrong_type_becomes_an_error_outcome_not_a_crash() -> None:
    class BrokenReviewer:
        def assess(self, case_id: str, payload: str) -> AiAssessment:
            return {"label": "none"}  # type: ignore[return-value]

    outcome = review_case(_timeline(), BrokenReviewer(), AiReview())
    assert outcome.error is not None
    assert "AiAssessment" in outcome.error
    assert outcome.needs_review
