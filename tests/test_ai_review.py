"""AI second opinion: only candidates are sent, only client text, only downgrades."""

from __future__ import annotations

import json
from pathlib import Path

from silent_leads.config import AiReview, Config, Phrases
from silent_leads.io.reader import read_export
from silent_leads.models import Decision, Direction
from silent_leads.pipeline import run_pipeline
from silent_leads.rules.ai_review import (
    AiAssessment,
    AiLabel,
    AiOutcome,
    apply_ai_outcome,
    build_payload,
    review_candidate,
)
from silent_leads.rules.decide import Verdict
from silent_leads.timeline import build_timelines
from tests.conftest import build_workbook, lead, message
from tests.test_timeline import _lead, _msg

SETTINGS = AiReview(enabled=True, model="test-model", max_input_chars=500)


class FakeReviewer:
    def __init__(self, assessment: AiAssessment | Exception) -> None:
        self.assessment = assessment
        self.calls: list[tuple[str, str]] = []

    def assess(self, case_id: str, payload: str) -> AiAssessment:
        self.calls.append((case_id, payload))
        if isinstance(self.assessment, Exception):
            raise self.assessment
        return self.assessment


def _timeline():  # type: ignore[no-untyped-def]
    transcript = "Jane (C): Hi, I was waiting for a quote.\n[СПІВРОБІТНИК] (S): We will send it today."
    return build_timelines(
        [_lead()],
        [
            _msg("M1", "2026-10-01 09:00:00", text="Hello from sales"),
            _msg("M2", "2026-10-02 09:00:00", direction=Direction.INBOUND, text="sure, thanks"),
            _msg("M3", "2026-10-03 09:00:00", text=transcript, event_type="CALL"),
        ],
    )["L1"]


CANDIDATE = Verdict(Decision.CONTACT_CANDIDATE, "строк настав", (), "")


def test_payload_contains_only_client_utterances_with_ids() -> None:
    payload, truncated = build_payload(_timeline(), 500)
    items = json.loads(payload)
    assert [i["message_id"] for i in items] == ["M2", "M3"]
    assert items[1]["text"] == "Hi, I was waiting for a quote."
    assert "Hello from sales" not in payload
    assert "send it today" not in payload
    assert "СПІВРОБІТНИК" not in payload
    assert truncated is False


def test_payload_is_truncated_at_limit() -> None:
    payload, truncated = build_payload(_timeline(), 15)
    assert json.loads(payload) == [{"message_id": "M2", "text": "sure, thanks"}]
    assert truncated is True


def test_label_none_keeps_candidate_and_notes_it() -> None:
    reviewer = FakeReviewer(AiAssessment(label=AiLabel.NONE, explanation="Відмови немає."))
    outcome = review_candidate(_timeline(), reviewer, SETTINGS)
    verdict = apply_ai_outcome(CANDIDATE, outcome)
    assert verdict.decision == Decision.CONTACT_CANDIDATE
    assert "[AI test-model]" in verdict.reviewer_note
    assert reviewer.calls[0][0] == "L1"


def test_opt_out_label_downgrades_with_own_evidence_only() -> None:
    reviewer = FakeReviewer(
        AiAssessment(
            label=AiLabel.OPT_OUT, evidence_message_ids=["M2", "L9-M1"], explanation="Просить не писати."
        )
    )
    verdict = apply_ai_outcome(CANDIDATE, review_candidate(_timeline(), reviewer, SETTINGS))
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert verdict.evidence_message_ids == ("M2",)
    assert "можлива відписка" in verdict.reason
    assert "правила: строк настав" in verdict.reviewer_note


def test_unclear_label_downgrades() -> None:
    reviewer = FakeReviewer(AiAssessment(label=AiLabel.UNCLEAR, explanation="Текст фрагментарний."))
    verdict = apply_ai_outcome(CANDIDATE, review_candidate(_timeline(), reviewer, SETTINGS))
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "неоднозначний" in verdict.reason


def test_api_failure_downgrades_instead_of_crashing() -> None:
    reviewer = FakeReviewer(ConnectionError("network down"))
    outcome = review_candidate(_timeline(), reviewer, SETTINGS)
    assert outcome.error is not None and "network down" in outcome.error
    verdict = apply_ai_outcome(CANDIDATE, outcome)
    assert verdict.decision == Decision.MANUAL_REVIEW
    assert "не виконана" in verdict.reason


def test_ai_never_upgrades_other_decisions() -> None:
    blocked = Verdict(Decision.DO_NOT_CONTACT, "відписка", ("M2",), "")
    clean = AiOutcome(model="m", label=AiLabel.NONE, evidence_message_ids=(), explanation="ok")
    assert apply_ai_outcome(blocked, clean) == blocked
    skipped = Verdict(Decision.SKIP_NOT_DUE, "рано", (), "")
    assert apply_ai_outcome(skipped, clean) == skipped


def test_pipeline_calls_reviewer_only_for_candidates(workbook_path: Path) -> None:
    build_workbook(
        workbook_path,
        [
            lead("L1"),  # candidate
            lead("L2"),  # opt-out -> never reviewed
            lead("L3", silent_since_crm="2026-11-03 10:00:00"),  # not due -> never reviewed
        ],
        [
            message("L1", "L1-M1"),
            message("L1", "L1-M2", direction="INBOUND", text="ok, thanks"),
            message("L2", "L2-M1"),
            message("L2", "L2-M2", direction="INBOUND", text="STOP"),
            message("L3", "L3-M1"),
            message("L3", "L3-M2", direction="INBOUND", text="later"),
        ],
    )
    config = Config(phrases=Phrases(opt_out=("stop",)), ai_review=AiReview(enabled=True, model="test-model"))
    reviewer = FakeReviewer(
        AiAssessment(label=AiLabel.NEGATIVE, evidence_message_ids=["L1-M2"], explanation="x")
    )
    run = run_pipeline(config, read_export(workbook_path), reviewer=reviewer)

    assert [c for c, _ in reviewer.calls] == ["L1"]
    decisions = {r.case_id: r.decision for r in run.results}
    assert decisions == {
        "L1": Decision.MANUAL_REVIEW,
        "L2": Decision.DO_NOT_CONTACT,
        "L3": Decision.SKIP_NOT_DUE,
    }
    assert set(run.ai_outcomes) == {"L1"}
    by_id = {r.case_id: r for r in run.results}
    assert by_id["L1"].draft_sms == ""  # downgraded cases get no text
    assert by_id["L1"].evidence_message_ids == ("L1-M2",)


def test_pipeline_skips_reviewer_when_disabled(workbook_path: Path) -> None:
    build_workbook(workbook_path, [lead("L1")], [message("L1", "L1-M1", direction="INBOUND", text="ok")])
    reviewer = FakeReviewer(AiAssessment(label=AiLabel.OPT_OUT, explanation="x"))
    run = run_pipeline(Config(), read_export(workbook_path), reviewer=reviewer)
    assert reviewer.calls == []
    assert run.results[0].decision == Decision.CONTACT_CANDIDATE
    assert run.ai_outcomes == {}
