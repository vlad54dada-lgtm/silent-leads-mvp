"""v2 end to end on a synthetic workbook: labels sheet, AI drafts only for candidates, hard draft limits,
reference-label evaluation, rerun stability. The AI parts are fakes; nothing leaves the machine."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from openpyxl import load_workbook
from typer.testing import CliRunner

from silent_leads.cli import app
from silent_leads.config import AiReview, Config, Drafts, Phrases, load_config
from silent_leads.io.reader import read_export
from silent_leads.io.writer import write_outputs
from silent_leads.models import Decision, PolicyLabel
from silent_leads.pipeline import RunResult, run_pipeline
from silent_leads.policy_eval import (
    PolicyRow,
    evaluate_policy,
    join_rows,
    load_policy_set,
    read_reference_labels,
    read_tool_labels,
    render_policy_markdown,
    run_policy_eval,
)
from silent_leads.rules.ai_review import AiConfidence
from silent_leads.rules.contact_policy import AiReason, PolicyAssessment
from silent_leads.rules.drafts import DraftAnswer, DraftStatus, check_draft
from silent_leads.summary import build_summary, render_summary_markdown
from tests.conftest import build_workbook, lead, message

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Config(
    policy="v2",
    phrases=Phrases(exact_opt_out=("stop",), hints=()),
    ai_review=AiReview(enabled=True, model="test-model"),
    drafts=Drafts(enabled=True),
)

# case -> what the fake judge answers (label, reason, position)
ANSWERS = {
    "L1": (PolicyLabel.CONTACT_OK, AiReason.PRICE, "L1-M2"),  # due -> candidate + draft
    "L2": (PolicyLabel.DO_NOT_CONTACT, AiReason.HIRED_OTHER_OR_DONE, "L2-M2"),  # blocked
    "L3": (PolicyLabel.CONTACT_OK, AiReason.POSTPONE_OR_COMPARING, "L3-M2"),  # not due -> skip
    "L4": (PolicyLabel.CONTACT_OK, AiReason.SILENT, "L4-M2"),  # hidden conversation after -> need_human
    "L6": (PolicyLabel.CONTACT_OK, AiReason.PRICE, "L6-M2"),  # candidate, bad draft -> rejected
}


class FakeJudge:
    def __init__(self) -> None:
        self.cases: list[str] = []

    def judge(self, case_id: str, payload: str) -> PolicyAssessment:
        self.cases.append(case_id)
        label, reason, position = ANSWERS[case_id]
        return PolicyAssessment(
            label=label,
            reason=reason,
            confidence=AiConfidence.HIGH,
            last_client_position_message_id=position,
            evidence_message_ids=[position],
            explanation="синтетична оцінка",
        )


class FakeWriter:
    def __init__(self) -> None:
        self.cases: list[str] = []

    def write(self, case_id: str, content: str) -> DraftAnswer:
        self.cases.append(case_id)
        if case_id == "L6":
            return DraftAnswer(sms="We can take 15% off if you sign by Friday!")
        return DraftAnswer(
            sms="Hi! I understand the budget concern for your hallway. Want to go over some options?"
        )


def _build(path: Path, **extra: object) -> Path:
    return build_workbook(
        path,
        [
            lead("L1", has_quote="yes", quote_at_crm="2026-09-02 09:00:00"),
            lead("L2"),
            lead("L3", silent_since_crm="2026-11-03 10:00:00"),
            lead("L4"),
            lead("L5"),  # client never wrote -> silent by rule, no model call
            lead("L6"),
            lead("L7"),  # exact STOP -> blocked by rule, no model call
        ],
        [
            message("L1", "L1-M1"),
            message("L1", "L1-M2", direction="INBOUND", text="Too pricey for us right now."),
            message("L2", "L2-M1"),
            message("L2", "L2-M2", direction="INBOUND", text="We went with another company."),
            message("L3", "L3-M1"),
            message("L3", "L3-M2", direction="INBOUND", text="Let me compare quotes first."),
            message("L4", "L4-M1"),
            message("L4", "L4-M2", direction="INBOUND", text="ok"),
            message(
                "L4",
                "L4-M3",
                occurred_at_crm="2026-10-05 10:00:00",
                event_type="CALL",
                call_result="SUCCESS",
                content_state="withheld_hidden",
                text=None,
                hidden_kind="conversation",
            ),
            message("L5", "L5-M1"),
            message("L6", "L6-M1"),
            message("L6", "L6-M2", direction="INBOUND", text="Hmm, that is more than we planned."),
            message("L7", "L7-M1"),
            message("L7", "L7-M2", direction="INBOUND", text="STOP"),
        ],
        **extra,  # type: ignore[arg-type]
    )


def _run(path: Path) -> tuple[RunResult, FakeJudge, FakeWriter]:
    judge, writer = FakeJudge(), FakeWriter()
    return run_pipeline(CONFIG, read_export(path), judge=judge, draft_writer=writer), judge, writer


def test_v2_pipeline_labels_decisions_and_drafts(workbook_path: Path) -> None:
    result, judge, writer = _run(_build(workbook_path))
    decisions = {r.case_id: r.decision for r in result.results}
    assert decisions == {
        "L1": Decision.CONTACT_CANDIDATE,
        "L2": Decision.DO_NOT_CONTACT,
        "L3": Decision.SKIP_NOT_DUE,
        "L4": Decision.MANUAL_REVIEW,
        "L5": Decision.CONTACT_CANDIDATE,
        "L6": Decision.CONTACT_CANDIDATE,
        "L7": Decision.DO_NOT_CONTACT,
    }
    assert sorted(judge.cases) == ["L1", "L2", "L3", "L4", "L6"]  # L5, L7 decided by code
    assert result.verdicts["L4"].label == PolicyLabel.NEED_HUMAN
    assert sorted(writer.cases) == ["L1", "L5", "L6"]  # drafts only for candidates
    assert result.drafts["L1"].status == DraftStatus.AI_DRAFT
    assert result.drafts["L6"].status == DraftStatus.REJECTED and result.drafts["L6"].text == ""
    # the «Результат» template columns keep following the configured templates
    assert all(r.draft_sms == "" for r in result.results)


def test_v2_outputs_have_label_and_draft_sheets_and_rerun_is_byte_identical(
    workbook_path: Path, tmp_path: Path
) -> None:
    path = _build(workbook_path)

    def write(out: Path) -> list[Path]:
        result, _, _ = _run(path)
        summary = build_summary(CONFIG, result)
        return write_outputs(
            out,
            result.results,
            result.issues,
            summary,
            render_summary_markdown(summary),
            result.ai_outcomes,
            result.verdicts,
            result.policy_outcomes,
            result.drafts,
        )

    first = write(tmp_path / "a")
    second = write(tmp_path / "b")
    assert [p.name for p in first] == [
        "results.xlsx",
        "results.csv",
        "data_issues.csv",
        "labels.csv",
        "drafts.csv",
        "summary.json",
        "summary.md",
    ]
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name
    wb = load_workbook(first[0], read_only=True)
    assert wb.sheetnames == ["Результат", "Проблеми даних", "Мітки", "Чернетки SMS"]
    drafts = list(wb["Чернетки SMS"].iter_rows(values_only=True))
    assert [row[0] for row in drafts[1:]] == ["L1", "L5", "L6"]
    with first[3].open(encoding="utf-8-sig") as fh:
        labels = {row["case_id"]: row for row in csv.DictReader(fh)}
    assert labels["L2"]["label"] == "do_not_contact"
    assert labels["L4"]["reason"] == "hidden_conversation"
    assert labels["L7"]["confidence"] == "rule"
    assert "Чернетки SMS" in first[-1].read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("", "порожній"),
        ("Save 20 dollars today", "\\d"),
        ("Only $ left", "\\$"),
        ("See https://example.com", "https?://"),
        ("Hi [ІМʼЯ], any news?", "\\["),
        ("x" * 400, "довжина"),
    ],
)
def test_draft_hard_limits(text: str, problem: str) -> None:
    problems = check_draft(text, Drafts())
    assert any(problem in p for p in problems), problems


def test_clean_draft_passes() -> None:
    assert check_draft("Hi! Any thoughts on the quote for your living room?", Drafts()) == []


def test_drafts_need_the_ai_switch(workbook_path: Path) -> None:
    config = CONFIG.model_copy(update={"ai_review": AiReview(enabled=False)})
    writer = FakeWriter()
    result = run_pipeline(config, read_export(_build(workbook_path)), draft_writer=writer)
    assert writer.cases == [] and result.drafts == {}


def test_invalid_forbidden_pattern_is_a_config_error() -> None:
    with pytest.raises(ValueError, match="invalid regex"):
        Drafts(forbidden_patterns=("[unclosed",))


# --------------------------------------------------------------------------- evaluation


def test_reference_labels_come_from_both_sheets_and_labels_sheet_wins(workbook_path: Path) -> None:
    path = _build(
        workbook_path,
        reference_labels={"L1": "contact_ok", "L2": "do_not_contact", "L3": ""},
        examples={"L2": "contact_ok", "L4": "need_human", "L5": "(не впливає)"},
    )
    reference = read_reference_labels(path)
    assert reference == {
        "L1": ("contact_ok", "Розмітка"),
        "L2": ("do_not_contact", "Розмітка"),
        "L4": ("need_human", "Приклади"),
    }


def test_evaluation_counts_missed_and_false_refusals() -> None:
    rows = [
        PolicyRow("A", "do_not_contact", "contact_ok", final_decision="contact_candidate"),
        PolicyRow("B", "do_not_contact", "need_human", final_decision="skip_not_due"),  # passes through
        PolicyRow("C", "do_not_contact", "do_not_contact", final_decision="do_not_contact"),
        PolicyRow("D", "contact_ok", "do_not_contact", final_decision="do_not_contact"),
        PolicyRow("E", "contact_ok", "contact_ok", final_decision="contact_candidate"),
        PolicyRow("F", "need_human", None),
    ]
    report = evaluate_policy(rows)
    assert report["compared"] == 5
    assert report["accuracy"] == pytest.approx(0.4)
    assert report["missed_refusals"] == ["A", "B"]
    assert report["false_refusals"] == ["D"]
    assert report["missing_in_tool"] == ["F"]
    assert report["confusion"]["do_not_contact"]["contact_ok"] == 1


def test_cli_run_and_evaluate_policy_labels(workbook_path: Path, tmp_path: Path) -> None:
    path = _build(workbook_path, reference_labels={"L2": "do_not_contact", "L7": "do_not_contact"})
    config = tmp_path / "v2.yaml"
    example = (ROOT / "config.example.yaml").read_text(encoding="utf-8")
    config.write_text(example, encoding="utf-8")
    assert load_config(config).policy == "v2"  # the documented baseline is v2, AI off
    out = tmp_path / "out"
    runner = CliRunner()
    ran = runner.invoke(app, ["run", "-i", str(path), "-c", str(config), "-o", str(out)])
    assert ran.exit_code == 0, ran.output
    assert "Політика v2" in ran.output
    evaluated = runner.invoke(
        app,
        [
            "evaluate-policy-labels",
            "-i",
            str(path),
            "-c",
            str(config),
            "-l",
            str(out / "labels.csv"),
            "-o",
            str(out),
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output
    tool = read_tool_labels(out / "labels.csv")
    report = evaluate_policy(join_rows(read_reference_labels(path), tool))
    # offline: L7 blocked by the exact STOP rule, L2 needs the model -> need_human, never let through
    assert report["missed_refusals"] == []
    assert (out / "policy_evaluation.md").exists()


def test_tool_labels_csv_saved_by_excel_with_semicolons_is_read(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("case_id;label;confidence\nL1;contact_ok;high\n", encoding="utf-8-sig")
    assert read_tool_labels(path)["L1"]["label"] == "contact_ok"


def test_synthetic_policy_set_is_valid_and_covers_every_label_and_reason() -> None:
    policy_set = load_policy_set(ROOT / "evals" / "policy_cases.yaml")
    assert {c.expected for c in policy_set.cases} == set(PolicyLabel)
    reasons = {c.reason for c in policy_set.cases if c.reason is not None}
    assert reasons >= set(AiReason) - {AiReason.UNCLEAR}


def test_boundary_case_allowed_with_high_confidence_is_reported() -> None:
    """A boundary case may keep the label contact_ok, but only below high confidence."""
    rows = [
        PolicyRow("B1", "contact_ok", "contact_ok", confidence="high", must_not_be_candidate=True),
        PolicyRow("B2", "contact_ok", "contact_ok", confidence="medium", must_not_be_candidate=True),
        PolicyRow("B3", "do_not_contact", "do_not_contact", confidence="high", must_not_be_candidate=True),
        PolicyRow("B4", "contact_ok", "contact_ok", confidence="high"),
    ]
    report = evaluate_policy(rows)
    assert report["guarded"] == 3
    assert report["overconfident"] == ["B1"]
    assert report["accuracy"] == 1.0  # the label is right; the confidence is what fails
    assert "(B1)" in render_policy_markdown(report, "t")


def test_report_without_boundary_cases_does_not_mention_them() -> None:
    report = evaluate_policy([PolicyRow("A1", "contact_ok", "contact_ok", confidence="high")])
    assert report["guarded"] == 0
    assert "Межові кейси" not in render_policy_markdown(report, "t")


def test_synthetic_set_marks_its_boundary_cases() -> None:
    policy_set = load_policy_set(ROOT / "evals" / "policy_cases.yaml")
    guarded = {c.id for c in policy_set.cases if c.not_candidate}
    assert guarded  # the audit of agreed labels added them
    rows = run_policy_eval(policy_set, _HighContactOk(), AiReview(enabled=True))
    assert set(evaluate_policy(rows)["overconfident"]) == guarded


class _HighContactOk:
    def judge(self, case_id: str, payload: str) -> PolicyAssessment:
        return PolicyAssessment(
            label=PolicyLabel.CONTACT_OK,
            reason=AiReason.SILENT,
            confidence=AiConfidence.HIGH,
            explanation="x",
        )


def test_policy_eval_runs_the_judge_on_synthetic_ids() -> None:
    policy_set = load_policy_set(ROOT / "evals" / "policy_cases.yaml")

    class EchoJudge:
        def judge(self, case_id: str, payload: str) -> PolicyAssessment:
            assert f"{case_id}-1" in payload
            return PolicyAssessment(
                label=PolicyLabel.CONTACT_OK,
                reason=AiReason.SILENT,
                confidence=AiConfidence.HIGH,
                explanation="x",
            )

    rows = run_policy_eval(policy_set, EchoJudge(), AiReview(enabled=True))
    assert len(rows) == len(policy_set.cases)
    assert all(r.predicted == "contact_ok" for r in rows)
