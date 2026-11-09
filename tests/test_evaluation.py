"""Validation tooling: human labels vs AI labels, and the synthetic eval runner."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from silent_leads.ai_eval import EvalCase, EvalSet, EvalTurn, case_items, load_eval_set, run_ai_eval
from silent_leads.cli import app
from silent_leads.config import AiReview
from silent_leads.evaluation import (
    EvalRow,
    evaluate,
    normalize_label,
    read_labels_csv,
    render_evaluation_markdown,
)
from silent_leads.io.writer import AI_LABEL_COLUMNS
from silent_leads.rules.ai_review import AiAssessment, AiConfidence, AiLabel
from silent_leads.timeline import Speaker
from tests.test_ai_review import FakeReviewer

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "config.example.yaml"

ROOT = Path(__file__).resolve().parents[1]


def _row(case_id: str, expected: str | None, predicted: str | None, confidence: str = "high") -> EvalRow:
    return EvalRow(case_id, expected, predicted, confidence, "чому", "", expected or "")


def test_metrics_exact_and_decision_level() -> None:
    rows = [
        _row("A", "none", "none"),
        _row("B", "opt_out", "opt_out"),
        _row("C", "opt_out", "negative", "medium"),  # wrong label, same decision
        _row("D", "none", "opt_out"),  # false refusal
        _row("E", "negative", "none"),  # missed refusal
        _row("F", "unclear", "none"),  # unclear is excluded from decision-level
        _row("G", None, "none"),  # not labelled by the human
        _row("H", "opt_out", None),  # AI did not answer
        _row("I", "maybe", "none"),  # invalid human label
    ]
    report = evaluate(rows)
    assert report["rows_total"] == 9
    assert report["labelled"] == 7
    assert report["comparable"] == 6
    assert report["not_predicted"] == ["H"]
    assert report["invalid_labels"] == ["I"]
    assert report["exact_correct"] == 2 and report["exact_accuracy"] == 0.3333  # rounded to 4 places
    assert report["decision_compared"] == 5 and report["decision_accuracy"] == 0.6
    assert report["missed_refusals"] == ["E"] and report["false_refusals"] == ["D"]
    assert report["confusion"]["opt_out"]["negative"] == 1
    assert report["by_confidence"]["medium"] == {"total": 1, "correct": 0, "accuracy": 0.0}
    assert [m["case_id"] for m in report["mismatches"]] == ["C", "D", "E", "F"]
    markdown = render_evaluation_markdown(report, "T")
    assert "33.3%" in markdown and "60.0%" in markdown and "**C**" in markdown


def test_empty_report_has_no_division_by_zero() -> None:
    report = evaluate([])
    assert report["exact_accuracy"] is None and report["decision_accuracy"] is None
    assert "—" in render_evaluation_markdown(report, "T")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Opt_Out", "opt_out"),
        (" opt-out ", "opt_out"),
        ("Відписка", "opt_out"),
        ("", None),
        (None, None),
        ("x", "x"),
    ],
)
def test_label_normalization(raw: str | None, expected: str | None) -> None:
    assert normalize_label(raw) == expected


def test_read_labels_csv_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "ai_labels.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(AI_LABEL_COLUMNS)
        writer.writerow(["L1", "contact_candidate", "none", "high", "", "ok", "", "none", ""])
        writer.writerow(["L2", "manual_review", "", "", "", "", "RuntimeError: x", "opt_out", "clear stop"])
    rows = read_labels_csv(path)
    assert rows[0] == EvalRow("L1", "none", "none", "high", "ok", "", "none")
    assert rows[1].predicted is None and rows[1].expected == "opt_out" and rows[1].error == "RuntimeError: x"


def test_cli_evaluate_writes_report(tmp_path: Path) -> None:
    path = tmp_path / "ai_labels.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(AI_LABEL_COLUMNS)
        writer.writerow(["L1", "contact_candidate", "none", "high", "", "", "", "none", ""])
        writer.writerow(["L2", "manual_review", "unclear", "low", "", "", "", "none", ""])
    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["evaluate-labels", "--labels", str(path), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert "50.0%" in result.output
    assert (out / "evaluation.md").exists() and (out / "evaluation.json").exists()


# --- synthetic eval set ---


def test_bundled_eval_set_is_valid_and_balanced() -> None:
    eval_set = load_eval_set(ROOT / "evals" / "context_cases.yaml")
    labels = {c.expected for c in eval_set.cases}
    assert labels == set(AiLabel)
    assert len(eval_set.cases) >= 20
    assert all(
        any(t.speaker == Speaker.CLIENT or t.speaker == Speaker.UNKNOWN for t in c.dialogue)
        for c in eval_set.cases
    )


def test_case_items_get_synthetic_ids() -> None:
    case = EvalCase(
        id="X1",
        expected=AiLabel.UNCLEAR,
        dialogue=[EvalTurn(speaker=Speaker.SALES, text="hi"), EvalTurn(speaker=Speaker.CLIENT, hidden=True)],
    )
    prepared = case_items(case)
    assert prepared.items == [
        {"message_id": "X1-1", "speaker": "sales", "text": "hi"},
        {"message_id": "X1-2", "speaker": "client", "hidden": True},
    ]
    assert prepared.own_ids == {"X1-1", "X1-2"}


def test_run_ai_eval_scores_against_expected() -> None:
    eval_set = EvalSet(
        cases=[
            EvalCase(id="A", expected=AiLabel.NONE, dialogue=[EvalTurn(speaker=Speaker.CLIENT, text="ok")]),
            EvalCase(
                id="B", expected=AiLabel.OPT_OUT, dialogue=[EvalTurn(speaker=Speaker.CLIENT, text="stop it")]
            ),
        ]
    )
    reviewer = FakeReviewer(
        {
            "A": AiAssessment(label=AiLabel.NONE, confidence=AiConfidence.HIGH, explanation="x"),
            "B": AiAssessment(
                label=AiLabel.NONE,
                confidence=AiConfidence.LOW,
                evidence_message_ids=["B-1", "Z-9"],
                explanation="y",
            ),
        }
    )
    rows = run_ai_eval(eval_set, reviewer, AiReview(enabled=True, model="m"))
    assert [r.case_id for r in rows] == ["A", "B"]
    report = evaluate(rows)
    assert report["exact_accuracy"] == 0.5
    assert report["missed_refusals"] == ["B"]
    assert reviewer.calls[1][0] == "B"


def test_duplicate_eval_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text(
        "cases:\n"
        "  - {id: A, expected: none, dialogue: [{speaker: client, text: hi}]}\n"
        "  - {id: A, expected: none, dialogue: [{speaker: client, text: hi}]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_set(path)


def test_cli_ai_eval_refuses_when_ai_disabled(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["ai-eval", "--config", str(EXAMPLE_CONFIG), "--output", str(tmp_path / "o")]
    )
    assert result.exit_code == 5


def test_labels_csv_saved_by_excel_with_semicolons_is_read(tmp_path: Path) -> None:
    path = tmp_path / "ai_labels.csv"
    path.write_text(
        "\ufeff" + ";".join(AI_LABEL_COLUMNS) + "\n"
        "L1;contact_candidate;none;high;;ok;;none;\n"
        "L2;do_not_contact;negative;high;L2-M1;refused;;opt_out;close enough\n",
        encoding="utf-8",
    )
    rows = read_labels_csv(path)
    assert [r.case_id for r in rows] == ["L1", "L2"]
    assert rows[0].expected == "none"
    assert rows[1].expected == "opt_out"
    assert rows[1].predicted == "negative"
