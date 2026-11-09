"""End-to-end on a synthetic workbook: every lead gets one row, rerun is byte-identical."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from openpyxl import load_workbook
from typer.testing import CliRunner

from silent_leads.cli import app
from silent_leads.config import Config, Phrases, Template, Templates
from silent_leads.io.reader import read_export
from silent_leads.io.writer import write_outputs
from silent_leads.models import TemplateStatus
from silent_leads.pipeline import run_pipeline
from silent_leads.summary import build_summary, render_summary_markdown
from tests.conftest import build_workbook, lead, message

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "config.example.yaml"

CONFIG = Config(
    phrases=Phrases(exact_opt_out=("stop",), hints=("not interested",)),
    templates=Templates(
        with_quote=Template(text="[TEST] quote follow-up", status=TemplateStatus.TEST_TEMPLATE),
        without_quote=Template(text="", status=TemplateStatus.TEMPLATE_MISSING),
    ),
)


def _build(path: Path) -> Path:
    return build_workbook(
        path,
        [
            lead("L1", has_quote="yes", quote_at_crm="2026-09-02 09:00:00"),  # due, clean -> candidate
            lead("L2", has_quote="no"),  # due, clean, no template -> candidate + template_missing
            lead("L3", silent_since_crm="2026-11-03 10:00:00"),  # not due
            lead("L4"),  # opt-out
            lead("L5"),  # contact after silent
            lead("L6", silent_since_crm="bad date"),  # data error
        ],
        [
            message("L1", "L1-M1"),
            message("L1", "L1-M2", direction="INBOUND", text="sure, send it over"),
            message("L2", "L2-M1"),
            message("L2", "L2-M2", direction="INBOUND", text="ok"),
            message("L3", "L3-M1"),
            message("L3", "L3-M2", direction="INBOUND", text="maybe later"),
            message("L4", "L4-M1"),
            message("L4", "L4-M2", direction="INBOUND", text="STOP"),
            message("L5", "L5-M1"),
            message("L5", "L5-M2", direction="INBOUND", text="hi"),
            message("L5", "L5-M3", occurred_at_crm="2026-10-25 10:00:00", text="following up"),
            message("L6", "L6-M1"),
            message("L6", "L6-M2", direction="INBOUND", text="hello"),
        ],
    )


def test_every_lead_gets_exactly_one_row_with_expected_decision(workbook_path: Path, tmp_path: Path) -> None:
    export = read_export(_build(workbook_path))
    run = run_pipeline(CONFIG, export)

    decisions = {r.case_id: r.decision.value for r in run.results}
    assert decisions == {
        "L1": "contact_candidate",
        "L2": "contact_candidate",
        "L3": "skip_not_due",
        "L4": "do_not_contact",
        "L5": "manual_review",
        "L6": "manual_review",
    }
    by_id = {r.case_id: r for r in run.results}
    assert by_id["L1"].draft_sms == "[TEST] quote follow-up"
    assert by_id["L1"].template_status == TemplateStatus.TEST_TEMPLATE
    assert by_id["L2"].draft_sms == ""
    assert by_id["L2"].template_status == TemplateStatus.TEMPLATE_MISSING
    assert by_id["L4"].evidence_message_ids == ("L4-M2",)
    assert by_id["L5"].evidence_message_ids == ("L5-M3",)
    assert by_id["L6"].draft_sms == ""
    assert "lead_bad_silent_since_crm" in by_id["L6"].reason
    for r in run.results:
        if r.decision.value != "contact_candidate":
            assert r.draft_sms == ""

    summary = build_summary(CONFIG, run)
    assert summary["cases_total"] == 6
    assert summary["decisions"] == {
        "contact_candidate": 2,
        "skip_not_due": 1,
        "do_not_contact": 1,
        "manual_review": 2,
    }
    assert summary["manual_review_cases"] == ["L5", "L6"]
    assert summary["data_issues"]["errors"] == 1
    assert summary["control_date"] == "2026-11-05"
    assert "Зведення" in render_summary_markdown(summary)


def test_outputs_are_written_and_rerun_is_byte_identical(workbook_path: Path, tmp_path: Path) -> None:
    export = read_export(_build(workbook_path))
    run = run_pipeline(CONFIG, export)
    summary = build_summary(CONFIG, run)
    out = tmp_path / "out"

    first = write_outputs(
        out, run.results, run.issues, summary, render_summary_markdown(summary), run.ai_outcomes
    )
    snapshot1 = {p.name: p.read_bytes() for p in first}

    # second run on the same input/config, later than the 2-second zip timestamp granularity
    time.sleep(2.1)
    run2 = run_pipeline(CONFIG, read_export(workbook_path))
    summary2 = build_summary(CONFIG, run2)
    second = write_outputs(out, run2.results, run2.issues, summary2, render_summary_markdown(summary2))
    snapshot2 = {p.name: p.read_bytes() for p in second}

    assert snapshot1 == snapshot2
    assert set(snapshot1) == {
        "results.xlsx",
        "results.csv",
        "data_issues.csv",
        "ai_labels.csv",
        "summary.json",
        "summary.md",
    }

    with (out / "results.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["case_id"] for r in rows] == ["L1", "L2", "L3", "L4", "L5", "L6"]
    assert len({r["case_id"] for r in rows}) == 6
    assert rows[3]["decision"] == "do_not_contact"
    assert rows[3]["evidence_message_ids"] == "L4-M2"

    ws = load_workbook(out / "results.xlsx")["Результат"]
    header = [c.value for c in ws[1]]
    assert header == [
        "case_id",
        "decision",
        "reason",
        "evidence_message_ids",
        "draft_sms",
        "template_status",
        "reviewer_note",
    ]
    assert ws.max_row == 7

    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert data["decisions"]["do_not_contact"] == 1


def test_cli_run_and_validate(workbook_path: Path, tmp_path: Path) -> None:
    _build(workbook_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "first_touch_delay_days: 7\nphrases:\n  exact_opt_out: ['stop']\n", encoding="utf-8"
    )
    runner = CliRunner()

    result = runner.invoke(app, ["validate", "--input", str(workbook_path), "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "2026-11-05" in result.output

    out = tmp_path / "out"
    result = runner.invoke(
        app, ["run", "--input", str(workbook_path), "--config", str(config_path), "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert (out / "results.xlsx").exists()
    assert "do_not_contact: 1" in result.output


def test_cli_stops_on_checksum_mismatch(workbook_path: Path, tmp_path: Path) -> None:
    build_workbook(workbook_path, [lead("L1")], [message("L1", "L1-M1")], control={"cases": 99})
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(workbook_path),
            "--config",
            str(EXAMPLE_CONFIG),
            "--output",
            str(tmp_path / "o"),
        ],
    )
    assert result.exit_code == 3
    assert not (tmp_path / "o").exists()


def test_cli_run_refuses_to_work_without_a_config(workbook_path: Path, tmp_path: Path) -> None:
    """The bare Config() has no phrase hints; an implicit default would silently weaken the rules."""
    build_workbook(workbook_path, [lead("L1")], [message("L1", "L1-M1")])
    result = CliRunner().invoke(app, ["run", "--input", str(workbook_path), "--output", str(tmp_path / "o")])
    assert result.exit_code == 2
    assert not (tmp_path / "o").exists()
