from __future__ import annotations

from pathlib import Path

from silent_leads.io.reader import read_export
from silent_leads.models import IssueLevel
from silent_leads.validation import check_control_sheet, validate_links
from tests.conftest import build_workbook, lead, message


def test_control_sheet_matches_clean_file(workbook_path: Path) -> None:
    build_workbook(
        workbook_path,
        [lead("L1", has_quote="yes", quote_at_crm="2026-09-02 09:00:00"), lead("L2")],
        [message("L1", "L1-M1"), message("L2", "L2-M1", content_state="withheld_hidden", text=None)],
    )
    export = read_export(workbook_path)
    assert check_control_sheet(export) == []
    assert validate_links(export) == []


def test_control_sheet_mismatch_is_reported(workbook_path: Path) -> None:
    build_workbook(
        workbook_path, [lead("L1")], [message("L1", "L1-M1")], control={"cases": 50, "message_rows": 1246}
    )
    export = read_export(workbook_path)
    mismatches = check_control_sheet(export)
    assert any(m.startswith("cases:") for m in mismatches)
    assert any(m.startswith("message_rows:") for m in mismatches)


def test_orphan_message_and_lead_without_history(workbook_path: Path) -> None:
    build_workbook(workbook_path, [lead("L1")], [message("L9", "L9-M1")], auto_counters=False)
    export = read_export(workbook_path)
    issues = validate_links(export)
    codes = {(i.code, i.case_id) for i in issues}
    assert ("message_orphan", "L9") in codes
    assert ("lead_without_history", "L1") in codes
    assert all(i.level == IssueLevel.ERROR for i in issues)


def test_counter_mismatch_is_an_error(workbook_path: Path) -> None:
    build_workbook(
        workbook_path,
        [lead("L1", events_count=5, text_events=1, hidden_events=0, calls_without_text=0)],
        [message("L1", "L1-M1")],
        auto_counters=False,
    )
    export = read_export(workbook_path)
    issues = validate_links(export)
    assert [i.code for i in issues] == ["lead_events_count_mismatch"]
    assert issues[0].level == IssueLevel.ERROR


def test_duplicate_ids_are_errors(workbook_path: Path) -> None:
    build_workbook(
        workbook_path,
        [lead("L1"), lead("L1")],
        [message("L1", "L1-M1"), message("L2", "L1-M1")],
        auto_counters=False,
    )
    export = read_export(workbook_path)
    codes = {i.code for i in validate_links(export)}
    assert "lead_duplicate_case_id" in codes
    assert "message_duplicate_id" in codes
