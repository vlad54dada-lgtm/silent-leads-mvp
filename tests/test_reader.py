from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from silent_leads.io.reader import ExportFormatError, read_export
from silent_leads.models import ContentState, Direction, IssueLevel, Role
from tests.conftest import build_workbook, lead, message


def test_reads_leads_and_messages_with_types(workbook_path: Path) -> None:
    build_workbook(
        workbook_path,
        [lead("L1", has_quote="yes", quote_at_crm="2026-09-02 09:00:00")],
        [
            message(
                "L1", "L1-M1", event_type="CALL", direction="INBOUND", call_result="SUCCESS", call_seconds=85
            ),
            message("L1", "L1-M2", occurred_at_crm="2026-10-02 10:00:00"),
        ],
    )
    export = read_export(workbook_path)

    assert [lead_.case_id for lead_ in export.leads] == ["L1"]
    only = export.leads[0]
    assert only.has_quote is True
    assert only.silent_since_crm == datetime(2026, 10, 20, 10, 0, 0)
    assert only.quote_at_crm == datetime(2026, 9, 2, 9, 0, 0)
    assert only.events_count == 2

    by_id = {m.message_id: m for m in export.messages}
    assert by_id["L1-M1"].direction == Direction.INBOUND
    assert by_id["L1-M1"].role == Role.CLIENT
    assert by_id["L1-M1"].call_seconds == 85
    assert by_id["L1-M1"].content_state == ContentState.TEXT_AVAILABLE
    assert export.control["cases"] == "1"
    assert export.instruction["Знімок UTC"].startswith("2026-11-05")
    assert export.issues == []


def test_multipart_rows_are_merged_in_part_order(workbook_path: Path) -> None:
    build_workbook(
        workbook_path,
        [lead("L1")],
        [
            message("L1", "L1-M1", text_part="2/2", text="second half"),
            message("L1", "L1-M1", text_part="1/2", text="first half"),
        ],
    )
    export = read_export(workbook_path)

    assert len(export.messages) == 1
    merged = export.messages[0]
    assert merged.text == "first half\nsecond half"
    assert merged.parts_expected == 2
    assert merged.parts_found == 2
    assert export.issues == []


def test_incomplete_multipart_is_flagged_not_dropped(workbook_path: Path) -> None:
    build_workbook(workbook_path, [lead("L1")], [message("L1", "L1-M1", text_part="1/3", text="only part")])
    export = read_export(workbook_path)

    assert export.messages[0].text == "only part"
    codes = {i.code for i in export.issues}
    assert "message_incomplete_parts" in codes


def test_bad_values_become_none_plus_issue(workbook_path: Path) -> None:
    build_workbook(
        workbook_path,
        [lead("L1", silent_since_crm="not a date", has_quote="maybe")],
        [
            message(
                "L1",
                "L1-M1",
                occurred_at_crm="??",
                direction="SIDEWAYS",
                content_state="text_available",
                text=None,
            )
        ],
    )
    export = read_export(workbook_path)

    assert export.leads[0].silent_since_crm is None
    assert export.leads[0].has_quote is None
    assert export.messages[0].occurred_at_crm is None
    assert export.messages[0].direction is None

    by_code = {i.code: i for i in export.issues}
    assert by_code["lead_bad_silent_since_crm"].level == IssueLevel.ERROR
    assert by_code["lead_bad_has_quote"].level == IssueLevel.ERROR
    assert by_code["message_bad_occurred_at"].level == IssueLevel.ERROR
    assert by_code["message_bad_direction"].level == IssueLevel.WARNING
    assert by_code["message_text_missing"].case_id == "L1"
    assert by_code["message_text_missing"].message_id == "L1-M1"


def test_role_direction_mismatch_is_flagged(workbook_path: Path) -> None:
    build_workbook(workbook_path, [lead("L1")], [message("L1", "L1-M1", direction="INBOUND", role="sales")])
    export = read_export(workbook_path)
    assert any(i.code == "message_role_mismatch" for i in export.issues)


def test_missing_sheet_or_column_is_a_format_error(tmp_path: Path, workbook_path: Path) -> None:
    build_workbook(workbook_path, [lead("L1")], [])
    from openpyxl import load_workbook

    wb = load_workbook(workbook_path)
    wb["Повідомлення"].delete_cols(2)  # drop message_id
    wb.save(workbook_path)
    with pytest.raises(ExportFormatError, match="message_id"):
        read_export(workbook_path)

    wb = load_workbook(workbook_path)
    del wb["Ліди"]
    wb.save(workbook_path)
    with pytest.raises(ExportFormatError, match="Ліди"):
        read_export(workbook_path)


def test_incomplete_inbound_multipart_is_an_error_not_a_warning(workbook_path: Path) -> None:
    build_workbook(
        workbook_path,
        [lead("L1")],
        [
            message(
                "L1", "L1-M1", direction="INBOUND", text_part="1/2", text="only half of what the client said"
            )
        ],
    )
    export = read_export(workbook_path)
    issue = next(i for i in export.issues if i.code == "message_incomplete_parts")
    assert issue.level == IssueLevel.ERROR
    assert issue.case_id == "L1"
