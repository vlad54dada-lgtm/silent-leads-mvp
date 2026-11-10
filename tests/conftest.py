"""Synthetic workbook builder. No real export data may ever appear here."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook

LEAD_HEADER = [
    "case_id",
    "team_alias",
    "sales_alias",
    "status",
    "created_at_crm",
    "silent_since_crm",
    "silent_date_source",
    "has_quote",
    "quote_at_crm",
    "last_inbound_at_crm",
    "last_outbound_at_crm",
    "events_count",
    "text_events",
    "hidden_events",
    "calls_without_text",
    "client_timezone",
    "timezone_basis",
]
MESSAGE_HEADER = [
    "case_id",
    "message_id",
    "occurred_at_crm",
    "event_type",
    "direction",
    "role",
    "call_result",
    "call_seconds",
    "content_state",
    "text_part",
    "text",
    "hidden_kind",
]

SNAPSHOT_UTC = "2026-11-05T14:53:41Z"  # 06:53:41 in America/Los_Angeles -> control date 2026-11-05


def lead(case_id: str, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "case_id": case_id,
        "team_alias": "T_TEST",
        "sales_alias": "S00",
        "status": "junk silent",
        "created_at_crm": "2026-09-01 10:00:00",
        "silent_since_crm": "2026-10-20 10:00:00",
        "silent_date_source": "crm_export_lead.junk_at",
        "has_quote": "no",
        "quote_at_crm": None,
        "last_inbound_at_crm": None,
        "last_outbound_at_crm": None,
        "events_count": None,
        "text_events": None,
        "hidden_events": None,
        "calls_without_text": None,
        "client_timezone": "America/Los_Angeles",
        "timezone_basis": "state",
    }
    base.update(overrides)
    return base


def message(case_id: str, message_id: str, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "case_id": case_id,
        "message_id": message_id,
        "occurred_at_crm": "2026-10-01 10:00:00",
        "event_type": "SMS",
        "direction": "OUTBOUND",
        "role": "sales",
        "call_result": "DELIVERED",
        "call_seconds": None,
        "content_state": "text_available",
        "text_part": "1/1",
        "text": "Hello, following up on your request.",
        "hidden_kind": None,
    }
    base.update(overrides)
    if base["direction"] == "INBOUND" and "role" not in overrides:
        base["role"] = "client"
    return base


def build_workbook(
    path: Path,
    leads: Iterable[Mapping[str, Any]],
    messages: Iterable[Mapping[str, Any]],
    *,
    control: Mapping[str, Any] | None = None,
    instruction: Mapping[str, Any] | None = None,
    auto_counters: bool = True,
    reference_labels: Mapping[str, str] | None = None,
    examples: Mapping[str, str] | None = None,
) -> Path:
    leads = [dict(row) for row in leads]
    messages = [dict(row) for row in messages]
    if auto_counters:
        for row in leads:
            own = [m for m in messages if m["case_id"] == row["case_id"]]
            ids = {m["message_id"] for m in own}
            row.setdefault("events_count", len(ids))
            if row["events_count"] is None:
                row["events_count"] = len(ids)
            if row["text_events"] is None:
                row["text_events"] = len(
                    {m["message_id"] for m in own if m["content_state"] == "text_available"}
                )
            if row["hidden_events"] is None:
                row["hidden_events"] = len(
                    {m["message_id"] for m in own if m["content_state"] == "withheld_hidden"}
                )
            if row["calls_without_text"] is None:
                row["calls_without_text"] = len(
                    {m["message_id"] for m in own if m["content_state"] == "call_without_text"}
                )

    wb = Workbook()
    ws_instr = wb.active
    assert ws_instr is not None
    ws_instr.title = "Інструкція"
    ws_instr.append(["Розділ", "Зміст"])
    instr = {"Знімок UTC": SNAPSHOT_UTC, "Часовий пояс": "America/Los_Angeles"}
    if instruction is not None:
        instr = dict(instruction)
    for key, value in instr.items():
        ws_instr.append([key, value])

    ws_leads = wb.create_sheet("Ліди")
    ws_leads.append(LEAD_HEADER)
    for row in leads:
        ws_leads.append([row.get(col) for col in LEAD_HEADER])

    ws_msgs = wb.create_sheet("Повідомлення")
    ws_msgs.append(MESSAGE_HEADER)
    for row in messages:
        ws_msgs.append([row.get(col) for col in MESSAGE_HEADER])

    ws_res = wb.create_sheet("Результат")
    ws_res.append(
        [
            "case_id",
            "decision",
            "reason",
            "evidence_message_ids",
            "draft_sms",
            "template_status",
            "reviewer_note",
        ]
    )

    if reference_labels is not None:
        ws_lab = wb.create_sheet("Розмітка")
        ws_lab.append(["case_id", "label", "reason", "evidence_message_ids", "question_for_owner"])
        for case_id, label in reference_labels.items():
            ws_lab.append([case_id, label, None, None, None])
    if examples is not None:
        ws_ex = wb.create_sheet("Приклади")
        ws_ex.append(["№", "case_id", "message_id", "Що там", "Мітка кейсу", "category", "Чому"])
        for index, (case_id, label) in enumerate(examples.items(), start=1):
            ws_ex.append([index, case_id, None, "synthetic", label, None, None])

    ws_ctrl = wb.create_sheet("Контроль")
    ws_ctrl.append(["Перевірка", "Результат"])
    ctrl: dict[str, Any] = {
        "cases": len(leads),
        "message_rows": len(messages),
        "with_quote": sum(1 for r in leads if r["has_quote"] == "yes"),
        "without_quote": sum(1 for r in leads if r["has_quote"] == "no"),
        "withheld_hidden": sum(1 for m in messages if m["content_state"] == "withheld_hidden"),
    }
    if control is not None:
        ctrl.update(control)
    for key, value in ctrl.items():
        ws_ctrl.append([key, value])

    wb.save(path)
    return path


@pytest.fixture
def workbook_path(tmp_path: Path) -> Path:
    return tmp_path / "export.xlsx"
