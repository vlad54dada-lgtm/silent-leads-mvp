"""Read the CRM export workbook into raw rows and typed models.

Parsing is lenient on purpose: a bad value becomes ``None`` plus a DataIssue,
never a silently invented default and never a dropped row.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from silent_leads.config import SheetNames
from silent_leads.models import (
    ContentState,
    DataIssue,
    Direction,
    IssueLevel,
    Lead,
    Message,
    Role,
)

RawRow = dict[str, Any]

LEAD_COLUMNS = (
    "case_id",
    "team_alias",
    "sales_alias",
    "status",
    "created_at_crm",
    "silent_since_crm",
    "has_quote",
    "quote_at_crm",
    "last_inbound_at_crm",
    "last_outbound_at_crm",
    "events_count",
    "text_events",
    "hidden_events",
    "calls_without_text",
    "client_timezone",
)
MESSAGE_COLUMNS = (
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
)

_DATETIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")
_PART_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


@dataclass
class Export:
    """Everything read from the workbook, before validation."""

    leads: list[Lead]
    messages: list[Message]
    control: dict[str, str]
    instruction: dict[str, str]
    issues: list[DataIssue] = field(default_factory=list)


class ExportFormatError(Exception):
    """The workbook does not look like the expected export (missing sheet/columns)."""


# --------------------------------------------------------------------------- helpers


def _s(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_datetime(value: Any) -> datetime | None:
    """Naive datetime in CRM local time, or None when unparseable."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    text = str(value).strip()
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def _parse_yes_no(value: Any) -> bool | None:
    text = _s(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered == "yes":
        return True
    if lowered == "no":
        return False
    return None


def _read_sheet_rows(path: Path, sheet: str, required: tuple[str, ...]) -> list[RawRow]:
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet not in wb.sheetnames:
            msg = f"sheet '{sheet}' not found; available: {wb.sheetnames}"
            raise ExportFormatError(msg)
        ws = wb[sheet]
        rows_iter = ws.iter_rows(values_only=True)
        header_raw = next(rows_iter, None)
        if header_raw is None:
            msg = f"sheet '{sheet}' is empty"
            raise ExportFormatError(msg)
        header = [_s(h) or "" for h in header_raw]
        missing = [c for c in required if c not in header]
        if missing:
            msg = f"sheet '{sheet}' lacks columns {missing}"
            raise ExportFormatError(msg)
        rows: list[RawRow] = []
        for raw in rows_iter:
            if all(v is None or str(v).strip() == "" for v in raw):
                continue
            rows.append(dict(zip(header, raw, strict=False)))
        return rows
    finally:
        wb.close()


def _read_key_value_sheet(path: Path, sheet: str) -> dict[str, str]:
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet not in wb.sheetnames:
            return {}
        result: dict[str, str] = {}
        for raw in wb[sheet].iter_rows(values_only=True):
            key = _s(raw[0]) if raw else None
            if key is None:
                continue
            value = _s(raw[1]) if len(raw) > 1 else None
            result[key] = value or ""
        return result
    finally:
        wb.close()


# --------------------------------------------------------------------------- leads


def _lead_from_row(row: RawRow, row_no: int, issues: list[DataIssue]) -> Lead | None:
    case_id = _s(row.get("case_id"))
    if case_id is None:
        issues.append(
            DataIssue(
                level=IssueLevel.ERROR,
                code="lead_missing_case_id",
                detail=f"Ліди, рядок {row_no}: порожній case_id — рядок не можна прив'язати до кейсу",
            )
        )
        return None

    def dt(col: str, *, required: bool) -> datetime | None:
        value = row.get(col)
        parsed = _parse_datetime(value)
        if parsed is None and value not in (None, ""):
            issues.append(
                DataIssue(
                    level=IssueLevel.ERROR if required else IssueLevel.WARNING,
                    code=f"lead_bad_{col}",
                    detail=f"{col}: значення '{value}' не є датою формату YYYY-MM-DD HH:MM:SS",
                    case_id=case_id,
                )
            )
        elif parsed is None and required:
            issues.append(
                DataIssue(
                    level=IssueLevel.ERROR,
                    code=f"lead_missing_{col}",
                    detail=f"{col}: відсутнє значення",
                    case_id=case_id,
                )
            )
        return parsed

    def num(col: str) -> int | None:
        value = row.get(col)
        parsed = _parse_int(value)
        if parsed is None and value not in (None, ""):
            issues.append(
                DataIssue(
                    level=IssueLevel.WARNING,
                    code=f"lead_bad_{col}",
                    detail=f"{col}: '{value}' не є цілим числом",
                    case_id=case_id,
                )
            )
        return parsed

    has_quote = _parse_yes_no(row.get("has_quote"))
    if has_quote is None:
        issues.append(
            DataIssue(
                level=IssueLevel.ERROR,
                code="lead_bad_has_quote",
                detail=(
                    f"has_quote: '{row.get('has_quote')}' — очікується yes/no; гілку шаблону обрати неможливо"
                ),
                case_id=case_id,
            )
        )

    return Lead(
        case_id=case_id,
        team_alias=_s(row.get("team_alias")),
        sales_alias=_s(row.get("sales_alias")),
        status=_s(row.get("status")),
        created_at_crm=dt("created_at_crm", required=False),
        silent_since_crm=dt("silent_since_crm", required=True),
        has_quote=has_quote,
        quote_at_crm=dt("quote_at_crm", required=False),
        last_inbound_at_crm=dt("last_inbound_at_crm", required=False),
        last_outbound_at_crm=dt("last_outbound_at_crm", required=False),
        events_count=num("events_count"),
        text_events=num("text_events"),
        hidden_events=num("hidden_events"),
        calls_without_text=num("calls_without_text"),
        client_timezone=_s(row.get("client_timezone")),
    )


# --------------------------------------------------------------------------- messages


def _enum_or_issue[E: (Direction, Role, ContentState)](
    enum_cls: type[E], value: Any, col: str, case_id: str, message_id: str, issues: list[DataIssue]
) -> E | None:
    text = _s(value)
    if text is None:
        issues.append(
            DataIssue(
                level=IssueLevel.WARNING,
                code=f"message_missing_{col}",
                detail=f"{col}: відсутнє значення",
                case_id=case_id,
                message_id=message_id,
            )
        )
        return None
    try:
        return enum_cls(text)
    except ValueError:
        issues.append(
            DataIssue(
                level=IssueLevel.WARNING,
                code=f"message_bad_{col}",
                detail=f"{col}: невідоме значення '{text}'",
                case_id=case_id,
                message_id=message_id,
            )
        )
        return None


def _merge_parts(rows: list[tuple[int, RawRow]], issues: list[DataIssue]) -> Message | None:
    """Rows sharing one message_id are parts of one event; glue the text in part order."""
    first_no, first = rows[0]
    case_id = _s(first.get("case_id"))
    message_id = _s(first.get("message_id"))
    if case_id is None or message_id is None:
        issues.append(
            DataIssue(
                level=IssueLevel.ERROR,
                code="message_missing_ids",
                detail=f"Повідомлення, рядок {first_no}: порожній case_id або message_id",
                case_id=case_id,
                message_id=message_id,
            )
        )
        return None

    parts: list[tuple[int, str]] = []
    expected: int | None = None
    for row_no, row in rows:
        part_raw = _s(row.get("text_part")) or "1/1"
        match = _PART_RE.match(part_raw)
        if match is None:
            issues.append(
                DataIssue(
                    level=IssueLevel.WARNING,
                    code="message_bad_text_part",
                    detail=(
                        f"text_part '{part_raw}' (рядок {row_no}) не має вигляду k/n; частину вважаємо 1/1"
                    ),
                    case_id=case_id,
                    message_id=message_id,
                )
            )
            index, total = 1, 1
        else:
            index, total = int(match.group(1)), int(match.group(2))
        if expected is None:
            expected = total
        elif total != expected:
            issues.append(
                DataIssue(
                    level=IssueLevel.WARNING,
                    code="message_inconsistent_parts",
                    detail=f"частини одного message_id заявляють різну кількість ({expected} і {total})",
                    case_id=case_id,
                    message_id=message_id,
                )
            )
        parts.append((index, _s(row.get("text")) or ""))

    parts.sort(key=lambda p: p[0])
    expected = expected or 1
    if len(parts) != expected or [p[0] for p in parts] != list(range(1, expected + 1)):
        # A client message with a missing part may hide the refusal: error (-> manual_review).
        inbound = (_s(first.get("direction")) or "").upper() == Direction.INBOUND.value
        issues.append(
            DataIssue(
                level=IssueLevel.ERROR if inbound else IssueLevel.WARNING,
                code="message_incomplete_parts",
                detail=f"знайдено {len(parts)} частин з {expected}; текст може бути неповним",
                case_id=case_id,
                message_id=message_id,
            )
        )
    text = "\n".join(p[1] for p in parts if p[1]) or None

    occurred = _parse_datetime(first.get("occurred_at_crm"))
    if occurred is None:
        issues.append(
            DataIssue(
                level=IssueLevel.ERROR,
                code="message_bad_occurred_at",
                detail=(
                    f"occurred_at_crm: '{first.get('occurred_at_crm')}' — не дата; "
                    "хронологію кейсу відновити неможливо"
                ),
                case_id=case_id,
                message_id=message_id,
            )
        )

    direction = _enum_or_issue(Direction, first.get("direction"), "direction", case_id, message_id, issues)
    role = _enum_or_issue(Role, first.get("role"), "role", case_id, message_id, issues)
    content_state = _enum_or_issue(
        ContentState, first.get("content_state"), "content_state", case_id, message_id, issues
    )

    if content_state == ContentState.TEXT_AVAILABLE and not text:
        issues.append(
            DataIssue(
                level=IssueLevel.WARNING,
                code="message_text_missing",
                detail="content_state=text_available, але текст порожній",
                case_id=case_id,
                message_id=message_id,
            )
        )
    if content_state not in (None, ContentState.TEXT_AVAILABLE) and text:
        issues.append(
            DataIssue(
                level=IssueLevel.WARNING,
                code="message_unexpected_text",
                detail=f"content_state={content_state}, але текст присутній",
                case_id=case_id,
                message_id=message_id,
            )
        )
    expected_role: Role | None = None
    if direction is not None:
        expected_role = {Direction.INBOUND: Role.CLIENT, Direction.OUTBOUND: Role.SALES}[direction]
    if expected_role is not None and role is not None and role != expected_role:
        issues.append(
            DataIssue(
                level=IssueLevel.WARNING,
                code="message_role_mismatch",
                detail=f"direction={direction} і role={role} суперечать одне одному",
                case_id=case_id,
                message_id=message_id,
            )
        )

    return Message(
        case_id=case_id,
        message_id=message_id,
        occurred_at_crm=occurred,
        event_type=_s(first.get("event_type")),
        direction=direction,
        role=role,
        call_result=_s(first.get("call_result")),
        call_seconds=_parse_int(first.get("call_seconds")),
        content_state=content_state,
        text=text,
        hidden_kind=_s(first.get("hidden_kind")),
        parts_expected=expected,
        parts_found=len(parts),
    )


# --------------------------------------------------------------------------- public


def read_export(path: Path, sheets: SheetNames | None = None) -> Export:
    """Load the workbook. Raises ExportFormatError if sheets/columns are missing."""
    sheets = sheets or SheetNames()
    issues: list[DataIssue] = []

    lead_rows = _read_sheet_rows(path, sheets.leads, LEAD_COLUMNS)
    leads: list[Lead] = []
    for row_no, row in enumerate(lead_rows, start=2):
        lead = _lead_from_row(row, row_no, issues)
        if lead is not None:
            leads.append(lead)

    message_rows = _read_sheet_rows(path, sheets.messages, MESSAGE_COLUMNS)
    grouped: dict[tuple[str | None, str | None], list[tuple[int, RawRow]]] = defaultdict(list)
    for row_no, row in enumerate(message_rows, start=2):
        grouped[(_s(row.get("case_id")), _s(row.get("message_id")))].append((row_no, row))
    messages: list[Message] = []
    for rows in grouped.values():
        message = _merge_parts(rows, issues)
        if message is not None:
            messages.append(message)

    return Export(
        leads=leads,
        messages=messages,
        control=_read_key_value_sheet(path, sheets.control),
        instruction=_read_key_value_sheet(path, sheets.instruction),
        issues=issues,
    )
