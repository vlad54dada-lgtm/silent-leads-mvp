"""Cross-row checks: uniqueness, case_id links, per-lead counters, control checksums."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from silent_leads.io.reader import Export
from silent_leads.models import ContentState, DataIssue, IssueLevel


class ChecksumMismatchError(Exception):
    """The «Контроль» sheet disagrees with what was read — wrong or damaged file."""


@dataclass(frozen=True)
class ControlSummary:
    cases: int
    message_rows: int
    with_quote: int
    without_quote: int
    withheld_hidden: int


def summarize(export: Export) -> ControlSummary:
    return ControlSummary(
        cases=len(export.leads),
        message_rows=sum(m.parts_found for m in export.messages),
        with_quote=sum(1 for lead in export.leads if lead.has_quote is True),
        without_quote=sum(1 for lead in export.leads if lead.has_quote is False),
        withheld_hidden=sum(1 for m in export.messages if m.content_state == ContentState.WITHHELD_HIDDEN),
    )


def check_control_sheet(export: Export) -> list[str]:
    """Return human-readable mismatches between «Контроль» and the data actually read."""
    actual = summarize(export)
    mismatches: list[str] = []
    for key, value in (
        ("cases", actual.cases),
        ("message_rows", actual.message_rows),
        ("with_quote", actual.with_quote),
        ("without_quote", actual.without_quote),
        ("withheld_hidden", actual.withheld_hidden),
    ):
        declared = export.control.get(key)
        if declared is None:
            continue
        try:
            declared_int = int(float(declared))
        except ValueError:
            mismatches.append(f"{key}: у «Контроль» не число ('{declared}')")
            continue
        if declared_int != value:
            mismatches.append(f"{key}: «Контроль» = {declared_int}, прочитано = {value}")
    return mismatches


def validate_links(export: Export) -> list[DataIssue]:
    """Uniqueness of ids, orphan messages, leads without history, counter consistency."""
    issues: list[DataIssue] = []

    lead_ids = Counter(lead.case_id for lead in export.leads)
    for case_id, count in lead_ids.items():
        if count > 1:
            issues.append(
                DataIssue(
                    level=IssueLevel.ERROR,
                    code="lead_duplicate_case_id",
                    detail=f"case_id зустрічається {count} разів у «Ліди»",
                    case_id=case_id,
                )
            )

    message_ids = Counter(m.message_id for m in export.messages)
    for message_id, count in message_ids.items():
        if count > 1:
            issues.append(
                DataIssue(
                    level=IssueLevel.ERROR,
                    code="message_duplicate_id",
                    detail=f"message_id зустрічається {count} разів у різних кейсах",
                    message_id=message_id,
                )
            )

    known = set(lead_ids)
    per_case = Counter(m.case_id for m in export.messages)
    for m in export.messages:
        if m.case_id not in known:
            issues.append(
                DataIssue(
                    level=IssueLevel.ERROR,
                    code="message_orphan",
                    detail="повідомлення посилається на case_id, якого немає у «Ліди»",
                    case_id=m.case_id,
                    message_id=m.message_id,
                )
            )

    hidden = Counter(m.case_id for m in export.messages if m.content_state == ContentState.WITHHELD_HIDDEN)
    texted = Counter(m.case_id for m in export.messages if m.content_state == ContentState.TEXT_AVAILABLE)
    for lead in export.leads:
        if per_case[lead.case_id] == 0:
            issues.append(
                DataIssue(
                    level=IssueLevel.ERROR,
                    code="lead_without_history",
                    detail=(
                        "у «Повідомлення» немає жодної події для цього кейсу — контексту для рішення немає"
                    ),
                    case_id=lead.case_id,
                )
            )
        for name, declared, actual in (
            ("events_count", lead.events_count, per_case[lead.case_id]),
            ("hidden_events", lead.hidden_events, hidden[lead.case_id]),
            ("text_events", lead.text_events, texted[lead.case_id]),
        ):
            if declared is not None and declared != actual:
                issues.append(
                    DataIssue(
                        level=IssueLevel.ERROR,
                        code=f"lead_{name}_mismatch",
                        detail=(
                            f"{name} у «Ліди» = {declared}, у «Повідомлення» знайдено {actual} — "
                            "історія може бути неповною"
                        ),
                        case_id=lead.case_id,
                    )
                )
        if lead.status is not None and lead.status.strip().lower() != "junk silent":
            issues.append(
                DataIssue(
                    level=IssueLevel.WARNING,
                    code="lead_unexpected_status",
                    detail=f"status='{lead.status}', очікувався 'junk silent'",
                    case_id=lead.case_id,
                )
            )
        if lead.has_quote is True and lead.quote_at_crm is None:
            issues.append(
                DataIssue(
                    level=IssueLevel.WARNING,
                    code="lead_quote_without_date",
                    detail="has_quote=yes, але quote_at_crm порожнє",
                    case_id=lead.case_id,
                )
            )
    return issues
