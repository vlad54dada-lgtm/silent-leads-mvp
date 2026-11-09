"""Write results.xlsx / results.csv / data_issues.csv / summary.md / summary.json.

Every file is fully rewritten on each run and contains nothing run-time dependent,
so two runs on the same input and config produce byte-identical outputs.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook

from silent_leads.models import CaseResult, DataIssue

RESULT_COLUMNS = (
    "case_id",
    "decision",
    "reason",
    "evidence_message_ids",
    "draft_sms",
    "template_status",
    "reviewer_note",
)
ISSUE_COLUMNS = ("level", "code", "case_id", "message_id", "detail")
EVIDENCE_SEPARATOR = ", "

_FIXED_STAMP = datetime(2000, 1, 1)  # keeps the xlsx core.xml byte-stable between runs
_ZIP_STAMP = (1980, 1, 1, 0, 0, 0)  # earliest DOS time; zip entries otherwise carry "now"
_CORE_MODIFIED_RE = re.compile(r"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)")


def _save_stable_xlsx(wb: Workbook, path: Path) -> None:
    """Save the workbook, then rewrite the zip with fixed entry timestamps.

    openpyxl stamps every zip member with the current time, which would make two
    identical runs differ byte-for-byte. The content itself is already deterministic.
    """
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as src, path.open("wb") as fh, zipfile.ZipFile(fh, "w") as dst:
        for entry in sorted(src.infolist(), key=lambda e: e.filename):
            data = src.read(entry.filename)
            if entry.filename == "docProps/core.xml":
                # openpyxl overwrites dcterms:modified with "now" on every save.
                data = _CORE_MODIFIED_RE.sub(
                    rf"\g<1>{_FIXED_STAMP.strftime('%Y-%m-%dT%H:%M:%SZ')}\g<2>", data.decode("utf-8")
                ).encode("utf-8")
            stable = zipfile.ZipInfo(entry.filename, date_time=_ZIP_STAMP)
            stable.compress_type = zipfile.ZIP_DEFLATED
            stable.external_attr = entry.external_attr
            dst.writestr(stable, data)


def _result_row(result: CaseResult) -> list[str]:
    return [
        result.case_id,
        result.decision.value,
        result.reason,
        EVIDENCE_SEPARATOR.join(result.evidence_message_ids),
        result.draft_sms,
        result.template_status.value,
        result.reviewer_note,
    ]


def _issue_row(issue: DataIssue) -> list[str]:
    return [issue.level.value, issue.code, issue.case_id or "", issue.message_id or "", issue.detail]


def write_outputs(
    output_dir: Path,
    results: tuple[CaseResult, ...],
    issues: tuple[DataIssue, ...],
    summary: dict[str, Any],
    summary_markdown: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_result_row(r) for r in sorted(results, key=lambda r: r.case_id)]
    issue_rows = [
        _issue_row(i)
        for i in sorted(issues, key=lambda i: (i.case_id or "", i.message_id or "", i.code, i.detail))
    ]

    xlsx = output_dir / "results.xlsx"
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Результат"
    ws.append(list(RESULT_COLUMNS))
    for row in rows:
        ws.append(row)
    ws_issues = wb.create_sheet("Проблеми даних")
    ws_issues.append(list(ISSUE_COLUMNS))
    for row in issue_rows:
        ws_issues.append(row)
    wb.properties.created = _FIXED_STAMP
    wb.properties.modified = _FIXED_STAMP
    _save_stable_xlsx(wb, xlsx)

    results_csv = output_dir / "results.csv"
    with results_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RESULT_COLUMNS)
        writer.writerows(rows)

    issues_csv = output_dir / "data_issues.csv"
    with issues_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ISSUE_COLUMNS)
        writer.writerows(issue_rows)

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md = output_dir / "summary.md"
    summary_md.write_text(summary_markdown, encoding="utf-8")

    return [xlsx, results_csv, issues_csv, summary_json, summary_md]
