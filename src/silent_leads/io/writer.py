"""Write results.xlsx / results.csv / data_issues.csv / ai_labels.csv / summary.md / summary.json.

Every file is fully rewritten on each run and contains nothing run-time dependent,
so two runs on the same input and config produce byte-identical outputs
(with the AI review on, identical up to the model's answers).
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook

from silent_leads.models import CaseResult, DataIssue
from silent_leads.rules.ai_review import AiOutcome

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
# One row per case for the validation round: the reviewer fills human_label / human_comment,
# then `silent_leads.cli evaluate` compares the two columns.
AI_LABEL_COLUMNS = (
    "case_id",
    "final_decision",
    "ai_label",
    "ai_confidence",
    "ai_evidence_message_ids",
    "ai_explanation",
    "ai_error",
    "human_label",
    "human_comment",
)
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


def _ai_label_row(result: CaseResult, outcome: AiOutcome | None) -> list[str]:
    if outcome is None:
        return [result.case_id, result.decision.value, "", "", "", "", "", "", ""]
    return [
        result.case_id,
        result.decision.value,
        outcome.label.value if outcome.label else "",
        outcome.confidence.value if outcome.confidence else "",
        EVIDENCE_SEPARATOR.join(outcome.evidence_message_ids),
        outcome.explanation,
        outcome.error or "",
        "",
        "",
    ]


def _write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    results: tuple[CaseResult, ...],
    issues: tuple[DataIssue, ...],
    summary: dict[str, Any],
    summary_markdown: str,
    ai_outcomes: Mapping[str, AiOutcome] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda r: r.case_id)
    rows = [_result_row(r) for r in ordered]
    issue_rows = [
        _issue_row(i)
        for i in sorted(issues, key=lambda i: (i.case_id or "", i.message_id or "", i.code, i.detail))
    ]
    outcomes = ai_outcomes or {}
    ai_rows = [_ai_label_row(r, outcomes.get(r.case_id)) for r in ordered]

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
    _write_csv(results_csv, RESULT_COLUMNS, rows)
    issues_csv = output_dir / "data_issues.csv"
    _write_csv(issues_csv, ISSUE_COLUMNS, issue_rows)
    ai_csv = output_dir / "ai_labels.csv"
    _write_csv(ai_csv, AI_LABEL_COLUMNS, ai_rows)

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md = output_dir / "summary.md"
    summary_md.write_text(summary_markdown, encoding="utf-8")

    return [xlsx, results_csv, issues_csv, ai_csv, summary_json, summary_md]
