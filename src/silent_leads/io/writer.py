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
from silent_leads.rules.contact_policy import PolicyOutcome, PolicyVerdict
from silent_leads.rules.drafts import DraftOutcome

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
# v2: one row per case in the shape of the «Розмітка» sheet, so a human label can be compared directly.
LABEL_COLUMNS = (
    "case_id",
    "label",
    "reason",
    "evidence_message_ids",
    "question_for_owner",
    "new_contact_after_silent",
    "confidence",
    "last_client_position",
    "final_decision",
    "explanation",
)
# v2: AI draft SMS for contact candidates — suggestions for the sales rep, not approved templates.
DRAFT_COLUMNS = (
    "case_id",
    "client_position",
    "last_client_position",
    "draft_status",
    "ai_draft_sms",
    "problems",
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


def _label_row(result: CaseResult, verdict: PolicyVerdict, outcome: PolicyOutcome | None) -> list[str]:
    explanation = ""
    if outcome is not None and outcome.assessment is not None:
        explanation = outcome.assessment.explanation.strip()
    return [
        result.case_id,
        verdict.label.value,
        verdict.label_reason,
        EVIDENCE_SEPARATOR.join(verdict.evidence_message_ids),
        verdict.question_for_owner,
        "yes" if verdict.new_contact_after_silent else "no",
        verdict.confidence,
        verdict.position_id or "",
        result.decision.value,
        explanation,
    ]


def _draft_row(verdict: PolicyVerdict, draft: DraftOutcome) -> list[str]:
    return [
        draft.case_id,
        verdict.label_reason,
        verdict.position_id or "",
        draft.status.value,
        draft.text,
        "; ".join(draft.problems),
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
    verdicts: Mapping[str, PolicyVerdict] | None = None,
    policy_outcomes: Mapping[str, PolicyOutcome] | None = None,
    drafts: Mapping[str, DraftOutcome] | None = None,
) -> list[Path]:
    """v1 writes ai_labels.csv; v2 (``verdicts`` given) writes labels.csv / drafts.csv and two more sheets."""
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
    v2 = bool(verdicts)
    label_rows: list[list[str]] = []
    draft_rows: list[list[str]] = []
    if verdicts:
        judged = policy_outcomes or {}
        label_rows = [_label_row(r, verdicts[r.case_id], judged.get(r.case_id)) for r in ordered]
        made = drafts or {}
        draft_rows = [_draft_row(verdicts[cid], made[cid]) for cid in sorted(made)]
        ws_labels = wb.create_sheet("Мітки")
        ws_labels.append(list(LABEL_COLUMNS))
        for row in label_rows:
            ws_labels.append(row)
        ws_drafts = wb.create_sheet("Чернетки SMS")
        ws_drafts.append(list(DRAFT_COLUMNS))
        for row in draft_rows:
            ws_drafts.append(row)
    wb.properties.created = _FIXED_STAMP
    wb.properties.modified = _FIXED_STAMP
    _save_stable_xlsx(wb, xlsx)

    results_csv = output_dir / "results.csv"
    _write_csv(results_csv, RESULT_COLUMNS, rows)
    issues_csv = output_dir / "data_issues.csv"
    _write_csv(issues_csv, ISSUE_COLUMNS, issue_rows)
    extra: list[Path] = []
    if v2:
        labels_csv = output_dir / "labels.csv"
        _write_csv(labels_csv, LABEL_COLUMNS, label_rows)
        drafts_csv = output_dir / "drafts.csv"
        _write_csv(drafts_csv, DRAFT_COLUMNS, draft_rows)
        extra = [labels_csv, drafts_csv]
    else:
        ai_csv = output_dir / "ai_labels.csv"
        _write_csv(ai_csv, AI_LABEL_COLUMNS, ai_rows)
        extra = [ai_csv]

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md = output_dir / "summary.md"
    summary_md.write_text(summary_markdown, encoding="utf-8")

    return [xlsx, results_csv, issues_csv, *extra, summary_json, summary_md]
