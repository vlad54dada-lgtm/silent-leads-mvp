"""Quality of the v2 policy labels.

Two sources of truth, never committed to the repo:
- the reference labels inside the export: «Розмітка» (human labels) and «Приклади» (worked
  examples with the case label) — compared with the tool's ``labels.csv``;
- the synthetic set ``evals/policy_cases.yaml`` — hand-written dialogues with known labels, run
  through the judge directly (prompt regression, no export data involved).

The headline safety number is "missed refusals": the reference says do_not_contact but the tool
would let the case through (label contact_ok, or a final decision that is not a block/review).
The sprint target for it is zero.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from openpyxl import load_workbook
from pydantic import BaseModel, ConfigDict, Field

from silent_leads.config import AiReview, SheetNames
from silent_leads.models import PolicyLabel
from silent_leads.rules.contact_policy import AiReason, PolicyJudge, judge_items
from silent_leads.timeline import Speaker

LABELS = tuple(label.value for label in PolicyLabel)
PASS_THROUGH_DECISIONS = {"contact_candidate", "skip_not_due"}


@dataclass(frozen=True)
class PolicyRow:
    case_id: str
    expected: str
    predicted: str | None
    confidence: str = ""
    reason: str = ""
    final_decision: str = ""
    source: str = ""
    explanation: str = ""
    error: str = ""


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in LABELS else None


def _sheet_rows(path: Path, sheet: str) -> list[dict[str, Any]]:
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet not in wb.sheetnames:
            return []
        rows = list(wb[sheet].iter_rows(values_only=True))
    finally:
        wb.close()
    if not rows:
        return []
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    return [dict(zip(header, r, strict=False)) for r in rows[1:]]


def read_reference_labels(path: Path, sheets: SheetNames | None = None) -> dict[str, tuple[str, str]]:
    """case_id -> (label, source). «Розмітка» wins over «Приклади»; rows without a valid label are skipped."""
    sheets = sheets or SheetNames()
    reference: dict[str, tuple[str, str]] = {}
    for row in _sheet_rows(path, sheets.examples):
        case_id = str(row.get("case_id") or "").strip()
        label = _norm(row.get("Мітка кейсу"))
        if case_id and label:
            reference[case_id] = (label, sheets.examples)
    for row in _sheet_rows(path, sheets.labels):
        case_id = str(row.get("case_id") or "").strip()
        label = _norm(row.get("label"))
        if case_id and label:
            reference[case_id] = (label, sheets.labels)
    return reference


def read_tool_labels(path: Path) -> dict[str, dict[str, str]]:
    """labels.csv written by ``run`` with ``policy: v2``."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        header = fh.readline()
        fh.seek(0)
        delimiter = ";" if header.count(";") > header.count(",") else ","
        return {row["case_id"]: row for row in csv.DictReader(fh, delimiter=delimiter) if row.get("case_id")}


def join_rows(reference: dict[str, tuple[str, str]], tool: dict[str, dict[str, str]]) -> list[PolicyRow]:
    rows: list[PolicyRow] = []
    for case_id in sorted(reference):
        expected, source = reference[case_id]
        got = tool.get(case_id)
        rows.append(
            PolicyRow(
                case_id=case_id,
                expected=expected,
                predicted=_norm(got.get("label")) if got else None,
                confidence=(got or {}).get("confidence", ""),
                reason=(got or {}).get("reason", ""),
                final_decision=(got or {}).get("final_decision", ""),
                source=source,
                explanation=(got or {}).get("explanation", ""),
            )
        )
    return rows


def _ratio(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def evaluate_policy(rows: list[PolicyRow]) -> dict[str, Any]:
    compared = [r for r in rows if r.predicted is not None]
    correct = sum(1 for r in compared if r.predicted == r.expected)
    confusion = Counter((r.expected, r.predicted) for r in compared)
    missed = [
        r.case_id
        for r in compared
        if r.expected == PolicyLabel.DO_NOT_CONTACT
        and (r.predicted == PolicyLabel.CONTACT_OK or r.final_decision in PASS_THROUGH_DECISIONS)
    ]
    false_refusals = [
        r.case_id
        for r in compared
        if r.expected == PolicyLabel.CONTACT_OK and r.predicted == PolicyLabel.DO_NOT_CONTACT
    ]
    return {
        "reference": len(rows),
        "compared": len(compared),
        "missing_in_tool": sorted(r.case_id for r in rows if r.predicted is None),
        "accuracy": _ratio(correct, len(compared)),
        "confusion": {e: {p: confusion.get((e, p), 0) for p in LABELS} for e in LABELS},
        "missed_refusals": missed,
        "false_refusals": false_refusals,
        "mismatches": [
            {
                "case_id": r.case_id,
                "expected": r.expected,
                "predicted": r.predicted,
                "confidence": r.confidence,
                "reason": r.reason,
                "final_decision": r.final_decision,
                "source": r.source,
                "explanation": r.explanation,
                "error": r.error,
            }
            for r in compared
            if r.predicted != r.expected
        ],
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def render_policy_markdown(report: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Еталонних міток: {report['reference']}, порівняно: {report['compared']}",
        f"- Точність по мітках: **{_pct(report['accuracy'])}**",
        "- Пропущені відмови (еталон do_not_contact, інструмент пропускає): "
        f"**{len(report['missed_refusals'])}**"
        + (f" ({', '.join(report['missed_refusals'])})" if report["missed_refusals"] else ""),
        f"- Хибні відмови (еталон contact_ok, інструмент do_not_contact): {len(report['false_refusals'])}"
        + (f" ({', '.join(report['false_refusals'])})" if report["false_refusals"] else ""),
    ]
    if report["missing_in_tool"]:
        lines.append(f"- Немає у виході інструмента: {', '.join(report['missing_in_tool'])}")
    lines += ["", "## Матриця (рядки — еталон, стовпці — інструмент)", ""]
    lines.append("| еталон \\ інструмент | " + " | ".join(LABELS) + " |")
    lines.append("|---|" + "---|" * len(LABELS))
    for expected, row in report["confusion"].items():
        lines.append(f"| `{expected}` | " + " | ".join(str(row[p]) for p in LABELS) + " |")
    lines += ["", "## Розбіжності", ""]
    if not report["mismatches"]:
        lines.append("Немає.")
    for m in report["mismatches"]:
        lines.append(
            f"- **{m['case_id']}** ({m['source']}): еталон `{m['expected']}`, інструмент `{m['predicted']}` "
            f"({m['reason'] or '?'}, {m['confidence'] or '?'}; рішення {m['final_decision'] or '—'})"
            + (f". {m['explanation']}" if m["explanation"] else "")
            + (f" Помилка: {m['error']}" if m["error"] else "")
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- synthetic set


class PolicyTurn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    speaker: Speaker = Speaker.CLIENT
    text: str = ""
    event: str = ""  # e.g. "CALL INBOUND SUCCESS"; default derived from the speaker
    hidden: bool = False
    hidden_kind: str = ""
    no_transcript: bool = False


class PolicyCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    expected: PolicyLabel
    reason: AiReason | None = None
    note: str = ""
    dialogue: list[PolicyTurn] = Field(min_length=1)


class PolicySet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str = ""
    cases: list[PolicyCase] = Field(min_length=1)


def load_policy_set(path: Path) -> PolicySet:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    policy_set = PolicySet.model_validate(raw)
    ids = [c.id for c in policy_set.cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        msg = f"duplicate eval case ids: {duplicates}"
        raise ValueError(msg)
    return policy_set


def policy_case_items(case: PolicyCase) -> tuple[list[dict[str, Any]], set[str]]:
    """Synthetic ids ``<case>-<n>``; returns (items, ids holding client words)."""
    items: list[dict[str, Any]] = []
    client_ids: set[str] = set()
    for index, turn in enumerate(case.dialogue, start=1):
        message_id = f"{case.id}-{index}"
        default = "SMS INBOUND" if turn.speaker == Speaker.CLIENT else "SMS OUTBOUND"
        event = turn.event or default
        if turn.hidden:
            items.append(
                {
                    "message_id": message_id,
                    "event": event,
                    "hidden": True,
                    "hidden_kind": turn.hidden_kind or "unknown",
                }
            )
        elif turn.no_transcript:
            items.append({"message_id": message_id, "event": event, "no_transcript": True})
        else:
            items.append(
                {"message_id": message_id, "event": event, "speaker": turn.speaker.value, "text": turn.text}
            )
            if turn.speaker in (Speaker.CLIENT, Speaker.UNKNOWN):
                client_ids.add(message_id)
    return items, client_ids


def run_policy_eval(policy_set: PolicySet, judge: PolicyJudge, ai: AiReview) -> list[PolicyRow]:
    rows: list[PolicyRow] = []
    for case in sorted(policy_set.cases, key=lambda c: c.id):
        items, client_ids = policy_case_items(case)
        own_ids = {str(i["message_id"]) for i in items}
        outcome = judge_items(case.id, items, own_ids, client_ids, judge, ai)
        a = outcome.assessment
        rows.append(
            PolicyRow(
                case_id=case.id,
                expected=case.expected.value,
                predicted=a.label.value if a else None,
                confidence=a.confidence.value if a else "",
                reason=a.reason.value if a else "",
                source="synthetic" + (f" (очікувана причина {case.reason.value})" if case.reason else ""),
                explanation=a.explanation if a else "",
                error=outcome.error or "",
            )
        )
    return rows
