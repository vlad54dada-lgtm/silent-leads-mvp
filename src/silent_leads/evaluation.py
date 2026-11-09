"""Compare AI labels with a reference (human labels or a synthetic expected set).

Pure functions: rows in, metrics dict out. Used by both validation flows:
- ``evaluate``: ``ai_labels.csv`` after a reviewer filled ``human_label``;
- ``ai-eval``: synthetic dialogues from ``evals/*.yaml`` with an ``expected`` label.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from silent_leads.rules.ai_review import AiLabel

LABELS = tuple(label.value for label in AiLabel)
# Labels that mean "do not contact" when compared at decision level.
REFUSAL_LABELS = {AiLabel.OPT_OUT.value, AiLabel.NEGATIVE.value}

_ALIASES = {
    "optout": "opt_out",
    "opt-out": "opt_out",
    "відписка": "opt_out",
    "негатив": "negative",
    "неоднозначно": "unclear",
    "немає": "none",
    "ok": "none",
    "clear": "none",
}


@dataclass(frozen=True)
class EvalRow:
    case_id: str
    expected: str | None  # normalized reference label, None when not labelled
    predicted: str | None  # normalized AI label, None when the AI did not answer
    confidence: str
    explanation: str
    error: str = ""
    raw_expected: str = ""  # what the human actually wrote, for the report


def normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip().lower().replace(" ", "")
    if not key:
        return None
    key = _ALIASES.get(key, key)
    return key  # unknown labels are kept as written and reported as invalid


def read_labels_csv(path: Path) -> list[EvalRow]:
    """Rows of ``ai_labels.csv`` (as written by ``run``, with human columns filled in)."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[EvalRow] = []
        for record in reader:
            rows.append(
                EvalRow(
                    case_id=record.get("case_id", "").strip(),
                    expected=normalize_label(record.get("human_label")),
                    predicted=normalize_label(record.get("ai_label")) or None,
                    confidence=(record.get("ai_confidence") or "").strip(),
                    explanation=(record.get("ai_explanation") or "").strip(),
                    error=(record.get("ai_error") or "").strip(),
                    raw_expected=(record.get("human_label") or "").strip(),
                )
            )
    return rows


def _as_refusal(label: str | None) -> bool | None:
    if label is None or label == AiLabel.UNCLEAR.value:
        return None
    return label in REFUSAL_LABELS


def evaluate(rows: list[EvalRow]) -> dict[str, Any]:
    """Accuracy of AI labels against the reference, plus the breakdown a reviewer needs."""
    invalid = sorted(r.case_id for r in rows if r.expected is not None and r.expected not in LABELS)
    labelled = [r for r in rows if r.expected in LABELS]
    comparable = [r for r in labelled if r.predicted in LABELS]
    exact = [r for r in comparable if r.predicted == r.expected]

    # Decision-level agreement: refusal (opt_out/negative) vs none; unclear on either side is "undecided".
    decided = [
        r for r in comparable if _as_refusal(r.expected) is not None and _as_refusal(r.predicted) is not None
    ]
    decision_agree = [r for r in decided if _as_refusal(r.expected) == _as_refusal(r.predicted)]
    missed_refusals = [
        r for r in decided if _as_refusal(r.expected) is True and _as_refusal(r.predicted) is False
    ]
    false_refusals = [
        r for r in decided if _as_refusal(r.expected) is False and _as_refusal(r.predicted) is True
    ]

    confusion: dict[str, dict[str, int]] = {e: dict.fromkeys(LABELS, 0) for e in LABELS}
    for r in comparable:
        assert r.expected is not None and r.predicted is not None
        confusion[r.expected][r.predicted] += 1

    by_confidence: dict[str, dict[str, Any]] = {}
    for level, group in Counter(r.confidence for r in comparable).items():
        hits = sum(1 for r in comparable if r.confidence == level and r.predicted == r.expected)
        by_confidence[level or "?"] = {"total": group, "correct": hits, "accuracy": _ratio(hits, group)}

    mismatches = [
        {
            "case_id": r.case_id,
            "expected": r.expected,
            "predicted": r.predicted,
            "confidence": r.confidence,
            "explanation": r.explanation,
        }
        for r in comparable
        if r.predicted != r.expected
    ]
    return {
        "rows_total": len(rows),
        "labelled": len(labelled),
        "comparable": len(comparable),
        "not_predicted": sorted(r.case_id for r in labelled if r.predicted not in LABELS),
        "invalid_labels": invalid,
        "exact_accuracy": _ratio(len(exact), len(comparable)),
        "exact_correct": len(exact),
        "decision_accuracy": _ratio(len(decision_agree), len(decided)),
        "decision_compared": len(decided),
        "missed_refusals": sorted(r.case_id for r in missed_refusals),
        "false_refusals": sorted(r.case_id for r in false_refusals),
        "confusion": confusion,
        "by_confidence": dict(sorted(by_confidence.items())),
        "mismatches": sorted(mismatches, key=lambda m: m["case_id"]),
    }


def _ratio(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def render_evaluation_markdown(report: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Рядків: {report['rows_total']}, з еталонною міткою: {report['labelled']}, "
        f"порівняно з відповіддю AI: {report['comparable']}",
        f"- **Точність по мітках (4 класи): {_pct(report['exact_accuracy'])}** "
        f"({report['exact_correct']} з {report['comparable']})",
        f"- **Точність на рівні рішення (відмова / не відмова): {_pct(report['decision_accuracy'])}** "
        f"(порівняно {report['decision_compared']}; `unclear` з будь-якого боку не рахується)",
        f"- Пропущені відмови (людина: відмова, AI: немає): {len(report['missed_refusals'])}"
        + (f" — {', '.join(report['missed_refusals'])}" if report["missed_refusals"] else ""),
        f"- Хибні відмови (людина: немає, AI: відмова): {len(report['false_refusals'])}"
        + (f" — {', '.join(report['false_refusals'])}" if report["false_refusals"] else ""),
    ]
    if report["not_predicted"]:
        lines.append(f"- Без відповіді AI (помилка виклику тощо): {', '.join(report['not_predicted'])}")
    if report["invalid_labels"]:
        lines.append(
            f"- Нерозпізнані мітки людини (допустимі: {', '.join(LABELS)}): "
            f"{', '.join(report['invalid_labels'])}"
        )
    lines += [
        "",
        "## Матриця (рядок — еталон, стовпець — AI)",
        "",
        "| еталон \\ AI | " + " | ".join(LABELS) + " |",
    ]
    lines.append("|---|" + "---|" * len(LABELS))
    for expected in LABELS:
        row = report["confusion"][expected]
        lines.append(f"| `{expected}` | " + " | ".join(str(row[p]) for p in LABELS) + " |")
    lines += ["", "## Точність за впевненістю AI", ""]
    for level, stats in report["by_confidence"].items():
        lines.append(f"- `{level}`: {_pct(stats['accuracy'])} ({stats['correct']} з {stats['total']})")
    if not report["by_confidence"]:
        lines.append("- немає даних")
    lines += ["", "## Розбір розбіжностей", ""]
    if not report["mismatches"]:
        lines.append("Розбіжностей немає.")
    for m in report["mismatches"]:
        lines.append(
            f"- **{m['case_id']}**: еталон `{m['expected']}`, AI `{m['predicted']}` "
            f"(впевненість {m['confidence'] or '?'}). AI: {m['explanation'] or '—'}"
        )
    lines.append("")
    return "\n".join(lines)
