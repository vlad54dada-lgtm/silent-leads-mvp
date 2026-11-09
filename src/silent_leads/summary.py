"""Run summary: counts per decision, template status, findings and data issues."""

from __future__ import annotations

from collections import Counter
from typing import Any

from silent_leads.config import Config
from silent_leads.models import Decision, IssueLevel, TemplateStatus
from silent_leads.pipeline import RunResult

_DECISION_ORDER = (
    Decision.CONTACT_CANDIDATE,
    Decision.SKIP_NOT_DUE,
    Decision.DO_NOT_CONTACT,
    Decision.MANUAL_REVIEW,
)


def build_summary(config: Config, run: RunResult) -> dict[str, Any]:
    by_decision = Counter(r.decision for r in run.results)
    by_template = Counter(r.template_status for r in run.results)
    by_finding = Counter(f.kind for findings in run.findings.values() for f in findings)
    by_issue_code = Counter(i.code for i in run.issues)
    by_issue_level = Counter(i.level for i in run.issues)
    cases_with_errors = {i.case_id for i in run.issues if i.level == IssueLevel.ERROR and i.case_id}
    return {
        "snapshot_utc": run.snapshot.utc.isoformat(),
        "control_date": run.snapshot.control_date.isoformat(),
        "control_date_source": run.snapshot.source,
        "crm_timezone": config.crm_timezone,
        "first_touch_delay_days": config.first_touch_delay_days,
        "cases_total": len(run.results),
        "decisions": {d.value: by_decision.get(d, 0) for d in _DECISION_ORDER},
        "manual_review_cases": sorted(r.case_id for r in run.results if r.decision == Decision.MANUAL_REVIEW),
        "template_status": {t.value: by_template.get(t, 0) for t in TemplateStatus if by_template.get(t, 0)},
        "findings": {k.value: v for k, v in sorted(by_finding.items())},
        "data_issues": {
            "total": len(run.issues),
            "errors": by_issue_level.get(IssueLevel.ERROR, 0),
            "warnings": by_issue_level.get(IssueLevel.WARNING, 0),
            "cases_with_errors": sorted(c for c in cases_with_errors if c),
            "by_code": dict(sorted(by_issue_code.items())),
        },
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Зведення прогону",
        "",
        f"- Знімок вигрузки (UTC): `{summary['snapshot_utc']}`",
        f"- Контрольна дата (зона CRM {summary['crm_timezone']}): **{summary['control_date']}** "
        f"— джерело: {summary['control_date_source']}",
        f"- Строк першого касання: {summary['first_touch_delay_days']} календарних днів (тестове правило)",
        f"- Кейсів опрацьовано: **{summary['cases_total']}**",
        "",
        "## Рішення",
        "",
        "| Рішення | Кейсів |",
        "|---|---|",
    ]
    lines.extend(f"| `{k}` | {v} |" for k, v in summary["decisions"].items())
    lines += [
        "",
        f"На ручну перевірку: **{len(summary['manual_review_cases'])}** "
        f"({', '.join(summary['manual_review_cases']) or '—'})",
        "",
        "## Статус шаблонів",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in summary["template_status"].items())
    lines += ["", "## Знахідки в історії (кількість кейсів з ознакою)", ""]
    lines.extend(f"- `{k}`: {v}" for k, v in summary["findings"].items())
    if not summary["findings"]:
        lines.append("- немає")
    issues = summary["data_issues"]
    lines += [
        "",
        "## Проблеми даних",
        "",
        f"Усього {issues['total']} (помилок {issues['errors']}, попереджень {issues['warnings']}). "
        f"Кейсів з помилками: {len(issues['cases_with_errors'])}"
        + (f" ({', '.join(issues['cases_with_errors'])})" if issues["cases_with_errors"] else ""),
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in issues["by_code"].items())
    if not issues["by_code"]:
        lines.append("- не виявлено")
    lines.append("")
    return "\n".join(lines)
