"""Run summary: counts per decision, template status, findings, AI verdicts and data issues."""

from __future__ import annotations

from collections import Counter
from typing import Any

from silent_leads.config import Config
from silent_leads.models import Decision, IssueLevel, PolicyLabel, TemplateStatus
from silent_leads.pipeline import RunResult
from silent_leads.rules.ai_review import AiConfidence, AiLabel
from silent_leads.rules.contact_policy import REASON_HIDDEN_CONVERSATION
from silent_leads.rules.drafts import DraftStatus

_DECISION_ORDER = (
    Decision.CONTACT_CANDIDATE,
    Decision.SKIP_NOT_DUE,
    Decision.DO_NOT_CONTACT,
    Decision.MANUAL_REVIEW,
)


def _policy_summary(config: Config, run: RunResult) -> dict[str, Any]:
    verdicts = run.verdicts
    by_label = Counter(v.label for v in verdicts.values())
    by_reason = Counter(v.label_reason for v in verdicts.values())
    by_confidence = Counter(v.confidence or "-" for v in verdicts.values())
    outcomes = run.policy_outcomes
    drafts = run.drafts
    by_draft = Counter(d.status for d in drafts.values())
    return {
        "labels": {label.value: by_label.get(label, 0) for label in PolicyLabel},
        "reasons": dict(sorted(by_reason.items())),
        "confidence": dict(sorted(by_confidence.items())),
        "judged_by_ai": len(outcomes),
        "ai_errors": sum(1 for o in outcomes.values() if o.error is not None),
        "ai_error_kinds": dict(
            sorted(Counter(o.error.split(":")[0] for o in outcomes.values() if o.error is not None).items())
        ),
        "step4_cases": sorted(c for c, v in verdicts.items() if v.label_reason == REASON_HIDDEN_CONVERSATION),
        "questions_for_owner": sorted(c for c, v in verdicts.items() if v.question_for_owner),
        "new_contact_after_silent": sum(1 for v in verdicts.values() if v.new_contact_after_silent),
        "candidates": sorted(r.case_id for r in run.results if r.decision == Decision.CONTACT_CANDIDATE),
        "drafts": {
            "enabled": config.ai_review.enabled and config.drafts.enabled,
            "by_status": {s.value: by_draft.get(s, 0) for s in DraftStatus},
        },
    }


def build_summary(config: Config, run: RunResult) -> dict[str, Any]:
    by_decision = Counter(r.decision for r in run.results)
    by_template = Counter(r.template_status for r in run.results)
    by_finding = Counter(f.kind for findings in run.findings.values() for f in findings)
    by_issue_code = Counter(i.code for i in run.issues)
    by_issue_level = Counter(i.level for i in run.issues)
    cases_with_errors = {i.case_id for i in run.issues if i.level == IssueLevel.ERROR and i.case_id}
    outcomes = run.ai_outcomes
    by_label = Counter(o.label for o in outcomes.values() if o.label is not None)
    by_confidence = Counter(o.confidence for o in outcomes.values() if o.confidence is not None)
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
        "ai_review": {
            "enabled": config.ai_review.enabled,
            "model": config.ai_review.model if config.ai_review.enabled else None,
            "reviewed": len(outcomes),
            "by_label": {label.value: by_label.get(label, 0) for label in AiLabel},
            "by_confidence": {c.value: by_confidence.get(c, 0) for c in AiConfidence},
            "blocked": sum(1 for o in outcomes.values() if o.blocks),
            "cleared": sum(1 for o in outcomes.values() if o.clears),
            "needs_review": sum(1 for o in outcomes.values() if o.needs_review and o.error is None),
            "errors": sum(1 for o in outcomes.values() if o.error is not None),
            "blocked_cases": sorted(c for c, o in outcomes.items() if o.blocks),
            "review_cases": sorted(c for c, o in outcomes.items() if o.needs_review),
        },
        "policy": run.policy,
        "contact_policy": _policy_summary(config, run) if run.policy == "v2" else None,
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
    if summary.get("contact_policy") is not None:
        lines += _policy_markdown(summary)
    else:
        lines += _v1_markdown(summary)
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


def _v1_markdown(summary: dict[str, Any]) -> list[str]:
    lines = ["", "## Знахідки в історії (кількість кейсів з ознакою)", ""]
    lines.extend(f"- `{k}`: {v}" for k, v in summary["findings"].items())
    if not summary["findings"]:
        lines.append("- немає")
    ai = summary["ai_review"]
    lines += ["", "## AI-аналіз контексту", ""]
    if ai["enabled"]:
        lines += [
            f"Модель `{ai['model']}`. Кейсів проаналізовано: {ai['reviewed']}; "
            f"заблоковано (висока впевненість у відмові): {ai['blocked']}, "
            f"без зауважень: {ai['cleared']}, на ручну перевірку: {ai['needs_review']}, "
            f"помилок виклику: {ai['errors']}.",
            "",
            "| Мітка | Кейсів |",
            "|---|---|",
        ]
        lines.extend(f"| `{k}` | {v} |" for k, v in ai["by_label"].items())
        lines += ["", "Впевненість: " + ", ".join(f"{k} — {v}" for k, v in ai["by_confidence"].items())]
        if ai["blocked_cases"]:
            lines.append(f"Заблоковано: {', '.join(ai['blocked_cases'])}")
        if ai["review_cases"]:
            lines.append(f"На ручну: {', '.join(ai['review_cases'])}")
        lines += [
            "",
            "Розмітка для валідації людиною: `ai_labels.csv` (колонки human_label / human_comment).",
        ]
    else:
        lines.append(
            "Вимкнений (`ai_review.enabled: false`): фрази-підказки відправляють кейс на ручну перевірку."
        )
    return lines


def _policy_markdown(summary: dict[str, Any]) -> list[str]:
    cp = summary["contact_policy"]
    ai = summary["ai_review"]
    lines = ["", "## Політика «Кому пишемо» (v2)", ""]
    if ai["enabled"]:
        lines.append(
            f"Позицію клієнта (кроки 1-3) оцінювала модель `{ai['model']}`: кейсів {cp['judged_by_ai']}, "
            f"помилок виклику {cp['ai_errors']}. Крок 4, строк і новий контакт після silent рахує код."
        )
    else:
        lines.append(
            "AI вимкнено: код вирішує лише точний STOP і кейси без слів клієнта, решта — `need_human`."
        )
    if cp["ai_error_kinds"]:
        kinds = ", ".join(f"{k} {v}" for k, v in cp["ai_error_kinds"].items())
        lines.append(f"Помилки виклику AI: {kinds}. Ці кейси не оцінено і вони на ручній перевірці.")
    lines += ["", "| Мітка | Кейсів |", "|---|---|"]
    lines.extend(f"| `{k}` | {v} |" for k, v in cp["labels"].items())
    lines += ["", "Причини: " + ", ".join(f"`{k}` {v}" for k, v in cp["reasons"].items())]
    lines.append("Впевненість: " + ", ".join(f"{k} — {v}" for k, v in cp["confidence"].items()))
    lines.append(f"Крок 4 (прихована розмова після позиції клієнта): {len(cp['step4_cases'])}")
    lines.append(f"Новий контакт після silent: {cp['new_contact_after_silent']}")
    if cp["questions_for_owner"]:
        lines.append(f"Питання до замовника: {', '.join(cp['questions_for_owner'])}")
    lines.append(f"Кандидати: {', '.join(cp['candidates']) or '—'}")
    drafts = cp["drafts"]
    lines += ["", "### Чернетки SMS", ""]
    if drafts["enabled"]:
        lines.append(", ".join(f"`{k}` {v}" for k, v in drafts["by_status"].items()))
        lines.append("Чернетки — у `results.xlsx`, аркуш «Чернетки SMS». SMS надсилає сейлз після перевірки.")
    else:
        lines.append("Вимкнено (`drafts.enabled: false`).")
    lines += ["", "Мітки по кожному кейсу: аркуш «Мітки» / `labels.csv` (формат вкладки «Розмітка»)."]
    return lines
