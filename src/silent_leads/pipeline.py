"""Pure orchestration: export + config -> per-case results. No file I/O here."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from silent_leads.config import Config
from silent_leads.io.reader import Export
from silent_leads.models import CaseResult, DataIssue
from silent_leads.rules.decide import decide
from silent_leads.rules.due_date import check_due
from silent_leads.rules.history import Finding, analyse_history
from silent_leads.snapshot import Snapshot, resolve_snapshot
from silent_leads.templates import prepare_draft
from silent_leads.timeline import build_timelines
from silent_leads.validation import validate_links


@dataclass(frozen=True)
class RunResult:
    snapshot: Snapshot
    results: tuple[CaseResult, ...]  # sorted by case_id, exactly one per lead
    issues: tuple[DataIssue, ...]
    findings: dict[str, tuple[Finding, ...]]


def run_pipeline(config: Config, export: Export) -> RunResult:
    snapshot = resolve_snapshot(config, export)
    issues = [*export.issues, *validate_links(export)]
    by_case: dict[str, list[DataIssue]] = defaultdict(list)
    for issue in issues:
        if issue.case_id is not None:
            by_case[issue.case_id].append(issue)

    timelines = build_timelines(export.leads, export.messages)
    results: list[CaseResult] = []
    findings_by_case: dict[str, tuple[Finding, ...]] = {}
    for case_id, timeline in timelines.items():
        due = check_due(timeline.lead, snapshot.control_date, config.first_touch_delay_days)
        findings = analyse_history(timeline, config)
        verdict = decide(timeline, due, findings, by_case.get(case_id, []))
        draft, template_status = prepare_draft(timeline.lead, verdict.decision, config.templates)
        results.append(
            CaseResult(
                case_id=case_id,
                decision=verdict.decision,
                reason=verdict.reason,
                evidence_message_ids=verdict.evidence_message_ids,
                draft_sms=draft,
                template_status=template_status,
                reviewer_note=verdict.reviewer_note,
            )
        )
        findings_by_case[case_id] = tuple(findings)

    return RunResult(
        snapshot=snapshot,
        results=tuple(sorted(results, key=lambda r: r.case_id)),
        issues=tuple(issues),
        findings=findings_by_case,
    )
