"""Pure orchestration: export + config -> per-case results. No file I/O here."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from silent_leads.config import Config
from silent_leads.io.reader import Export
from silent_leads.models import CaseResult, DataIssue
from silent_leads.rules.ai_review import AiOutcome, Reviewer, review_case
from silent_leads.rules.decide import decide
from silent_leads.rules.due_date import check_due
from silent_leads.rules.history import Finding, FindingKind, analyse_history, has_client_text
from silent_leads.snapshot import Snapshot, resolve_snapshot
from silent_leads.templates import prepare_draft
from silent_leads.timeline import CaseTimeline, build_timelines
from silent_leads.validation import validate_links


@dataclass(frozen=True)
class RunResult:
    snapshot: Snapshot
    results: tuple[CaseResult, ...]  # sorted by case_id, exactly one per lead
    issues: tuple[DataIssue, ...]
    findings: dict[str, tuple[Finding, ...]]
    ai_outcomes: dict[str, AiOutcome] = field(default_factory=dict)  # only reviewed cases


def needs_ai_review(timeline: CaseTimeline, findings: list[Finding]) -> bool:
    """The model judges context: skip cases with nothing from the client or an exact STOP."""
    if any(f.kind == FindingKind.EXACT_OPT_OUT for f in findings):
        return False
    return has_client_text(timeline)


def run_pipeline(config: Config, export: Export, reviewer: Reviewer | None = None) -> RunResult:
    """``reviewer`` is used only when ``config.ai_review.enabled``; None builds the Claude client."""
    snapshot = resolve_snapshot(config, export)
    issues = [*export.issues, *validate_links(export)]
    by_case: dict[str, list[DataIssue]] = defaultdict(list)
    for issue in issues:
        if issue.case_id is not None:
            by_case[issue.case_id].append(issue)

    if config.ai_review.enabled and reviewer is None:
        from silent_leads.rules.ai_review import ClaudeReviewer

        reviewer = ClaudeReviewer(config.ai_review)

    timelines = build_timelines(export.leads, export.messages)
    results: list[CaseResult] = []
    findings_by_case: dict[str, tuple[Finding, ...]] = {}
    ai_outcomes: dict[str, AiOutcome] = {}
    for case_id, timeline in timelines.items():
        due = check_due(timeline.lead, snapshot.control_date, config.first_touch_delay_days)
        findings = analyse_history(timeline, config)
        ai: AiOutcome | None = None
        if config.ai_review.enabled and reviewer is not None and needs_ai_review(timeline, findings):
            ai = review_case(timeline, reviewer, config.ai_review)
            ai_outcomes[case_id] = ai
        verdict = decide(timeline, due, findings, by_case.get(case_id, []), ai)
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
        ai_outcomes=ai_outcomes,
    )
