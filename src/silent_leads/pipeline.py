"""Pure orchestration: export + config -> per-case results. No file I/O here."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from silent_leads.config import Config
from silent_leads.io.reader import Export
from silent_leads.models import CaseResult, DataIssue, Decision
from silent_leads.rules.ai_review import AiOutcome, Reviewer, review_case
from silent_leads.rules.contact_policy import (
    PolicyJudge,
    PolicyOutcome,
    PolicyVerdict,
    decide_policy,
    judge_case,
    needs_judge,
)
from silent_leads.rules.decide import decide
from silent_leads.rules.drafts import DraftOutcome, DraftWriter, write_draft
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
    policy: str = "v1"
    verdicts: dict[str, PolicyVerdict] = field(default_factory=dict)  # v2: every case
    policy_outcomes: dict[str, PolicyOutcome] = field(default_factory=dict)  # v2: judged cases
    drafts: dict[str, DraftOutcome] = field(default_factory=dict)  # v2: candidates with a draft


def needs_ai_review(timeline: CaseTimeline, findings: list[Finding]) -> bool:
    """The model judges context: skip cases with nothing from the client or an exact STOP."""
    if any(f.kind == FindingKind.EXACT_OPT_OUT for f in findings):
        return False
    return has_client_text(timeline)


def _issues_by_case(issues: list[DataIssue]) -> dict[str, list[DataIssue]]:
    by_case: dict[str, list[DataIssue]] = defaultdict(list)
    for issue in issues:
        if issue.case_id is not None:
            by_case[issue.case_id].append(issue)
    return by_case


def run_pipeline(
    config: Config,
    export: Export,
    reviewer: Reviewer | None = None,
    judge: PolicyJudge | None = None,
    draft_writer: DraftWriter | None = None,
) -> RunResult:
    """Fakes may be injected for the AI parts; None builds the Claude clients when AI is enabled.

    ``config.policy`` picks the rules: v1 (first sprint) or v2 («Кому пишемо»).
    """
    if config.policy == "v2":
        return run_policy_pipeline(config, export, judge, draft_writer)
    snapshot = resolve_snapshot(config, export)
    issues = [*export.issues, *validate_links(export)]
    by_case = _issues_by_case(issues)

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


def run_policy_pipeline(
    config: Config,
    export: Export,
    judge: PolicyJudge | None = None,
    draft_writer: DraftWriter | None = None,
) -> RunResult:
    """v2: AI judges the client's position (steps 1-3), code applies step 4 and the final rules.

    Drafts are written only for ``contact_candidate`` and only with ``drafts.enabled``.
    """
    snapshot = resolve_snapshot(config, export)
    issues = [*export.issues, *validate_links(export)]
    by_case = _issues_by_case(issues)
    ai_on = config.ai_review.enabled
    if ai_on and judge is None:
        from silent_leads.rules.contact_policy import ClaudePolicyJudge

        judge = ClaudePolicyJudge(config.ai_review)
    drafts_on = ai_on and config.drafts.enabled
    if drafts_on and draft_writer is None:
        from silent_leads.rules.drafts import ClaudeDraftWriter

        draft_writer = ClaudeDraftWriter(config.ai_review, config.drafts)

    timelines = build_timelines(export.leads, export.messages)
    results: list[CaseResult] = []
    verdicts: dict[str, PolicyVerdict] = {}
    outcomes: dict[str, PolicyOutcome] = {}
    drafts: dict[str, DraftOutcome] = {}
    for case_id, timeline in timelines.items():
        due = check_due(timeline.lead, snapshot.control_date, config.first_touch_delay_days)
        outcome: PolicyOutcome | None = None
        if ai_on and judge is not None and needs_judge(timeline, config):
            outcome = judge_case(timeline, judge, config.ai_review, config.contact_policy)
            outcomes[case_id] = outcome
        verdict = decide_policy(timeline, due, by_case.get(case_id, []), outcome, config)
        verdicts[case_id] = verdict
        if drafts_on and draft_writer is not None and verdict.decision == Decision.CONTACT_CANDIDATE:
            drafts[case_id] = write_draft(
                timeline, verdict, draft_writer, config.ai_review, config.contact_policy, config.drafts
            )
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

    return RunResult(
        snapshot=snapshot,
        results=tuple(sorted(results, key=lambda r: r.case_id)),
        issues=tuple(issues),
        findings={},
        policy="v2",
        verdicts=verdicts,
        policy_outcomes=outcomes,
        drafts=drafts,
    )
