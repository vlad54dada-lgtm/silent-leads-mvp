"""Fold due-date check, history findings and data issues into one decision per case."""

from __future__ import annotations

from dataclasses import dataclass

from silent_leads.models import DataIssue, Decision, IssueLevel
from silent_leads.rules.due_date import DueCheck
from silent_leads.rules.history import Finding, FindingKind
from silent_leads.timeline import CaseTimeline

_BLOCKING = {FindingKind.OPT_OUT, FindingKind.NEGATIVE}
_REVIEW = {
    FindingKind.PHRASE_UNKNOWN_SPEAKER,
    FindingKind.CONTACT_AFTER_SILENT,
    FindingKind.HIDDEN_INBOUND,
    FindingKind.NO_CLIENT_TEXT,
}


class EvidenceLeakError(Exception):
    """A rule produced evidence ids that do not belong to the case — a bug, never written out."""


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str
    evidence_message_ids: tuple[str, ...]
    reviewer_note: str


def _dedupe(ids: list[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for message_id in ids:
        seen.setdefault(message_id, None)
    return tuple(seen)


def decide(
    timeline: CaseTimeline, due: DueCheck, findings: list[Finding], issues: list[DataIssue]
) -> Verdict:
    """Priority: do_not_contact > manual_review > skip_not_due > contact_candidate."""
    own_ids = {m.message_id for m in timeline.messages}
    errors = [i for i in issues if i.level == IssueLevel.ERROR]
    warnings = [i for i in issues if i.level == IssueLevel.WARNING]
    blocking = [f for f in findings if f.kind in _BLOCKING]
    review = [f for f in findings if f.kind in _REVIEW]
    info = [f for f in findings if f.kind not in _BLOCKING and f.kind not in _REVIEW]

    note_parts: list[str] = []
    note_parts.extend(f"[дані] {i.code}: {i.detail}" for i in errors)
    note_parts.extend(f"[дані] {i.code}: {i.detail}" for i in warnings)
    note_parts.extend(f"[інфо] {f.detail}" for f in info)

    if blocking:
        decision = Decision.DO_NOT_CONTACT
        reason = "; ".join(f.detail for f in blocking)
        evidence = _dedupe([mid for f in blocking for mid in f.message_ids])
        note_parts.extend(f"[перевірка] {f.detail}" for f in review)
        note_parts.append(f"[строк] {due.explain()}")
    elif errors or review or due.due is None:
        decision = Decision.MANUAL_REVIEW
        reasons = [f.detail for f in review]
        if due.due is None:
            reasons.append(due.explain())
        if errors:
            reasons.append(f"проблеми даних: {', '.join(sorted({i.code for i in errors}))}")
        reason = "; ".join(reasons)
        evidence = _dedupe([mid for f in review for mid in f.message_ids])
        if due.due is not None:
            note_parts.append(f"[строк] {due.explain()}")
    elif not due.due:
        decision = Decision.SKIP_NOT_DUE
        reason = due.explain()
        evidence = ()
    else:
        decision = Decision.CONTACT_CANDIDATE
        reason = f"{due.explain()}; явної відмови чи негативу в історії не знайдено, контекст достатній"
        evidence = ()

    leaked = [mid for mid in evidence if mid not in own_ids]
    if leaked:
        msg = f"case {timeline.lead.case_id}: evidence ids not from this case: {leaked}"
        raise EvidenceLeakError(msg)

    return Verdict(decision, reason, evidence, " | ".join(note_parts))
