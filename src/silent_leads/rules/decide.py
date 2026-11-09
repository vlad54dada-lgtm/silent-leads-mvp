"""Fold due-date check, history findings, the AI context verdict and data issues into one decision."""

from __future__ import annotations

from dataclasses import dataclass

from silent_leads.models import DataIssue, Decision, IssueLevel
from silent_leads.rules.ai_review import AiConfidence, AiLabel, AiOutcome
from silent_leads.rules.due_date import DueCheck
from silent_leads.rules.history import Finding, FindingKind
from silent_leads.timeline import CaseTimeline

_BLOCKING = {FindingKind.EXACT_OPT_OUT}
_REVIEW = {
    FindingKind.CONTACT_AFTER_SILENT,
    FindingKind.HIDDEN_INBOUND,
    FindingKind.NO_CLIENT_TEXT,
}

_LABEL_TEXT = {
    AiLabel.OPT_OUT: "клієнт просив не контактувати",
    AiLabel.NEGATIVE: "однозначний негатив клієнта",
    AiLabel.UNCLEAR: "контекст неоднозначний",
    AiLabel.NONE: "відмови чи негативу не знайдено",
}
_CONFIDENCE_TEXT = {
    AiConfidence.HIGH: "висока",
    AiConfidence.MEDIUM: "середня",
    AiConfidence.LOW: "низька",
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


def _ai_text(ai: AiOutcome) -> str:
    label = _LABEL_TEXT[ai.label or AiLabel.UNCLEAR]
    confidence = _CONFIDENCE_TEXT[ai.confidence or AiConfidence.LOW]
    explanation = f": {ai.explanation}" if ai.explanation else ""
    return f"{label} (впевненість {confidence}){explanation}"


def decide(
    timeline: CaseTimeline,
    due: DueCheck,
    findings: list[Finding],
    issues: list[DataIssue],
    ai: AiOutcome | None = None,
) -> Verdict:
    """Priority: do_not_contact > manual_review > skip_not_due > contact_candidate.

    ``ai`` is the model's context verdict when the AI review ran for this case. Without it,
    a phrase hint is enough to require a human; with it, the model's label decides whether
    the hint matters.
    """
    own_ids = {m.message_id for m in timeline.messages}
    errors = [i for i in issues if i.level == IssueLevel.ERROR]
    warnings = [i for i in issues if i.level == IssueLevel.WARNING]
    blocking = [f for f in findings if f.kind in _BLOCKING]
    review_kinds = _REVIEW | ({FindingKind.PHRASE_HINT} if ai is None else set())
    review = [f for f in findings if f.kind in review_kinds]
    info = [f for f in findings if f.kind not in _BLOCKING and f.kind not in review_kinds]

    note_parts: list[str] = []
    note_parts.extend(f"[дані] {i.code}: {i.detail}" for i in errors)
    note_parts.extend(f"[дані] {i.code}: {i.detail}" for i in warnings)
    note_parts.extend(f"[інфо] {f.detail}" for f in info)

    block_reasons = [f.detail for f in blocking]
    block_evidence = [mid for f in blocking for mid in f.message_ids]
    review_reasons = [f.detail for f in review]
    review_evidence = [mid for f in review for mid in f.message_ids]
    ai_cleared = False

    if ai is not None:
        tag = f"[AI {ai.model}]"
        trunc = " (вхід обрізано до ліміту)" if ai.truncated else ""
        if ai.error is not None:
            review_reasons.append("AI-перевірку не виконано — потрібен ручний перегляд")
            note_parts.append(f"{tag} помилка: {ai.error}{trunc}")
        elif ai.blocks:
            block_reasons.append(f"AI: {_ai_text(ai)}")
            block_evidence.extend(ai.evidence_message_ids)
            note_parts.append(f"{tag} {ai.label}{trunc}")
        elif ai.clears:
            ai_cleared = True
            note_parts.append(f"{tag} {_ai_text(ai)}{trunc}")
        else:
            review_reasons.append(f"AI: {_ai_text(ai)}")
            review_evidence.extend(ai.evidence_message_ids)
            note_parts.append(f"{tag} {ai.label}{trunc}")

    if block_reasons:
        decision = Decision.DO_NOT_CONTACT
        reason = "; ".join(block_reasons)
        evidence = _dedupe(block_evidence)
        note_parts.extend(f"[перевірка] {r}" for r in review_reasons)
        note_parts.append(f"[строк] {due.explain()}")
    elif errors or review_reasons or due.due is None:
        decision = Decision.MANUAL_REVIEW
        reasons = list(review_reasons)
        if due.due is None:
            reasons.append(due.explain())
        if errors:
            reasons.append(f"проблеми даних: {', '.join(sorted({i.code for i in errors}))}")
        reason = "; ".join(reasons)
        evidence = _dedupe(review_evidence)
        if due.due is not None:
            note_parts.append(f"[строк] {due.explain()}")
    elif not due.due:
        decision = Decision.SKIP_NOT_DUE
        reason = due.explain()
        evidence = ()
    else:
        decision = Decision.CONTACT_CANDIDATE
        context = (
            "AI не знайшла відмови чи негативу в контексті діалогу"
            if ai_cleared
            else "явної відмови в історії не знайдено, контекст достатній"
        )
        reason = f"{due.explain()}; {context}"
        evidence = ()

    leaked = [mid for mid in evidence if mid not in own_ids]
    if leaked:
        msg = f"case {timeline.lead.case_id}: evidence ids not from this case: {leaked}"
        raise EvidenceLeakError(msg)

    return Verdict(decision, reason, evidence, " | ".join(note_parts))
