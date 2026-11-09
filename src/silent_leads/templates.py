"""Pick the draft SMS for contact candidates. Never invents an approved text."""

from __future__ import annotations

from silent_leads.config import Templates
from silent_leads.models import Decision, Lead, TemplateStatus


def prepare_draft(lead: Lead, decision: Decision, templates: Templates) -> tuple[str, TemplateStatus]:
    """Return (draft_sms, template_status).

    Text is produced only for contact_candidate. A template whose status is
    template_missing, or whose text is empty, yields an empty draft and template_missing.
    """
    if decision != Decision.CONTACT_CANDIDATE:
        return "", TemplateStatus.NOT_APPLICABLE
    if lead.has_quote is None:
        # Cannot happen for a contact candidate (bad has_quote is a data error), kept for safety.
        return "", TemplateStatus.TEMPLATE_MISSING
    template = templates.with_quote if lead.has_quote else templates.without_quote
    text = template.text.strip()
    if not text or template.status == TemplateStatus.TEMPLATE_MISSING:
        return "", TemplateStatus.TEMPLATE_MISSING
    return text, template.status
