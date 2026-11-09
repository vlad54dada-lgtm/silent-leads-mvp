"""First re-touch due date: calendar days in the CRM timezone, from silent_since_crm."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from silent_leads.models import Lead


@dataclass(frozen=True)
class DueCheck:
    """``due`` is None when the silent date is unknown — the rule cannot be applied."""

    due: bool | None
    silent_date: date | None
    earliest_contact_date: date | None
    days_since_silent: int | None
    delay_days: int

    def explain(self) -> str:
        if self.silent_date is None:
            return "дата переходу в silent відсутня або некоректна — строк перевірити неможливо"
        assert self.earliest_contact_date is not None and self.days_since_silent is not None
        if self.due:
            return (
                f"silent з {self.silent_date}, минуло {self.days_since_silent} дн. "
                f"(>= {self.delay_days}) — строк першого касання настав"
            )
        return (
            f"silent з {self.silent_date}, минуло {self.days_since_silent} дн. "
            f"(< {self.delay_days}) — касання можливе з {self.earliest_contact_date}"
        )


def check_due(lead: Lead, control_date: date, delay_days: int) -> DueCheck:
    """Compare calendar dates only. Both are already in CRM local time.

    ``silent_since_crm`` is CRM-local wall-clock time (naive), ``control_date`` is the
    snapshot instant converted to the CRM timezone — so no further conversion is needed.
    """
    if lead.silent_since_crm is None:
        return DueCheck(None, None, None, None, delay_days)
    silent_date = lead.silent_since_crm.date()
    days = (control_date - silent_date).days
    return DueCheck(
        due=days >= delay_days,
        silent_date=silent_date,
        earliest_contact_date=silent_date + timedelta(days=delay_days),
        days_since_silent=days,
        delay_days=delay_days,
    )
