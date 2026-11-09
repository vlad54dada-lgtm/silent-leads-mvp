from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from silent_leads.config import Config
from silent_leads.rules.due_date import check_due
from silent_leads.snapshot import Snapshot
from tests.test_timeline import _lead

CONTROL = date(2026, 11, 5)


@pytest.mark.parametrize(
    ("silent", "due", "days"),
    [
        ("2026-10-28 23:59:59", True, 8),  # clearly past the delay
        ("2026-10-29 00:00:01", True, 7),  # exactly 7 calendar days -> due (>=)
        ("2026-10-30 00:00:00", False, 6),  # one day short
        ("2026-11-05 06:00:00", False, 0),  # went silent on the control date
    ],
)
def test_due_uses_calendar_days_and_inclusive_delay(silent: str, due: bool, days: int) -> None:
    check = check_due(_lead(silent=silent), CONTROL, 7)
    assert check.due is due
    assert check.days_since_silent == days
    assert check.earliest_contact_date == date.fromisoformat(silent[:10]) + timedelta(days=7)


def test_delay_is_configurable() -> None:
    lead = _lead(silent="2026-11-02 10:00:00")  # 3 days before control
    assert check_due(lead, CONTROL, 7).due is False
    assert check_due(lead, CONTROL, 3).due is True
    assert check_due(lead, CONTROL, 0).due is True


def test_unknown_silent_date_cannot_be_decided() -> None:
    check = check_due(_lead(silent=None), CONTROL, 7)
    assert check.due is None
    assert "неможливо" in check.explain()


def test_control_date_is_taken_in_crm_timezone_not_utc() -> None:
    # 03:30 UTC on 6 Nov is 19:30 on 5 Nov in Los Angeles: control date must be the 5th.
    utc = datetime(2026, 11, 6, 3, 30, tzinfo=ZoneInfo("UTC"))
    snapshot = Snapshot(utc=utc, crm_local=utc.astimezone(ZoneInfo("America/Los_Angeles")), source="test")
    assert snapshot.control_date == date(2026, 11, 5)
    # A lead silent since 29 Oct is due on the 5th (7 days) but would look like 8 days in UTC — same verdict,
    # whereas one silent since 30 Oct is NOT due on the 5th yet would be due if we wrongly used the UTC date.
    assert check_due(_lead(silent="2026-10-30 08:00:00"), snapshot.control_date, 7).due is False
    assert check_due(_lead(silent="2026-10-30 08:00:00"), utc.date(), 7).due is True


def test_default_config_delay_is_seven() -> None:
    assert Config().first_touch_delay_days == 7
