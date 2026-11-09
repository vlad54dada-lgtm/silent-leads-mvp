"""Resolve the control date: from config, else from the export's «Інструкція» sheet."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from silent_leads.config import Config
from silent_leads.io.reader import Export

_INSTRUCTION_SNAPSHOT_KEYS = ("Знімок UTC", "snapshot_utc")


class SnapshotUnknownError(Exception):
    """Neither config nor the export says when the snapshot was taken."""


@dataclass(frozen=True)
class Snapshot:
    utc: datetime
    crm_local: datetime
    source: str

    @property
    def control_date(self) -> date:
        return self.crm_local.date()


def resolve_snapshot(config: Config, export: Export) -> Snapshot:
    tz = ZoneInfo(config.crm_timezone)
    if config.snapshot_utc is not None:
        utc = config.snapshot_utc
        source = "config.snapshot_utc"
    else:
        raw = next(
            (export.instruction[k] for k in _INSTRUCTION_SNAPSHOT_KEYS if k in export.instruction), None
        )
        if raw is None:
            msg = "snapshot_utc не задано в конфігу і не знайдено у вкладці «Інструкція» (ключ «Знімок UTC»)"
            raise SnapshotUnknownError(msg)
        try:
            utc = datetime.fromisoformat(raw.strip())
        except ValueError as exc:
            msg = f"«Знімок UTC» = '{raw}' не є ISO-датою"
            raise SnapshotUnknownError(msg) from exc
        if utc.tzinfo is None:
            msg = f"«Знімок UTC» = '{raw}' не містить часової зони"
            raise SnapshotUnknownError(msg)
        source = "export.Інструкція.Знімок UTC"
    return Snapshot(utc=utc, crm_local=utc.astimezone(tz), source=source)
