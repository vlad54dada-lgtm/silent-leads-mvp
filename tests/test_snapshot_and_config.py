from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from silent_leads.config import Config, load_config
from silent_leads.io.reader import read_export
from silent_leads.models import TemplateStatus
from silent_leads.snapshot import SnapshotUnknownError, resolve_snapshot
from tests.conftest import build_workbook, lead


def test_snapshot_from_instruction_sheet_converted_to_crm_timezone(workbook_path: Path) -> None:
    build_workbook(workbook_path, [lead("L1")], [], instruction={"Знімок UTC": "2026-11-06T03:30:00Z"})
    export = read_export(workbook_path)
    snapshot = resolve_snapshot(Config(crm_timezone="America/Los_Angeles"), export)
    # 03:30 UTC on the 6th is still the 5th in Los Angeles (UTC-8 in November).
    assert snapshot.control_date == date(2026, 11, 5)
    assert snapshot.source.startswith("export.")


def test_snapshot_from_config_wins_over_sheet(workbook_path: Path) -> None:
    build_workbook(workbook_path, [lead("L1")], [])
    export = read_export(workbook_path)
    config = Config(snapshot_utc=datetime(2026, 12, 1, 12, 0, tzinfo=UTC))
    assert resolve_snapshot(config, export).control_date == date(2026, 12, 1)
    assert resolve_snapshot(config, export).source == "config.snapshot_utc"


def test_snapshot_missing_everywhere_is_an_error(workbook_path: Path) -> None:
    build_workbook(workbook_path, [lead("L1")], [], instruction={"Інше": "x"})
    export = read_export(workbook_path)
    with pytest.raises(SnapshotUnknownError):
        resolve_snapshot(Config(), export)


def test_naive_snapshot_is_rejected(workbook_path: Path) -> None:
    build_workbook(workbook_path, [lead("L1")], [], instruction={"Знімок UTC": "2026-11-05 14:53:41"})
    export = read_export(workbook_path)
    with pytest.raises(SnapshotUnknownError, match="зон"):
        resolve_snapshot(Config(), export)
    with pytest.raises(ValidationError):
        Config(snapshot_utc=datetime(2026, 11, 5, 14, 53))


def test_config_defaults_are_safe_and_example_loads(tmp_path: Path) -> None:
    config = Config()
    assert config.first_touch_delay_days == 7
    assert config.templates.with_quote.status == TemplateStatus.TEMPLATE_MISSING
    assert config.templates.with_quote.text == ""

    example = load_config(Path(__file__).resolve().parents[1] / "config.example.yaml")
    assert example.templates.without_quote.status == TemplateStatus.TEMPLATE_MISSING
    assert "stop" in example.phrases.opt_out

    with pytest.raises(ValidationError):
        Config(crm_timezone="Mars/Olympus")
    with pytest.raises(ValidationError):
        Config.model_validate({"templates": {"with_quote": {"text": "x", "status": "not_applicable"}}})
