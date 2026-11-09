"""Command-line entry point: ``silent-leads validate`` (and later ``run``)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from silent_leads.config import Config, load_config
from silent_leads.io.reader import ExportFormatError, read_export
from silent_leads.models import IssueLevel
from silent_leads.snapshot import SnapshotUnknownError, resolve_snapshot
from silent_leads.timeline import build_timelines
from silent_leads.validation import check_control_sheet, validate_links

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Silent Leads Retouch MVP (offline).")

EXIT_FORMAT = 2
EXIT_CHECKSUM = 3

INPUT_OPTION = typer.Option(..., "--input", "-i", exists=True, dir_okay=False, help="Файл вигрузки .xlsx")
CONFIG_OPTION = typer.Option(None, "--config", "-c", exists=True, dir_okay=False, help="config.yaml")


@app.callback()
def _root() -> None:
    """Silent Leads Retouch MVP — офлайн-обробка вигрузки silent-лідів."""


def _load_config(path: Path | None) -> Config:
    if path is None:
        return Config()
    return load_config(path)


@app.command()
def validate(
    input_path: Path = INPUT_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Прочитати вигрузку, звірити контрольні суми і показати знайдені проблеми даних."""
    config = _load_config(config_path)
    try:
        export = read_export(input_path, config.sheets)
    except ExportFormatError as exc:
        typer.secho(f"Формат файлу: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_FORMAT) from exc

    mismatches = check_control_sheet(export)
    if mismatches:
        typer.secho("Контрольні суми не збігаються:", fg=typer.colors.RED, err=True)
        for line in mismatches:
            typer.echo(f"  - {line}", err=True)
        raise typer.Exit(EXIT_CHECKSUM)

    issues = [*export.issues, *validate_links(export)]
    timelines = build_timelines(export.leads, export.messages)
    try:
        snapshot = resolve_snapshot(config, export)
        typer.echo(
            f"Знімок: {snapshot.utc.isoformat()} -> {snapshot.crm_local.isoformat()} "
            f"({config.crm_timezone}); контрольна дата {snapshot.control_date} [{snapshot.source}]"
        )
    except SnapshotUnknownError as exc:
        typer.secho(f"Знімок: {exc}", fg=typer.colors.YELLOW, err=True)

    typer.echo(
        f"Лідів: {len(export.leads)}, подій: {len(export.messages)}, кейсів з історією: {len(timelines)}"
    )
    after = sum(1 for t in timelines.values() if t.after_silent)
    typer.echo(f"Кейсів з подіями після silent: {after}")

    by_level = Counter(i.level for i in issues)
    typer.echo(
        f"Проблем даних: {len(issues)} "
        f"(error={by_level[IssueLevel.ERROR]}, warning={by_level[IssueLevel.WARNING]})"
    )
    by_code = Counter(i.code for i in issues)
    for code, count in sorted(by_code.items()):
        typer.echo(f"  {code}: {count}")


if __name__ == "__main__":
    app()
