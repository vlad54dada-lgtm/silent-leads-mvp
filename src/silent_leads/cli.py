"""Command-line entry point: ``validate`` (check the export) and ``run`` (full pipeline)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from silent_leads.config import Config, load_config
from silent_leads.io.reader import Export, ExportFormatError, read_export
from silent_leads.io.writer import write_outputs
from silent_leads.models import IssueLevel
from silent_leads.pipeline import run_pipeline
from silent_leads.snapshot import SnapshotUnknownError, resolve_snapshot
from silent_leads.summary import build_summary, render_summary_markdown
from silent_leads.timeline import build_timelines
from silent_leads.validation import check_control_sheet, validate_links

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Silent Leads Retouch MVP (offline).")

EXIT_FORMAT = 2
EXIT_CHECKSUM = 3
EXIT_SNAPSHOT = 4

INPUT_OPTION = typer.Option(..., "--input", "-i", exists=True, dir_okay=False, help="Файл вигрузки .xlsx")
CONFIG_OPTION = typer.Option(None, "--config", "-c", exists=True, dir_okay=False, help="config.yaml")
OUTPUT_OPTION = typer.Option(Path("data/output"), "--output", "-o", file_okay=False, help="Папка результатів")


@app.callback()
def _root() -> None:
    """Silent Leads Retouch MVP — офлайн-обробка вигрузки silent-лідів."""


def _load_config(path: Path | None) -> Config:
    if path is None:
        return Config()
    return load_config(path)


def _read_checked(input_path: Path, config: Config) -> Export:
    """Read the workbook and stop on format or checksum problems — never guess on a wrong file."""
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
    return export


@app.command()
def validate(
    input_path: Path = INPUT_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Прочитати вигрузку, звірити контрольні суми і показати знайдені проблеми даних."""
    config = _load_config(config_path)
    export = _read_checked(input_path, config)

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


@app.command()
def run(
    input_path: Path = INPUT_OPTION,
    config_path: Path | None = CONFIG_OPTION,
    output_dir: Path = OUTPUT_OPTION,
) -> None:
    """Повний прогін: рішення по кожному кейсу, results.xlsx/csv, data_issues.csv, summary.md/json."""
    config = _load_config(config_path)
    export = _read_checked(input_path, config)
    try:
        result = run_pipeline(config, export)
    except SnapshotUnknownError as exc:
        typer.secho(f"Знімок: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_SNAPSHOT) from exc

    summary = build_summary(config, result)
    written = write_outputs(
        output_dir, result.results, result.issues, summary, render_summary_markdown(summary)
    )

    typer.echo(
        f"Контрольна дата {summary['control_date']} ({config.crm_timezone}), "
        f"строк {config.first_touch_delay_days} дн."
    )
    typer.echo(f"Кейсів: {summary['cases_total']}")
    for decision, count in summary["decisions"].items():
        typer.echo(f"  {decision}: {count}")
    typer.echo(
        f"Проблем даних: {summary['data_issues']['total']} "
        f"(error={summary['data_issues']['errors']}, warning={summary['data_issues']['warnings']})"
    )
    ai = summary["ai_review"]
    if ai["enabled"]:
        typer.echo(
            f"AI-перевірка ({ai['model']}): перевірено {ai['reviewed']}, "
            f"на ручну {ai['flagged']}, помилок {ai['errors']}"
        )
    typer.echo("Записано:")
    for path in written:
        typer.echo(f"  {path}")


if __name__ == "__main__":
    app()
