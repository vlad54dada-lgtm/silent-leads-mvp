"""Command-line entry point.

``validate``  check the export;  ``run``  full pipeline;
``evaluate-labels``  compare human labels with AI labels (ai_labels.csv);
``ai-eval``   run the AI context analysis over synthetic dialogues with known labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer

from silent_leads.ai_eval import load_eval_set, run_ai_eval
from silent_leads.config import Config, load_config
from silent_leads.evaluation import evaluate, read_labels_csv, render_evaluation_markdown
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
EXIT_AI_DISABLED = 5

INPUT_OPTION = typer.Option(..., "--input", "-i", exists=True, dir_okay=False, help="Файл вигрузки .xlsx")
CONFIG_OPTION = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, help="config.yaml")
OUTPUT_OPTION = typer.Option(Path("data/output"), "--output", "-o", file_okay=False, help="Папка результатів")
LABELS_OPTION = typer.Option(
    ..., "--labels", "-l", exists=True, dir_okay=False, help="ai_labels.csv із заповненою human_label"
)
CASES_OPTION = typer.Option(
    Path("evals/context_cases.yaml"),
    "--cases",
    exists=True,
    dir_okay=False,
    help="Синтетичні діалоги з мітками",
)


@app.callback()
def _root() -> None:
    """Silent Leads Retouch MVP — офлайн-обробка вигрузки silent-лідів."""


def _load_config(path: Path) -> Config:
    # Always explicit: the bare Config() has no phrase hints, which would silently weaken the
    # offline rules. The example config is the documented baseline.
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
    config_path: Path = CONFIG_OPTION,
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
    config_path: Path = CONFIG_OPTION,
    output_dir: Path = OUTPUT_OPTION,
) -> None:
    """Повний прогін: рішення по кожному кейсу, results.xlsx/csv, data_issues.csv, ai_labels.csv, summary."""
    config = _load_config(config_path)
    export = _read_checked(input_path, config)
    try:
        result = run_pipeline(config, export)
    except SnapshotUnknownError as exc:
        typer.secho(f"Знімок: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_SNAPSHOT) from exc

    summary = build_summary(config, result)
    written = write_outputs(
        output_dir,
        result.results,
        result.issues,
        summary,
        render_summary_markdown(summary),
        result.ai_outcomes,
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
            f"AI-аналіз ({ai['model']}): кейсів {ai['reviewed']}, заблоковано {ai['blocked']}, "
            f"без зауважень {ai['cleared']}, на ручну {ai['needs_review']}, помилок {ai['errors']}"
        )
        labels = ", ".join(f"{k}={v}" for k, v in ai["by_label"].items())
        typer.echo(f"  мітки: {labels}")
    typer.echo("Записано:")
    for path in written:
        typer.echo(f"  {path}")


@app.command()
def evaluate_labels(
    labels_path: Path = LABELS_OPTION,
    output_dir: Path = OUTPUT_OPTION,
) -> None:
    """Порівняти розмітку людини (human_label в ai_labels.csv) з мітками AI: точність і розбіжності."""
    rows = read_labels_csv(labels_path)
    report = evaluate(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_evaluation_markdown(report, "Валідація AI-міток на розмітці людини")
    (output_dir / "evaluation.md").write_text(markdown, encoding="utf-8")
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _echo_report(report)
    typer.echo(f"Записано: {output_dir / 'evaluation.md'}, {output_dir / 'evaluation.json'}")


@app.command()
def ai_eval(
    cases_path: Path = CASES_OPTION,
    config_path: Path = CONFIG_OPTION,
    output_dir: Path = OUTPUT_OPTION,
) -> None:
    """Прогнати AI-аналіз по синтетичних діалогах з відомими мітками (регресія промпта)."""
    config = _load_config(config_path)
    if not config.ai_review.enabled:
        typer.secho(
            "ai_review.enabled: false — увімкніть AI-аналіз у конфігу.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(EXIT_AI_DISABLED)
    from silent_leads.rules.ai_review import ClaudeReviewer

    eval_set = load_eval_set(cases_path)
    rows = run_ai_eval(eval_set, ClaudeReviewer(config.ai_review), config.ai_review)
    report = evaluate(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    title = f"AI-аналіз на синтетичному наборі ({cases_path.name}, модель {config.ai_review.model})"
    (output_dir / "ai_eval.md").write_text(render_evaluation_markdown(report, title), encoding="utf-8")
    (output_dir / "ai_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _echo_report(report)
    typer.echo(f"Записано: {output_dir / 'ai_eval.md'}, {output_dir / 'ai_eval.json'}")


def _echo_report(report: dict[str, object]) -> None:
    def pct(value: object) -> str:
        return "-" if value is None else f"{float(value) * 100:.1f}%"  # type: ignore[arg-type]

    typer.echo(
        f"Порівняно: {report['comparable']} з {report['labelled']} розмічених; "
        f"точність по мітках {pct(report['exact_accuracy'])}, "
        f"на рівні рішення {pct(report['decision_accuracy'])}"
    )
    mismatches = report["mismatches"]
    assert isinstance(mismatches, list)
    typer.echo(f"Розбіжностей: {len(mismatches)}")
    for m in mismatches:
        typer.echo(
            f"  {m['case_id']}: еталон {m['expected']}, AI {m['predicted']} ({m['confidence'] or '?'})"
        )


if __name__ == "__main__":
    app()
