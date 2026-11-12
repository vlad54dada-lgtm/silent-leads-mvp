"""The local ``.env`` file: the API key may live there; the shell environment still wins."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from silent_leads.cli import app
from silent_leads.env_file import ENV_FILE_VARIABLE, load_env_file

NAMES = ("SL_TEST_PLAIN", "SL_TEST_QUOTED", "SL_TEST_EXPORTED", "SL_TEST_EMPTY", "SL_TEST_SET")


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)


def test_pairs_are_loaded_and_only_names_are_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SL_TEST_SET", "from-shell")
    path = tmp_path / ".env"
    path.write_text(
        "# a comment\n\nSL_TEST_PLAIN=abc\nSL_TEST_QUOTED = \"q w\"\nexport SL_TEST_EXPORTED='x'\n"
        "SL_TEST_EMPTY=\nSL_TEST_SET=from-file\nnot a pair\n",
        encoding="utf-8",
    )
    loaded = load_env_file(path)
    for name in loaded:  # set by the loader, so monkeypatch does not know to remove them
        monkeypatch.setenv(name, os.environ[name])
    assert loaded == ["SL_TEST_PLAIN", "SL_TEST_QUOTED", "SL_TEST_EXPORTED"]
    assert os.environ["SL_TEST_PLAIN"] == "abc"
    assert os.environ["SL_TEST_QUOTED"] == "q w"
    assert os.environ["SL_TEST_EXPORTED"] == "x"
    assert "SL_TEST_EMPTY" not in os.environ  # an empty value is the same as no value
    assert os.environ["SL_TEST_SET"] == "from-shell"  # the environment wins over the file


def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "absent.env") == []


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16"])
def test_file_written_by_windows_powershell_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoding: str
) -> None:
    path = tmp_path / ".env"
    path.write_text("SL_TEST_PLAIN=abc\r\n", encoding=encoding)
    loaded = load_env_file(path)
    monkeypatch.setenv("SL_TEST_PLAIN", os.environ["SL_TEST_PLAIN"])
    assert loaded == ["SL_TEST_PLAIN"]
    assert os.environ["SL_TEST_PLAIN"] == "abc"


def test_cli_loads_the_env_file_before_any_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "demo.env"
    path.write_text("SL_TEST_PLAIN=abc\n", encoding="utf-8")
    monkeypatch.setenv(ENV_FILE_VARIABLE, str(path))
    result = CliRunner().invoke(app, ["validate", "--help"])
    monkeypatch.setenv("SL_TEST_PLAIN", os.environ.get("SL_TEST_PLAIN", ""))
    assert result.exit_code == 0
    assert os.environ["SL_TEST_PLAIN"] == "abc"
    assert "abc" not in result.output  # values are never printed
