"""Read a local ``.env`` file so the API key can live next to the project instead of the shell.

Only ``KEY=VALUE`` lines are understood (an optional ``export`` prefix and quotes are accepted).
A variable that is already set in the environment wins, values are never printed or returned, and
the file is optional. The path can be overridden with ``SILENT_LEADS_ENV_FILE`` (tests point it at
a file that does not exist so a developer's real ``.env`` never leaks into them).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ENV_FILE_VARIABLE = "SILENT_LEADS_ENV_FILE"
_LINE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _read_text(path: Path) -> str:
    """Windows PowerShell 5 writes ``>`` redirections as UTF-16; accept that as well as UTF-8."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def load_env_file(path: Path | None = None) -> list[str]:
    """Copy ``KEY=VALUE`` pairs from the file into ``os.environ``; return the NAMES that were set."""
    target = path if path is not None else Path(os.environ.get(ENV_FILE_VARIABLE) or ".env")
    if not target.is_file():
        return []
    loaded: list[str] = []
    for raw_line in _read_text(target).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if match is None:
            continue
        name, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded
