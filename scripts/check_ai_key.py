"""Diagnose the AI provider key without ever printing it.

Reads the key from the environment variable named in the config (ai_review.api_key_env),
asks the provider who the key belongs to and sends one tiny test message. Prints only
HTTP statuses, the provider's error text and the key's length/prefix shape.

Usage: python scripts/check_ai_key.py [config.yaml]
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silent_leads.config import load_config


def _shape(key: str) -> str:
    head = key[:6] if key.startswith("sk-") else key[:2]
    return f"len={len(key)}, starts with '{head}...', whitespace inside={any(ch.isspace() for ch in key)}"


def _call(url: str, headers: dict[str, str], body: dict[str, object] | None = None) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode("utf-8")[:300]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:300]
    except OSError as exc:
        return 0, f"network: {exc}"


def main() -> int:
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    settings = load_config(config_path).ai_review
    key = os.environ.get(settings.api_key_env, "")
    print(f"config: model={settings.model} base_url={settings.base_url or 'https://api.anthropic.com'}")
    shape = f" ({_shape(key)})" if key else ""
    print(f"env var {settings.api_key_env}: {'set' if key else 'NOT SET'}{shape}")
    if not key:
        return 1
    base = (settings.base_url or "https://api.anthropic.com").rstrip("/")
    json_headers = {"Content-Type": "application/json"}

    if "openrouter.ai" in base:
        status, text = _call(f"{base}/v1/auth/key", {"Authorization": f"Bearer {key}"})
        print(f"GET /v1/auth/key (Bearer): {status} {text}")

    body = {
        "model": settings.model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
    }
    for name, auth in (
        ("x-api-key", {"x-api-key": key, "anthropic-version": "2023-06-01"}),
        ("Bearer", {"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"}),
    ):
        status, text = _call(f"{base}/v1/messages", {**json_headers, **auth}, body)
        print(f"POST /v1/messages ({name}): {status} {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
