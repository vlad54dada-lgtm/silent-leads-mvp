"""Shared Claude client returning a validated pydantic object (v2 policy judge and draft writer).

The key is read from the environment variable named in ``AiReview.api_key_env``; an
Anthropic-compatible gateway (e.g. OpenRouter) is reached via ``AiReview.base_url``.
Structured output is tried first; gateways that reject it get a plain-JSON request whose
answer is validated against the same schema. A refusal or any mismatch raises — callers
turn every exception into a ``manual_review`` outcome.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from silent_leads.config import AiReview

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class ModelRefusedError(RuntimeError):
    """The model declined to answer (stop_reason == "refusal")."""


def parse_json_answer[T: BaseModel](text: str, schema: type[T]) -> T:
    """Validate a plain-text answer that should contain one JSON object. Raises on anything else."""
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        msg = "відповідь моделі не містить JSON-об'єкта"
        raise ValueError(msg)
    return schema.model_validate_json(match.group(0))


def json_instruction(schema: type[BaseModel]) -> str:
    fields = ", ".join(f'"{name}"' for name in schema.model_fields)
    return (
        "Reply with a single JSON object and nothing else, no markdown fences, with exactly these keys: "
        f"{fields}."
    )


def _read_cached[T: BaseModel](path: Path | None, schema: type[T]) -> T | None:
    if path is None or not path.is_file():
        return None
    try:
        return schema.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None  # a damaged or outdated file is ignored and overwritten by a fresh answer


class StructuredClaude:
    def __init__(self, settings: AiReview, max_tokens: int = 2048) -> None:
        import anthropic  # imported here so the offline path never needs the SDK

        self._anthropic = anthropic
        self.model = settings.model
        self._api_key_env = settings.api_key_env
        self._max_tokens = max_tokens
        api_key = os.environ.get(settings.api_key_env) or None
        self._has_key = api_key is not None
        self._cache_dir = Path(settings.cache_dir) if settings.cache_dir else None
        self._client = anthropic.Anthropic(api_key=api_key, base_url=settings.base_url)

    def ask[T: BaseModel](self, system: str, content: str, schema: type[T]) -> T:
        cache_file = self._cache_file(system, content, schema)
        cached = _read_cached(cache_file, schema)
        if cached is not None:
            return cached
        if not self._has_key:
            msg = f"змінну середовища {self._api_key_env} не задано"
            raise RuntimeError(msg)
        try:
            answer = self._structured(system, content, schema)
        except self._anthropic.BadRequestError:
            answer = self._plain_json(system, content, schema)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(answer.model_dump_json(), encoding="utf-8")
        return answer

    def _cache_file(self, system: str, content: str, schema: type[BaseModel]) -> Path | None:
        """A change of model, prompt, schema or dialogue gives another file: no stale answer is reused."""
        if self._cache_dir is None:
            return None
        raw = json.dumps([self.model, schema.__name__, system, content], ensure_ascii=False)
        return self._cache_dir / f"{hashlib.sha256(raw.encode('utf-8')).hexdigest()}.json"

    def _structured[T: BaseModel](self, system: str, content: str, schema: type[T]) -> T:
        response: Any = self._client.messages.parse(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=schema,
        )
        if response.stop_reason == "refusal":
            raise ModelRefusedError("модель відмовилась відповідати")
        parsed = response.parsed_output
        if not isinstance(parsed, schema):
            msg = "модель не повернула структуровану відповідь"
            raise ValueError(msg)
        return parsed

    def _plain_json[T: BaseModel](self, system: str, content: str, schema: type[T]) -> T:
        response: Any = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system + "\n" + json_instruction(schema),
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "refusal":
            raise ModelRefusedError("модель відмовилась відповідати")
        text = "".join(block.text for block in response.content if block.type == "text")
        return parse_json_answer(text, schema)
