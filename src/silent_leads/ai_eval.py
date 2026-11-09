"""Run the AI context analysis over synthetic dialogues with known labels (prompt regression set).

The real export may contain no explicit opt-out at all, so the prompt is exercised on
hand-written cases in ``evals/*.yaml``: each has a dialogue and the label a careful human
would give. No export data may ever be copied into those files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from silent_leads.config import AiReview
from silent_leads.evaluation import EvalRow
from silent_leads.rules.ai_review import AiLabel, Reviewer, payload_from_items, review_payload
from silent_leads.timeline import Speaker


class EvalTurn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    speaker: Speaker
    text: str = ""
    hidden: bool = False


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    expected: AiLabel
    note: str = ""
    dialogue: list[EvalTurn] = Field(min_length=1)


class EvalSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str = ""
    cases: list[EvalCase] = Field(min_length=1)


def load_eval_set(path: Path) -> EvalSet:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    eval_set = EvalSet.model_validate(raw)
    ids = [c.id for c in eval_set.cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        msg = f"duplicate eval case ids: {duplicates}"
        raise ValueError(msg)
    return eval_set


@dataclass(frozen=True)
class EvalItems:
    items: list[dict[str, Any]]
    own_ids: set[str]


def case_items(case: EvalCase) -> EvalItems:
    """Synthetic message ids ``<case>-<n>`` so evidence can be checked like a real case."""
    items: list[dict[str, Any]] = []
    for index, turn in enumerate(case.dialogue, start=1):
        message_id = f"{case.id}-{index}"
        if turn.hidden:
            items.append({"message_id": message_id, "speaker": turn.speaker.value, "hidden": True})
        else:
            items.append({"message_id": message_id, "speaker": turn.speaker.value, "text": turn.text})
    return EvalItems(items, {i["message_id"] for i in items})


def run_ai_eval(eval_set: EvalSet, reviewer: Reviewer, settings: AiReview) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for case in sorted(eval_set.cases, key=lambda c: c.id):
        prepared = case_items(case)
        payload, truncated = payload_from_items(prepared.items, settings.max_input_chars)
        outcome = review_payload(case.id, payload, truncated, prepared.own_ids, reviewer, settings)
        rows.append(
            EvalRow(
                case_id=case.id,
                expected=case.expected.value,
                predicted=outcome.label.value if outcome.label else None,
                confidence=outcome.confidence.value if outcome.confidence else "",
                explanation=outcome.explanation,
                error=outcome.error or "",
                raw_expected=case.expected.value,
            )
        )
    return rows
