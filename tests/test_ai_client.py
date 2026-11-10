"""Shared structured Claude client (v2 judge and draft writer): key, fallback, refusal. No network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2  # the HTTP client vendored by the anthropic SDK
import pytest

from silent_leads.ai_client import ModelRefusedError, StructuredClaude, parse_json_answer
from silent_leads.config import AiReview
from silent_leads.rules.contact_policy import SYSTEM_PROMPT, ClaudePolicyJudge, PolicyAssessment
from silent_leads.rules.drafts import ClaudeDraftWriter, DraftAnswer

SETTINGS = AiReview(
    enabled=True, model="m", base_url="https://gateway.example/api", api_key_env="GATEWAY_KEY"
)
GOOD_JSON = (
    '{"label": "contact_ok", "reason": "price", "confidence": "high", '
    '"last_client_position_message_id": "L1-M2", "evidence_message_ids": ["L1-M2"], '
    '"stop_then_resumed": false, "question_for_owner": "", "explanation": "ок"}'
)


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Response:
    stop_reason: str = "end_turn"
    parsed_output: Any = None
    content: list[_Block] = field(default_factory=list)


class _Messages:
    def __init__(self, parse_result: Any, create_result: Any) -> None:
        self.parse_result, self.create_result = parse_result, create_result
        self.parse_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        if isinstance(self.parse_result, Exception):
            raise self.parse_result
        return self.parse_result

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return self.create_result


def _install(monkeypatch: pytest.MonkeyPatch, parse_result: Any, create_result: Any = None) -> list[Any]:
    created: list[Any] = []

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.messages = _Messages(parse_result, create_result)
            created.append(self)

    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    monkeypatch.setenv("GATEWAY_KEY", "test-value")
    return created


def _bad_request() -> anthropic.BadRequestError:
    request = httpx2.Request("POST", "https://example.invalid/v1/messages")
    return anthropic.BadRequestError("unsupported", response=httpx2.Response(400, request=request), body=None)


def test_structured_answer_uses_named_key_base_url_and_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = PolicyAssessment.model_validate_json(GOOD_JSON)
    created = _install(monkeypatch, _Response(parsed_output=parsed))
    answer = ClaudePolicyJudge(SETTINGS).judge("L1", "[]")
    assert answer is parsed
    client = created[0]
    assert client.kwargs == {"api_key": "test-value", "base_url": "https://gateway.example/api"}
    call = client.messages.parse_calls[0]
    assert call["system"] == SYSTEM_PROMPT and call["output_format"] is PolicyAssessment
    assert "Case L1" in call["messages"][0]["content"]


def test_gateway_without_structured_output_falls_back_to_validated_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install(
        monkeypatch, _bad_request(), _Response(content=[_Block("```json\n" + GOOD_JSON + "\n```")])
    )
    answer = StructuredClaude(SETTINGS).ask("system", "content", PolicyAssessment)
    assert answer.label == "contact_ok"
    assert "JSON" in created[0].messages.create_calls[0]["system"]


@pytest.mark.parametrize(
    "text", ["no json here", '{"label": "maybe"}', GOOD_JSON.replace('"ок"}', '"ок", "x": 1}')]
)
def test_invalid_plain_answers_raise(text: str) -> None:
    with pytest.raises(ValueError, match=r".+"):
        parse_json_answer(text, PolicyAssessment)


def test_refusal_and_missing_structured_output_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _Response(stop_reason="refusal"))
    with pytest.raises(ModelRefusedError):
        StructuredClaude(SETTINGS).ask("s", "c", PolicyAssessment)
    _install(monkeypatch, _Response(parsed_output=None))
    with pytest.raises(ValueError, match="структуровану"):
        StructuredClaude(SETTINGS).ask("s", "c", PolicyAssessment)


def test_missing_key_is_an_error_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install(monkeypatch, _Response())
    monkeypatch.delenv("GATEWAY_KEY")
    with pytest.raises(RuntimeError, match="GATEWAY_KEY"):
        StructuredClaude(SETTINGS).ask("s", "c", PolicyAssessment)
    assert created[0].messages.parse_calls == []


def test_draft_writer_puts_the_length_limit_into_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from silent_leads.config import Drafts

    created = _install(monkeypatch, _Response(parsed_output=DraftAnswer(sms="Hi there, any thoughts?")))
    answer = ClaudeDraftWriter(SETTINGS, Drafts(max_chars=200)).write("L1", "content")
    assert answer.sms.startswith("Hi")
    system = created[0].messages.parse_calls[0]["system"]
    assert "at most 200 characters" in system and "{max_chars}" not in system
