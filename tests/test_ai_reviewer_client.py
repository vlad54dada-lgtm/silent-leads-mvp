"""ClaudeReviewer wiring: key from the named env var, base_url, plain-JSON fallback. No network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import anthropic
import httpx2  # the HTTP client vendored by the anthropic SDK
import pytest

from silent_leads.config import AiReview
from silent_leads.rules.ai_review import AiLabel, ClaudeReviewer, parse_assessment_text


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _Response:
    stop_reason: str = "end_turn"
    parsed_output: Any = None
    content: list[_TextBlock] = field(default_factory=list)


def _bad_request() -> anthropic.BadRequestError:
    request = httpx2.Request("POST", "https://example.invalid/v1/messages")
    response = httpx2.Response(400, request=request)
    return anthropic.BadRequestError("output_format is not supported", response=response, body=None)


class _FakeMessages:
    def __init__(self, parse_result: Any, create_result: Any) -> None:
        self.parse_result = parse_result
        self.create_result = create_result
        self.parse_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        if isinstance(self.parse_result, Exception):
            raise self.parse_result
        return self.parse_result

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        if isinstance(self.create_result, Exception):
            raise self.create_result
        return self.create_result


class _FakeClient:
    instances: ClassVar[list[_FakeClient]] = []
    parse_result: ClassVar[Any] = None
    create_result: ClassVar[Any] = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.messages = _FakeMessages(type(self).parse_result, type(self).create_result)
        type(self).instances.append(self)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    _FakeClient.instances = []
    _FakeClient.parse_result = None
    _FakeClient.create_result = None
    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    return _FakeClient


def test_key_comes_from_named_env_var_and_base_url_is_passed(
    fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GATEWAY_KEY", "test-value")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = AiReview(
        enabled=True, model="m", base_url="https://gateway.example/api/", api_key_env="GATEWAY_KEY"
    )
    ClaudeReviewer(settings)
    client = fake_client.instances[0]
    assert client.kwargs == {"api_key": "test-value", "base_url": "https://gateway.example/api"}


def test_missing_env_var_raises_a_clear_error_on_assess(
    fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISSING_KEY_VAR", raising=False)
    reviewer = ClaudeReviewer(AiReview(enabled=True, api_key_env="MISSING_KEY_VAR"))
    with pytest.raises(RuntimeError, match="MISSING_KEY_VAR"):
        reviewer.assess("L1", "[]")
    assert fake_client.instances[0].messages.parse_calls == []


def test_structured_answer_is_returned_as_is(
    fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    fake_client.parse_result = _Response(
        parsed_output=parse_assessment_text(
            '{"label": "opt_out", "confidence": "high", "evidence_message_ids": ["M2"], "explanation": "ok"}'
        )
    )
    result = ClaudeReviewer(AiReview(enabled=True)).assess("L1", "[]")
    assert result.label == AiLabel.OPT_OUT
    assert result.evidence_message_ids == ["M2"]
    assert fake_client.instances[0].messages.create_calls == []


def test_bad_request_falls_back_to_plain_json(
    fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    fake_client.parse_result = _bad_request()
    fake_client.create_result = _Response(
        content=[
            _TextBlock(
                'Sure:\n```json\n{"label": "negative", "confidence": "high", '
                '"evidence_message_ids": [], "explanation": "No."}\n```'
            )
        ]
    )
    result = ClaudeReviewer(AiReview(enabled=True)).assess("L1", "[]")
    assert result.label == AiLabel.NEGATIVE
    messages = fake_client.instances[0].messages
    assert len(messages.parse_calls) == 1 and len(messages.create_calls) == 1
    assert "JSON" in messages.create_calls[0]["system"]


def test_fallback_rejects_non_json_or_schema_violations(
    fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    fake_client.parse_result = _bad_request()
    fake_client.create_result = _Response(content=[_TextBlock("I think the client is fine.")])
    with pytest.raises(ValueError, match="JSON"):
        ClaudeReviewer(AiReview(enabled=True)).assess("L1", "[]")
    with pytest.raises(ValueError):
        parse_assessment_text('{"label": "maybe", "confidence": "high", "explanation": "x"}')
    with pytest.raises(ValueError):
        parse_assessment_text('{"label": "none", "explanation": "x"}')  # confidence missing
    with pytest.raises(ValueError):
        parse_assessment_text('{"label": "none", "confidence": "high", "explanation": "x", "extra": 1}')


def test_refusal_in_fallback_is_unclear(
    fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    fake_client.parse_result = _bad_request()
    fake_client.create_result = _Response(stop_reason="refusal")
    assert ClaudeReviewer(AiReview(enabled=True)).assess("L1", "[]").label == AiLabel.UNCLEAR


@pytest.mark.parametrize("bad", ["sk-ant-123", "my key", "", "x" * 65])
def test_config_rejects_key_like_or_malformed_env_names(bad: str) -> None:
    with pytest.raises(ValueError):
        AiReview(api_key_env=bad)


def test_config_requires_https_base_url() -> None:
    with pytest.raises(ValueError):
        AiReview(base_url="http://openrouter.ai/api")
    assert AiReview(base_url="https://openrouter.ai/api/").base_url == "https://openrouter.ai/api"
    assert AiReview().base_url is None
