from __future__ import annotations

from silent_leads.config import Template, Templates
from silent_leads.models import Decision, TemplateStatus
from silent_leads.templates import prepare_draft
from tests.test_timeline import _lead

MISSING = Templates()
TEST = Templates(
    with_quote=Template(
        text="[TEST] Hi, following up on the quote we sent.", status=TemplateStatus.TEST_TEMPLATE
    ),
    without_quote=Template(
        text="[TEST] Hi, still interested in a quote?", status=TemplateStatus.TEST_TEMPLATE
    ),
)


def _quoted(has_quote: bool | None):  # type: ignore[no-untyped-def]
    return _lead().model_copy(update={"has_quote": has_quote})


def test_missing_templates_give_empty_text_and_template_missing_for_both_branches() -> None:
    for has_quote in (True, False):
        assert prepare_draft(_quoted(has_quote), Decision.CONTACT_CANDIDATE, MISSING) == (
            "",
            TemplateStatus.TEMPLATE_MISSING,
        )


def test_quote_branch_selects_matching_test_template() -> None:
    text, status = prepare_draft(_quoted(True), Decision.CONTACT_CANDIDATE, TEST)
    assert status == TemplateStatus.TEST_TEMPLATE
    assert "quote we sent" in text
    text, status = prepare_draft(_quoted(False), Decision.CONTACT_CANDIDATE, TEST)
    assert status == TemplateStatus.TEST_TEMPLATE
    assert "still interested" in text


def test_no_text_for_non_candidates() -> None:
    for decision in (Decision.SKIP_NOT_DUE, Decision.DO_NOT_CONTACT, Decision.MANUAL_REVIEW):
        assert prepare_draft(_quoted(True), decision, TEST) == ("", TemplateStatus.NOT_APPLICABLE)


def test_blank_text_with_non_missing_status_is_still_missing() -> None:
    blank = Templates(with_quote=Template(text="   ", status=TemplateStatus.TEST_TEMPLATE))
    assert prepare_draft(_quoted(True), Decision.CONTACT_CANDIDATE, blank) == (
        "",
        TemplateStatus.TEMPLATE_MISSING,
    )


def test_unknown_quote_flag_never_gets_a_text() -> None:
    assert prepare_draft(_quoted(None), Decision.CONTACT_CANDIDATE, TEST) == (
        "",
        TemplateStatus.TEMPLATE_MISSING,
    )
