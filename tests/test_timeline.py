from __future__ import annotations

from datetime import datetime

from silent_leads.models import ContentState, Direction, Lead, Message, Role
from silent_leads.timeline import Speaker, build_timelines, split_utterances


def _lead(case_id: str = "L1", silent: str | None = "2026-10-20 10:00:00") -> Lead:
    return Lead(
        case_id=case_id,
        team_alias=None,
        sales_alias=None,
        status="junk silent",
        created_at_crm=None,
        silent_since_crm=datetime.fromisoformat(silent) if silent else None,
        has_quote=False,
        quote_at_crm=None,
        last_inbound_at_crm=None,
        last_outbound_at_crm=None,
        events_count=None,
        text_events=None,
        hidden_events=None,
        calls_without_text=None,
        client_timezone=None,
    )


def _msg(
    message_id: str,
    occurred: str | None,
    *,
    case_id: str = "L1",
    direction: Direction = Direction.OUTBOUND,
    text: str | None = "hi",
    event_type: str = "SMS",
) -> Message:
    return Message(
        case_id=case_id,
        message_id=message_id,
        occurred_at_crm=datetime.fromisoformat(occurred) if occurred else None,
        event_type=event_type,
        direction=direction,
        role=Role.CLIENT if direction == Direction.INBOUND else Role.SALES,
        call_result=None,
        call_seconds=None,
        content_state=ContentState.TEXT_AVAILABLE if text else ContentState.CALL_WITHOUT_TEXT,
        text=text,
    )


def test_timeline_is_chronological_with_undated_last_and_after_silent_detected() -> None:
    timelines = build_timelines(
        [_lead()],
        [
            _msg("M3", "2026-10-21 09:00:00"),
            _msg("M1", "2026-10-01 09:00:00"),
            _msg("M0", None),
            _msg("M2", "2026-10-01 09:00:00"),
        ],
    )
    tl = timelines["L1"]
    assert [m.message_id for m in tl.messages] == ["M1", "M2", "M3", "M0"]
    assert [m.message_id for m in tl.after_silent] == ["M3"]
    assert [m.message_id for m in tl.undated] == ["M0"]


def test_message_at_exact_silent_timestamp_is_not_after() -> None:
    tl = build_timelines([_lead()], [_msg("M1", "2026-10-20 10:00:00")])["L1"]
    assert tl.after_silent == ()


def test_lead_without_silent_date_has_no_after_silent() -> None:
    tl = build_timelines([_lead(silent=None)], [_msg("M1", "2026-12-01 10:00:00")])["L1"]
    assert tl.after_silent == ()


def test_messages_of_other_cases_never_leak_in() -> None:
    timelines = build_timelines([_lead("L1"), _lead("L2")], [_msg("A", "2026-10-01 09:00:00", case_id="L2")])
    assert timelines["L1"].messages == ()
    assert [m.message_id for m in timelines["L2"].messages] == ["A"]


def test_sms_text_is_attributed_by_direction() -> None:
    inbound = split_utterances(
        _msg("M1", "2026-10-01 09:00:00", direction=Direction.INBOUND, text="please stop")
    )
    outbound = split_utterances(_msg("M2", "2026-10-01 09:00:00", text="hello"))
    assert [(u.speaker, u.text) for u in inbound] == [(Speaker.CLIENT, "please stop")]
    assert [(u.speaker, u.text) for u in outbound] == [(Speaker.SALES, "hello")]


def test_transcript_lines_use_speaker_tags_not_direction() -> None:
    transcript = (
        "Jane (C): Hello?\n"
        "[СПІВРОБІТНИК] (S): Hi, do you still need the service?\n"
        "Jane (C): No, please don't call me again.\n"
        "(unclear audio)\n"
    )
    utterances = split_utterances(
        _msg("M1", "2026-10-01 09:00:00", direction=Direction.OUTBOUND, text=transcript, event_type="CALL")
    )
    assert [u.speaker for u in utterances] == [Speaker.CLIENT, Speaker.SALES, Speaker.CLIENT, Speaker.UNKNOWN]
    assert utterances[2].text == "No, please don't call me again."
    assert utterances[3].text == "(unclear audio)"
    assert {u.message_id for u in utterances} == {"M1"}


def test_hidden_or_empty_messages_yield_no_utterances() -> None:
    assert split_utterances(_msg("M1", "2026-10-01 09:00:00", text=None)) == []
    assert split_utterances(_msg("M2", "2026-10-01 09:00:00", text="   ")) == []


def test_call_transcript_without_speaker_tags_is_unknown_as_a_whole() -> None:
    call = _msg(
        "M1",
        "2026-10-02 09:00:00",
        direction=Direction.OUTBOUND,
        event_type="CALL",
        text="Hi, this is the team about your quote.\nDon't call me again, I'm not interested.",
    )
    utterances = split_utterances(call)
    assert [u.speaker for u in utterances] == [Speaker.UNKNOWN]  # never silently "sales"


def test_speaker_tags_are_case_insensitive() -> None:
    call = _msg(
        "M1",
        "2026-10-02 09:00:00",
        event_type="CALL",
        text="[СПІВРОБІТНИК] (s): Hello?\nKate (c): Please stop calling me",
    )
    assert [u.speaker for u in split_utterances(call)] == [Speaker.SALES, Speaker.CLIENT]


def test_tags_inside_an_sms_do_not_reattribute_the_client() -> None:
    sms = _msg("M1", "2026-10-02 09:00:00", direction=Direction.INBOUND, event_type="SMS", text="(S): STOP")
    [utterance] = split_utterances(sms)
    assert utterance.speaker == Speaker.CLIENT
    assert utterance.text == "(S): STOP"
