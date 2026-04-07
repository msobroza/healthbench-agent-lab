"""Tests for extract_ideal_completion_text schema variants."""

from __future__ import annotations

from healthbench_agent.dataset.extraction import extract_ideal_completion_text


def test_extract_returns_none_when_data_is_none():
    assert extract_ideal_completion_text(None) is None


def test_extract_returns_none_for_empty_dict():
    assert extract_ideal_completion_text({}) is None


def test_extract_handles_string_ideal_completion():
    data = {"ideal_completion": "the ideal answer"}
    assert extract_ideal_completion_text(data) == "the ideal answer"


def test_extract_handles_list_of_role_content_dicts():
    data = {"ideal_completion": [{"role": "assistant", "content": "ans"}]}
    assert extract_ideal_completion_text(data) == "ans"


def test_extract_handles_plural_ideal_completions():
    data = {"ideal_completions": [{"role": "assistant", "content": "ans2"}]}
    assert extract_ideal_completion_text(data) == "ans2"


def test_extract_returns_none_when_no_known_keys_match():
    assert extract_ideal_completion_text({"unknown": "x"}) is None


def test_extract_returns_none_when_list_has_no_assistant_turn():
    data = {"ideal_completion": [{"role": "user", "content": "?"}]}
    assert extract_ideal_completion_text(data) is None
