"""Tests for healthbench_agent.llm_eval.grader — grading logic.

Tests template rendering, response parsing, conversation formatting,
LLMJudgeGrader, create_judge, and grade_sample with mocked LLM clients.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.llm_client import LLMResponse
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.llm_eval.grader import (
    GRADER_TEMPLATE,
    LLMJudgeGrader,
    create_judge,
    format_conversation,
    grade_sample,
    load_grader_prompt,
    parse_grading_response,
)

# ---------------------------------------------------------------------------
# GRADER_TEMPLATE tests
# ---------------------------------------------------------------------------


class TestGraderTemplate:
    """Tests for the GRADER_TEMPLATE constant."""

    def test_template_is_nonempty_string(self):
        assert isinstance(GRADER_TEMPLATE, str)
        assert len(GRADER_TEMPLATE) > 100

    def test_template_contains_conversation_placeholder(self):
        assert "<<conversation>>" in GRADER_TEMPLATE

    def test_template_contains_rubric_item_placeholder(self):
        assert "<<rubric_item>>" in GRADER_TEMPLATE

    def test_template_mentions_criteria_met(self):
        assert "criteria_met" in GRADER_TEMPLATE

    def test_template_mentions_explanation(self):
        assert "explanation" in GRADER_TEMPLATE


# ---------------------------------------------------------------------------
# load_grader_prompt tests
# ---------------------------------------------------------------------------


class TestLoadGraderPrompt:
    """Tests for load_grader_prompt()."""

    def test_load_from_yaml_returns_template_version_hash(self):
        prompt_path = (
            Path(__file__).resolve().parents[2] / "prompts" / "llm_grader" / "v1_llm_grader.yaml"
        )
        template, version, sha256 = load_grader_prompt(prompt_path)
        assert version == "1.0.0"
        assert len(sha256) == 64  # SHA-256 hex digest
        rendered = template.render(conversation="test convo", rubric_item="test item")
        assert "test convo" in rendered
        assert "test item" in rendered

    def test_hash_is_deterministic(self):
        prompt_path = (
            Path(__file__).resolve().parents[2] / "prompts" / "llm_grader" / "v1_llm_grader.yaml"
        )
        _, _, hash1 = load_grader_prompt(prompt_path)
        _, _, hash2 = load_grader_prompt(prompt_path)
        assert hash1 == hash2


# ---------------------------------------------------------------------------
# format_conversation tests
# ---------------------------------------------------------------------------


class TestFormatConversation:
    """Tests for format_conversation()."""

    def test_single_turn(self):
        messages: MessageList = [{"role": "user", "content": "Hello"}]
        result = format_conversation(messages)
        assert result == "user: Hello"

    def test_multi_turn(self):
        messages: MessageList = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = format_conversation(messages)
        assert "user: Hi" in result
        assert "assistant: Hello!" in result
        assert "user: How are you?" in result
        # Turns separated by blank lines
        assert "\n\n" in result

    def test_empty_conversation(self):
        result = format_conversation([])
        assert result == ""

    def test_preserves_content(self):
        messages: MessageList = [{"role": "user", "content": "Special chars: <>\"'&"}]
        result = format_conversation(messages)
        assert "Special chars: <>\"'&" in result


# ---------------------------------------------------------------------------
# parse_grading_response tests
# ---------------------------------------------------------------------------


class TestParseGradingResponse:
    """Tests for parse_grading_response()."""

    def test_parse_bare_json(self):
        response = '{"explanation": "Good response", "criteria_met": true}'
        result = parse_grading_response(response)
        assert result["criteria_met"] is True
        assert result["explanation"] == "Good response"

    def test_parse_json_in_markdown_fence(self):
        response = '```json\n{"explanation": "Bad", "criteria_met": false}\n```'
        result = parse_grading_response(response)
        assert result["criteria_met"] is False

    def test_parse_json_in_plain_fence(self):
        response = '```\n{"explanation": "Ok", "criteria_met": true}\n```'
        result = parse_grading_response(response)
        assert result["criteria_met"] is True

    def test_criteria_met_false_coercion(self):
        response = '{"explanation": "No", "criteria_met": false}'
        result = parse_grading_response(response)
        assert result["criteria_met"] is False

    def test_missing_explanation_defaults_to_empty(self):
        response = '{"criteria_met": true}'
        result = parse_grading_response(response)
        assert result["explanation"] == ""
        assert result["criteria_met"] is True

    def test_missing_criteria_met_raises_value_error(self):
        response = '{"explanation": "No criteria_met field"}'
        with pytest.raises(ValueError, match="missing 'criteria_met'"):
            parse_grading_response(response)

    def test_invalid_json_raises_value_error(self):
        response = "This is not JSON"
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_grading_response(response)

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_grading_response("")

    def test_whitespace_handling(self):
        response = '  \n  {"explanation": "x", "criteria_met": true}  \n  '
        result = parse_grading_response(response)
        assert result["criteria_met"] is True


# ---------------------------------------------------------------------------
# grade_sample tests (mocked LLM client)
# ---------------------------------------------------------------------------


def _make_mock_llm_client(responses: list[str]) -> MagicMock:
    """Create a mock LLM client that returns predetermined responses."""
    llm_client = MagicMock()
    side_effects = [
        LLMResponse(
            response_text=r,
            actual_queried_message_list=[],
            response_metadata={},
        )
        for r in responses
    ]
    llm_client.side_effect = side_effects
    return llm_client


class TestGradeSample:
    """Tests for grade_sample() with mocked LLM clients."""

    def test_single_rubric_item_met(self):
        llm_client = _make_mock_llm_client(['{"explanation": "Correct", "criteria_met": true}'])
        rubric_items = [RubricItem("States the diagnosis", 5.0, ["axis:accuracy"])]
        conversation: MessageList = [
            {"role": "user", "content": "What's wrong?"},
            {"role": "assistant", "content": "You have condition X."},
        ]
        verdicts = grade_sample(llm_client, conversation, rubric_items)
        assert len(verdicts) == 1
        assert verdicts[0].criteria_met is True
        assert verdicts[0].criterion == "States the diagnosis"

    def test_single_rubric_item_not_met(self):
        llm_client = _make_mock_llm_client(
            ['{"explanation": "Missing info", "criteria_met": false}']
        )
        rubric_items = [RubricItem("Mentions side effects", 3.0)]
        conversation: MessageList = [{"role": "user", "content": "Tell me about X."}]
        verdicts = grade_sample(llm_client, conversation, rubric_items)
        assert len(verdicts) == 1
        assert verdicts[0].criteria_met is False

    def test_multiple_rubric_items(self):
        llm_client = _make_mock_llm_client(
            [
                '{"explanation": "Good", "criteria_met": true}',
                '{"explanation": "Bad", "criteria_met": false}',
                '{"explanation": "OK", "criteria_met": true}',
            ]
        )
        rubric_items = [
            RubricItem("Item A", 5.0),
            RubricItem("Item B", 3.0),
            RubricItem("Item C", -2.0),
        ]
        conversation: MessageList = [{"role": "user", "content": "Q"}]
        verdicts = grade_sample(llm_client, conversation, rubric_items)
        assert len(verdicts) == 3
        assert verdicts[0].criteria_met is True
        assert verdicts[1].criteria_met is False
        assert verdicts[2].criteria_met is True

    def test_malformed_response_defaults_to_not_met(self):
        llm_client = _make_mock_llm_client(["This is not valid JSON at all"])
        rubric_items = [RubricItem("Some criterion", 5.0)]
        conversation: MessageList = [{"role": "user", "content": "Q"}]
        verdicts = grade_sample(llm_client, conversation, rubric_items)
        assert len(verdicts) == 1
        assert verdicts[0].criteria_met is False
        assert "Failed to parse" in verdicts[0].explanation

    def test_llm_client_called_once_per_rubric_item(self):
        llm_client = _make_mock_llm_client(
            [
                '{"explanation": "A", "criteria_met": true}',
                '{"explanation": "B", "criteria_met": true}',
            ]
        )
        rubric_items = [RubricItem("X", 1.0), RubricItem("Y", 2.0)]
        conversation: MessageList = [{"role": "user", "content": "Q"}]
        grade_sample(llm_client, conversation, rubric_items)
        assert llm_client.call_count == 2

    def test_empty_rubric_returns_empty_verdicts(self):
        llm_client = _make_mock_llm_client([])
        verdicts = grade_sample(llm_client, [], [])
        assert verdicts == []


# ---------------------------------------------------------------------------
# LLMJudgeGrader tests
# ---------------------------------------------------------------------------


class TestLLMJudgeGrader:
    """Tests for LLMJudgeGrader concrete class."""

    def test_is_judge_grader_subclass(self):
        llm_client = _make_mock_llm_client([])
        judge = LLMJudgeGrader(llm_client=llm_client)
        assert isinstance(judge, JudgeGrader)

    def test_grade_single_item_met(self):
        llm_client = _make_mock_llm_client(['{"explanation": "Good", "criteria_met": true}'])
        judge = LLMJudgeGrader(llm_client=llm_client)
        rubric_items = [RubricItem("Correct diagnosis", 5.0)]
        conversation: MessageList = [
            {"role": "user", "content": "What do I have?"},
            {"role": "assistant", "content": "You have X."},
        ]
        verdicts = judge.grade(conversation, rubric_items)
        assert len(verdicts) == 1
        assert verdicts[0].criteria_met is True
        assert verdicts[0].criterion == "Correct diagnosis"

    def test_grade_multiple_items(self):
        llm_client = _make_mock_llm_client(
            [
                '{"explanation": "A", "criteria_met": true}',
                '{"explanation": "B", "criteria_met": false}',
            ]
        )
        judge = LLMJudgeGrader(llm_client=llm_client)
        rubric_items = [
            RubricItem("Item A", 5.0),
            RubricItem("Item B", 3.0),
        ]
        conversation: MessageList = [{"role": "user", "content": "Q"}]
        verdicts = judge.grade(conversation, rubric_items)
        assert len(verdicts) == 2
        assert verdicts[0].criteria_met is True
        assert verdicts[1].criteria_met is False

    def test_malformed_response_defaults_to_not_met(self):
        llm_client = _make_mock_llm_client(["not json"])
        judge = LLMJudgeGrader(llm_client=llm_client)
        verdicts = judge.grade(
            [{"role": "user", "content": "Q"}],
            [RubricItem("X", 5.0)],
        )
        assert verdicts[0].criteria_met is False
        assert "Failed to parse" in verdicts[0].explanation

    def test_llm_client_called_once_per_rubric_item(self):
        llm_client = _make_mock_llm_client(
            [
                '{"explanation": "A", "criteria_met": true}',
                '{"explanation": "B", "criteria_met": true}',
            ]
        )
        judge = LLMJudgeGrader(llm_client=llm_client)
        judge.grade(
            [{"role": "user", "content": "Q"}],
            [RubricItem("X", 1.0), RubricItem("Y", 2.0)],
        )
        assert llm_client.call_count == 2

    def test_empty_rubric_returns_empty(self):
        llm_client = _make_mock_llm_client([])
        judge = LLMJudgeGrader(llm_client=llm_client)
        assert judge.grade([], []) == []


# ---------------------------------------------------------------------------
# create_judge factory tests
# ---------------------------------------------------------------------------


class TestCreateJudge:
    """Tests for the create_judge() factory function."""

    def test_returns_llm_judge_grader(self):
        from healthbench_agent.llm_eval.config_grader import JudgeConfig

        config = JudgeConfig(prompt_path="prompts/llm_grader/v1_llm_grader.yaml")
        judge = create_judge(config)
        assert isinstance(judge, LLMJudgeGrader)
        assert isinstance(judge, JudgeGrader)

    def test_template_loaded_from_config_path(self):
        from healthbench_agent.llm_eval.config_grader import JudgeConfig

        config = JudgeConfig(prompt_path="prompts/llm_grader/v1_llm_grader.yaml")
        judge = create_judge(config)
        # Template should render successfully with <<>> placeholders
        rendered = judge.template.render(conversation="test convo", rubric_item="test item")
        assert "test convo" in rendered
        assert "test item" in rendered

    def test_grader_max_workers_passed_through(self):
        from healthbench_agent.llm_eval.config_grader import JudgeConfig

        config = JudgeConfig(
            prompt_path="prompts/llm_grader/v1_llm_grader.yaml",
            grader_max_workers=4,
        )
        judge = create_judge(config)
        assert judge.max_workers == 4
