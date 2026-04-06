"""Tests for the CritiqueRefineOptimizer adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.domain.sampler import SamplerResponse
from healthbench_agent.prompt_optimization.adapters.critique_refine_adapter import (
    THINKING_STYLES,
    CritiqueRefineOptimizer,
)
from healthbench_agent.prompt_optimization.config import CritiqueRefineConfig


def _make_sample() -> HealthBenchSample:
    return HealthBenchSample(
        prompt_id="test-1",
        prompt=[{"role": "user", "content": "What is aspirin?"}],
        rubrics=[RubricItem(criterion="Mentions pain relief", points=1.0, tags=["accuracy"])],
        example_tags=["theme:general"],
    )


class TestThinkingStyles:
    def test_styles_is_nonempty_list(self):
        assert isinstance(THINKING_STYLES, list)
        assert len(THINKING_STYLES) > 0

    def test_styles_are_strings(self):
        for style in THINKING_STYLES:
            assert isinstance(style, str)
            assert len(style) > 0


class TestCritiqueRefineOptimizerMutationOnly:
    def test_mutation_only_returns_result(self):
        config = CritiqueRefineConfig(
            mutation_rounds=2,
            style_variations=2,
            refine_iterations=1,
        )
        optimizer = CritiqueRefineOptimizer(config)

        mock_sampler = MagicMock()
        mock_sampler.return_value = SamplerResponse(
            response_text="Improved prompt: You are an expert health advisor.",
            actual_queried_message_list=[],
            response_metadata={},
        )

        with patch(
            "healthbench_agent.prompt_optimization.adapters.critique_refine_adapter.create_sampler",
            return_value=mock_sampler,
        ):
            result = optimizer.optimize(
                current_prompt="You are a health assistant.",
                samples=None,
                metric=None,
            )

        assert result.optimizer_name == "critique_refine"
        assert result.num_trials > 0
        assert result.optimized_prompt != ""
        assert result.baseline_score == 0.0
        assert result.optimized_score == 0.0

    def test_mutation_only_trial_scores_are_none(self):
        config = CritiqueRefineConfig(
            mutation_rounds=1,
            style_variations=1,
            refine_iterations=1,
        )
        optimizer = CritiqueRefineOptimizer(config)

        mock_sampler = MagicMock()
        mock_sampler.return_value = SamplerResponse(
            response_text="Mutated prompt.",
            actual_queried_message_list=[],
            response_metadata={},
        )

        with patch(
            "healthbench_agent.prompt_optimization.adapters.critique_refine_adapter.create_sampler",
            return_value=mock_sampler,
        ):
            result = optimizer.optimize("Original.", samples=None, metric=None)

        for trial in result.trial_history:
            assert trial.score is None


class TestCritiqueRefineOptimizerWithMetric:
    def test_with_metric_returns_scored_result(self):
        config = CritiqueRefineConfig(
            mutation_rounds=1,
            style_variations=2,
            refine_iterations=1,
        )
        optimizer = CritiqueRefineOptimizer(config)

        mock_sampler = MagicMock()
        call_count = 0

        def sampler_side_effect(message_list):
            nonlocal call_count
            call_count += 1
            return SamplerResponse(
                response_text=f"Variant {call_count}: Be a thorough health expert.",
                actual_queried_message_list=message_list,
                response_metadata={},
            )

        mock_sampler.side_effect = sampler_side_effect

        mock_metric = MagicMock()
        # Provide enough values: baseline(1) + mutations(2) + critique(1) + refine(1)
        # The exact count depends on implementation; provide extra for safety
        mock_metric.side_effect = [0.5, 0.7, 0.8, 0.9, 0.85, 0.95, 0.92, 0.88]

        sample = _make_sample()

        with patch(
            "healthbench_agent.prompt_optimization.adapters.critique_refine_adapter.create_sampler",
            return_value=mock_sampler,
        ):
            result = optimizer.optimize(
                current_prompt="You are a health assistant.",
                samples=[sample],
                metric=mock_metric,
            )

        assert result.optimizer_name == "critique_refine"
        assert len(result.trial_history) > 0

    def test_config_stored_in_result(self):
        config = CritiqueRefineConfig(
            mutation_rounds=1,
            style_variations=1,
            refine_iterations=1,
        )
        optimizer = CritiqueRefineOptimizer(config)

        mock_sampler = MagicMock()
        mock_sampler.return_value = SamplerResponse(
            response_text="Mutated.",
            actual_queried_message_list=[],
            response_metadata={},
        )

        with patch(
            "healthbench_agent.prompt_optimization.adapters.critique_refine_adapter.create_sampler",
            return_value=mock_sampler,
        ):
            result = optimizer.optimize("prompt", samples=None, metric=None)

        assert result.config["optimizer"] == "critique_refine"
        assert result.config["mutation_rounds"] == 1


class TestCritiqueRefineOptimizerBudget:
    """Verifies max_trials early-stop in both mutation and refine phases."""

    def _build_sampler(self) -> MagicMock:
        """Sampler that returns a unique response per call so candidates differ."""
        mock_sampler = MagicMock()
        counter = {"n": 0}

        def side_effect(message_list):
            counter["n"] += 1
            return SamplerResponse(
                response_text=f"Variant {counter['n']}.",
                actual_queried_message_list=message_list,
                response_metadata={},
            )

        mock_sampler.side_effect = side_effect
        return mock_sampler

    def test_max_trials_caps_mutation_phase(self):
        # 5 rounds * 5 styles = 25 mutations, but max_trials=3 caps it.
        config = CritiqueRefineConfig(
            mutation_rounds=5,
            style_variations=5,
            refine_iterations=1,
            max_trials=3,
        )
        optimizer = CritiqueRefineOptimizer(config)
        mock_sampler = self._build_sampler()

        with patch(
            "healthbench_agent.prompt_optimization.adapters.critique_refine_adapter.create_sampler",
            return_value=mock_sampler,
        ):
            result = optimizer.optimize(
                current_prompt="Baseline.",
                samples=None,
                metric=None,  # mutation-only mode
            )

        # Mutation loop must stop after 3 candidates regardless of round/style budget.
        assert result.num_trials == 3
        assert len(result.trial_history) == 3

    def test_max_trials_caps_refine_phase(self):
        # 1 mutation round * 1 style = 1 mutation; refine_iterations=10 but
        # max_trials=2 means only one refine iteration runs (1 mutation + 1 refine).
        config = CritiqueRefineConfig(
            mutation_rounds=1,
            style_variations=1,
            refine_iterations=10,
            max_trials=2,
        )
        optimizer = CritiqueRefineOptimizer(config)
        mock_sampler = self._build_sampler()

        # Mutation score, baseline score, then 1 refine score = 3 metric calls.
        end_metric = MagicMock(side_effect=[0.6, 0.5, 0.7])
        sample = _make_sample()

        with patch(
            "healthbench_agent.prompt_optimization.adapters.critique_refine_adapter.create_sampler",
            return_value=mock_sampler,
        ):
            result = optimizer.optimize(
                current_prompt="Baseline.",
                samples=[sample],
                metric=end_metric,
            )

        # 1 mutation trial + 1 refine trial = 2 trials, capped by max_trials.
        assert result.num_trials == 2
        # Refine improved over mutation (0.7 > 0.6) so it wins.
        assert result.optimized_score == 0.7
        assert result.baseline_score == 0.5
