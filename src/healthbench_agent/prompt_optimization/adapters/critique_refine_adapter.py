"""Critique-refine prompt optimizer adapter.

Implements the PromptWizard-style mutation + critique-refine loop using only
the existing SamplerBase for LLM calls. No external optimization framework
dependencies -- the mutation and refinement logic is driven by prompt
engineering over the meta-model.

Registered as ``"critique_refine"`` in the optimizer registry.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from healthbench_agent.domain.sampler import SamplerBase

from ..config import CritiqueRefineConfig
from ..optimizer import OptimizationResult, PromptOptimizer, TrialRecord
from ..optimizer_registry import register_prompt_optimizer

if TYPE_CHECKING:
    from healthbench_agent.domain.dataset import HealthBenchSample

    from ..metric import EndToEndMetric

THINKING_STYLES: list[str] = [
    "Think step-by-step about what makes a prompt effective for medical conversations.",
    "Consider the perspective of a patient who is anxious and seeking reassurance.",
    "Focus on clinical accuracy and evidence-based recommendations.",
    "Emphasize empathy, active listening, and patient-centered communication.",
    "Think about safety-critical scenarios where incorrect advice could cause harm.",
    "Consider how to handle uncertainty and when to recommend professional consultation.",
    "Focus on clarity and readability for patients with varying health literacy.",
    "Think about multi-turn conversations and maintaining context across exchanges.",
    "Consider cultural sensitivity and inclusive language in health communication.",
    "Focus on actionable advice that patients can follow without medical training.",
]


def create_sampler(config: CritiqueRefineConfig) -> SamplerBase:
    """Build a SamplerBase from CritiqueRefineConfig for meta-model calls.

    Bridges the optimization config to the existing sampler factory by
    constructing a JudgeConfig with the meta-model settings.

    Args:
        config: Critique-refine optimizer configuration containing
            meta_provider, meta_model, and API keys.

    Returns:
        A SamplerBase instance configured for the meta-model.
    """
    from healthbench_agent.llm_eval.config_grader import JudgeConfig
    from healthbench_agent.llm_eval.samplers import create_sampler as _create_llm_sampler

    google_key = config.google_api_key.get_secret_value() if config.google_api_key else None
    openai_key = config.openai_api_key.get_secret_value() if config.openai_api_key else None

    judge_config = JudgeConfig(
        provider=config.meta_provider,
        model=config.meta_model,
        temperature=0.7,
        google_api_key=google_key,
        openai_api_key=openai_key,
    )
    return _create_llm_sampler(judge_config)


@register_prompt_optimizer("critique_refine", CritiqueRefineConfig)
class CritiqueRefineOptimizer(PromptOptimizer):
    """Prompt optimizer using mutation and critique-refine cycles.

    Operates in two phases:

    1. **Mutation phase** -- generates ``mutation_rounds * style_variations``
       prompt variants by asking the meta-LLM to rewrite the current prompt
       incorporating different thinking styles from ``THINKING_STYLES``.

    2. **Critique-refine phase** (when metric is provided) -- scores all
       candidates, critiques the best one, and refines it. Repeats for
       ``refine_iterations`` cycles.

    When no metric is provided (mutation-only mode), returns the first
    candidate with all scores set to None.

    Attributes:
        config: The critique-refine optimizer configuration.
    """

    def __init__(self, config: CritiqueRefineConfig) -> None:
        self.config = config

    def optimize(
        self,
        current_prompt: str,
        samples: list[HealthBenchSample] | None,
        metric: EndToEndMetric | None,
    ) -> OptimizationResult:
        """Optimize a prompt via mutation and optional critique-refine.

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset. Required when metric is provided.
            metric: Callable that scores a prompt end-to-end. When None,
                runs in mutation-only mode (no scoring).

        Returns:
            OptimizationResult with the best prompt and trial history.
        """
        sampler = create_sampler(self.config)
        trial_history: list[TrialRecord] = []
        candidates: list[str] = []
        max_trials = self.config.max_trials

        # --- Phase 1: Mutation ---
        for round_index in range(self.config.mutation_rounds):
            if len(trial_history) >= max_trials:
                break
            for style_index in range(self.config.style_variations):
                if len(trial_history) >= max_trials:
                    break
                style = THINKING_STYLES[
                    (round_index * self.config.style_variations + style_index)
                    % len(THINKING_STYLES)
                ]
                mutated = self._mutate_prompt(sampler, current_prompt, style)
                candidates.append(mutated)

                score = None
                if metric is not None:
                    score = metric(mutated)

                trial_history.append(
                    TrialRecord(
                        trial_id=len(trial_history) + 1,
                        prompt=mutated,
                        score=score,
                        timestamp=datetime.now(tz=UTC).isoformat(),
                    )
                )

        # --- Mutation-only mode: return first candidate ---
        if metric is None:
            return OptimizationResult(
                optimized_prompt=candidates[0] if candidates else current_prompt,
                baseline_score=0.0,
                optimized_score=0.0,
                improvement=0.0,
                num_trials=len(trial_history),
                trial_history=trial_history,
                optimizer_name="critique_refine",
                config=self._serialized_config(),
            )

        # --- Phase 2: Critique-Refine ---
        baseline_score = metric(current_prompt)

        # Find the best candidate from mutation phase. The filter
        # guarantees every score is non-None, so we can safely cast.
        scored_trials = [t for t in trial_history if t.score is not None]
        best_trial = max(scored_trials, key=lambda t: t.score or 0.0)
        best_prompt = best_trial.prompt
        assert best_trial.score is not None  # narrowed by filter above
        best_score: float = best_trial.score

        for _ in range(self.config.refine_iterations):
            if len(trial_history) >= max_trials:
                break
            critique = self._critique_prompt(sampler, best_prompt)
            refined = self._refine_prompt(sampler, best_prompt, critique)
            refined_score = metric(refined)

            trial_history.append(
                TrialRecord(
                    trial_id=len(trial_history) + 1,
                    prompt=refined,
                    score=refined_score,
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            )

            if refined_score > best_score:
                best_prompt = refined
                best_score = refined_score

        return OptimizationResult(
            optimized_prompt=best_prompt,
            baseline_score=baseline_score,
            optimized_score=best_score,
            improvement=best_score - baseline_score,
            num_trials=len(trial_history),
            trial_history=trial_history,
            optimizer_name="critique_refine",
            config=self._serialized_config(),
        )

    def _mutate_prompt(
        self,
        sampler: SamplerBase,
        prompt: str,
        thinking_style: str,
    ) -> str:
        """Generate a mutated prompt variant using a thinking style.

        Args:
            sampler: The meta-model sampler for LLM calls.
            prompt: The current prompt to mutate.
            thinking_style: A directive guiding the mutation approach.

        Returns:
            The mutated prompt text from the meta-model.
        """
        message_list = [
            {
                "role": "user",
                "content": (
                    f"You are an expert prompt engineer for health AI systems.\n\n"
                    f"Thinking approach: {thinking_style}\n\n"
                    f"Rewrite and improve the following system prompt for a health "
                    f"assistant. Keep the core intent but make it more effective.\n\n"
                    f"Current prompt:\n{prompt}\n\n"
                    f"Respond with ONLY the improved prompt text, nothing else."
                ),
            }
        ]
        response = sampler(message_list)
        return response.response_text.strip()

    def _critique_prompt(
        self,
        sampler: SamplerBase,
        prompt: str,
    ) -> str:
        """Generate a critique of a prompt identifying weaknesses.

        Args:
            sampler: The meta-model sampler for LLM calls.
            prompt: The prompt to critique.

        Returns:
            A text critique identifying areas for improvement.
        """
        message_list = [
            {
                "role": "user",
                "content": (
                    f"You are an expert prompt engineer evaluating health AI prompts.\n\n"
                    f"Critically analyze the following system prompt. Identify specific "
                    f"weaknesses, missing elements, and areas where it could be more "
                    f"effective for a health assistant.\n\n"
                    f"Prompt to critique:\n{prompt}\n\n"
                    f"Provide a detailed critique with specific, actionable suggestions."
                ),
            }
        ]
        response = sampler(message_list)
        return response.response_text.strip()

    def _refine_prompt(
        self,
        sampler: SamplerBase,
        prompt: str,
        critique: str,
    ) -> str:
        """Refine a prompt based on a critique.

        Args:
            sampler: The meta-model sampler for LLM calls.
            prompt: The prompt to refine.
            critique: The critique to address in the refinement.

        Returns:
            The refined prompt text.
        """
        message_list = [
            {
                "role": "user",
                "content": (
                    f"You are an expert prompt engineer for health AI systems.\n\n"
                    f"Refine the following system prompt based on the critique below. "
                    f"Address each weakness while preserving the prompt's strengths.\n\n"
                    f"Current prompt:\n{prompt}\n\n"
                    f"Critique:\n{critique}\n\n"
                    f"Respond with ONLY the refined prompt text, nothing else."
                ),
            }
        ]
        response = sampler(message_list)
        return response.response_text.strip()

    def _serialized_config(self) -> dict[str, Any]:
        """Serialize config for the result, excluding secret keys.

        Returns:
            Dict of config fields with API keys excluded.
        """
        return self.config.model_dump(exclude={"google_api_key", "openai_api_key"})
