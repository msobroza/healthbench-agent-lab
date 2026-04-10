"""Critique-refine prompt optimizer adapter.

Implements the PromptWizard-style mutation + critique-refine loop using only
the existing LLMClient for LLM calls. No external optimization framework
dependencies -- the mutation and refinement logic is driven by prompt
engineering over the meta-model.

The adapter is **domain-agnostic**: the mutate, critique and refine
templates plus the thinking-styles list are loaded from a YAML file at
construction time (path comes from :class:`CritiqueRefineConfig.prompt_path`).
A neutral default ships under ``prompts/prompt_optimization/`` and can be
overridden to specialise the optimizer for any vertical without touching
this module.

Registered as ``"critique_refine"`` in the optimizer registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from jinja2 import Template

from healthbench_agent.domain.llm_client import LLMClient

from ..config import CritiqueRefineConfig
from ..optimizer import OptimizationResult, PromptOptimizer, TrialRecord
from ..optimizer_registry import register_prompt_optimizer

if TYPE_CHECKING:
    from healthbench_agent.domain.dataset import HealthBenchSample

    from ..metric import EndToEndMetric


@dataclass(frozen=True)
class _CritiqueRefinePrompts:
    """Bundle of Jinja2 templates and thinking styles loaded from YAML.

    Attributes:
        mutate: Template with ``{{ thinking_style }}`` and ``{{ prompt }}``
            variables. Renders the mutation request to the meta-model.
        critique: Template with ``{{ prompt }}``. Renders the critique
            request to the meta-model.
        refine: Template with ``{{ prompt }}`` and ``{{ critique }}``.
            Renders the refinement request to the meta-model.
        thinking_styles: Directives the optimizer cycles through to
            steer mutation diversity.
    """

    mutate: Template
    critique: Template
    refine: Template
    thinking_styles: list[str]


def _load_prompts(path: str | Path) -> _CritiqueRefinePrompts:
    """Load critique-refine prompts and thinking styles from a YAML file.

    Args:
        path: Path to the YAML file. Required keys: ``mutate_template``,
            ``critique_template``, ``refine_template``, ``thinking_styles``.

    Returns:
        A populated :class:`_CritiqueRefinePrompts` bundle.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If the file is missing one of the required keys.
        ValueError: If ``thinking_styles`` is empty.
    """
    with open(path) as fh:
        data = yaml.safe_load(fh)
    thinking_styles = list(data["thinking_styles"])
    if not thinking_styles:
        raise ValueError(f"{path}: 'thinking_styles' must contain at least one entry")
    return _CritiqueRefinePrompts(
        mutate=Template(data["mutate_template"]),
        critique=Template(data["critique_template"]),
        refine=Template(data["refine_template"]),
        thinking_styles=thinking_styles,
    )


def create_llm_client(config: CritiqueRefineConfig) -> LLMClient:
    """Build an LLMClient from CritiqueRefineConfig for meta-model calls.

    Bridges the optimization config to the existing LLM client factory by
    constructing a JudgeConfig with the meta-model settings.

    Args:
        config: Critique-refine optimizer configuration containing
            meta_provider, meta_model, and API keys.

    Returns:
        An LLMClient instance configured for the meta-model.
    """
    from healthbench_agent.llm_eval.clients import create_llm_client as _create_llm_client
    from healthbench_agent.llm_eval.grading.config import JudgeConfig

    judge_config = JudgeConfig(
        provider=config.meta_provider,
        model=config.meta_model,
        temperature=0.7,
        google_api_key=config.google_api_key,
        openai_api_key=config.openai_api_key,
    )
    return _create_llm_client(judge_config)


@register_prompt_optimizer("critique_refine", CritiqueRefineConfig)
class CritiqueRefineOptimizer(PromptOptimizer):
    """Prompt optimizer using mutation and critique-refine cycles.

    Operates in two phases:

    1. **Mutation phase** -- generates ``mutation_rounds * style_variations``
       prompt variants by asking the meta-LLM to rewrite the current prompt
       incorporating different thinking styles loaded from the YAML file
       at ``config.prompt_path``.

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
        # Loading the YAML once at construction lets ``optimize()`` stay
        # I/O-free and surfaces a malformed prompt file as soon as the
        # optimizer is instantiated, not mid-run.
        self._prompts = _load_prompts(config.prompt_path)

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
        llm_client = create_llm_client(self.config)
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
                styles = self._prompts.thinking_styles
                style = styles[
                    (round_index * self.config.style_variations + style_index) % len(styles)
                ]
                mutated = self._mutate_prompt(llm_client, current_prompt, style)
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
                config=self.config.dump_safe(),
            )

        # --- Phase 2: Critique-Refine ---
        baseline_score = metric(current_prompt)

        # Find the best candidate from the mutation phase. Building a
        # ``(prompt, score)`` tuple list lets ``max`` infer ``float``
        # without a separate type-narrowing step.
        scored = [(t.prompt, t.score) for t in trial_history if t.score is not None]
        best_prompt, best_score = max(scored, key=lambda pair: pair[1])

        for _ in range(self.config.refine_iterations):
            if len(trial_history) >= max_trials:
                break
            critique = self._critique_prompt(llm_client, best_prompt)
            refined = self._refine_prompt(llm_client, best_prompt, critique)
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
            config=self.config.dump_safe(),
        )

    @staticmethod
    def _call_meta(llm_client: LLMClient, content: str) -> str:
        """Send a one-shot user message to the meta-model and return the text.

        Used by every mutation/critique/refine call so the message
        construction lives in one place.

        Args:
            llm_client: The meta-model LLM client for LLM calls.
            content: The full user prompt text to send.

        Returns:
            The stripped response text from the meta-model.
        """
        response = llm_client([{"role": "user", "content": content}])
        return response.response_text.strip()

    def _mutate_prompt(
        self,
        llm_client: LLMClient,
        prompt: str,
        thinking_style: str,
    ) -> str:
        """Generate a mutated prompt variant using a thinking style.

        Args:
            llm_client: The meta-model LLM client for LLM calls.
            prompt: The current prompt to mutate.
            thinking_style: A directive guiding the mutation approach.

        Returns:
            The mutated prompt text from the meta-model.
        """
        return self._call_meta(
            llm_client,
            self._prompts.mutate.render(thinking_style=thinking_style, prompt=prompt),
        )

    def _critique_prompt(
        self,
        llm_client: LLMClient,
        prompt: str,
    ) -> str:
        """Generate a critique of a prompt identifying weaknesses.

        Args:
            llm_client: The meta-model LLM client for LLM calls.
            prompt: The prompt to critique.

        Returns:
            A text critique identifying areas for improvement.
        """
        return self._call_meta(llm_client, self._prompts.critique.render(prompt=prompt))

    def _refine_prompt(
        self,
        llm_client: LLMClient,
        prompt: str,
        critique: str,
    ) -> str:
        """Refine a prompt based on a critique.

        Args:
            llm_client: The meta-model LLM client for LLM calls.
            prompt: The prompt to refine.
            critique: The critique to address in the refinement.

        Returns:
            The refined prompt text.
        """
        return self._call_meta(
            llm_client,
            self._prompts.refine.render(prompt=prompt, critique=critique),
        )
