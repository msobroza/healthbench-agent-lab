"""TextGrad prompt optimizer adapter.

Wraps text-gradient descent for iterative prompt refinement. Uses the
``textgrad`` library to compute text-based gradients and update the prompt
over multiple steps — the dependency is optional and guarded by a
try/except at module level so the rest of the package works without
TextGrad installed.

Registered as ``"textgrad"`` in the optimizer registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import TextGradConfig
from ..optimizer import (
    OptimizationResult,
    PromptOptimizer,
    _TrialBudget,
    require_optional,
)
from ..optimizer_registry import register_prompt_optimizer

if TYPE_CHECKING:
    from healthbench_agent.domain.dataset import HealthBenchSample

    from ..metric import EndToEndMetric

try:
    import textgrad
except ImportError:
    textgrad = None  # type: ignore[assignment]


@register_prompt_optimizer("textgrad", TextGradConfig)
class TextGradOptimizer(PromptOptimizer):
    """Prompt optimizer using text-gradient descent.

    Creates a ``textgrad.Variable`` from the current prompt and iteratively
    refines it by computing text-based loss gradients and applying updates
    via ``textgrad.TGD``. Tracks the best prompt/score across all steps.

    The number of optimization steps is capped at
    ``min(config.steps, config.max_trials)`` so the global trial budget
    from :class:`BaseOptimizationConfig` is honoured.

    Requires both ``samples`` and ``metric`` — raises ``ValueError`` if
    either is missing.

    Attributes:
        config: TextGrad-specific optimization configuration.
    """

    def __init__(self, config: TextGradConfig) -> None:
        self.config = config

    def optimize(
        self,
        current_prompt: str,
        samples: list[HealthBenchSample] | None,
        metric: EndToEndMetric | None,
    ) -> OptimizationResult:
        """Optimize a prompt using text-gradient descent.

        Creates a TextGrad variable for the prompt, an engine for the
        meta-model, and a TGD optimizer. Runs up to
        ``min(config.steps, config.max_trials)`` iterations of: score
        the current prompt, compute a loss variable, backward pass to
        get text gradients, and optimizer step to update the prompt.

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset. Required — raises ValueError
                if None.
            metric: Callable that scores a prompt end-to-end. Required —
                raises ValueError if None.

        Returns:
            OptimizationResult with the best prompt found across all steps.

        Raises:
            ImportError: If the ``textgrad`` package is not installed.
            ValueError: If samples or metric is None.
        """
        require_optional(textgrad, "TextGradOptimizer")
        samples, metric = self._require_samples_and_metric(samples, metric, "TextGradOptimizer")

        # Create TextGrad variable for the prompt
        prompt_var = textgrad.Variable(
            current_prompt,
            role_description="system prompt for a health assistant",
            requires_grad=True,
        )

        # Create engine and optimizer
        engine = textgrad.get_engine(f"{self.config.meta_provider}/{self.config.meta_model}")
        optimizer = textgrad.TGD(parameters=[prompt_var], engine=engine)

        # Score baseline outside the budget — TextGrad treats baseline
        # scoring as a separate cost from the optimization loop.
        baseline_score = metric(current_prompt)

        # Centralised cache + history + best-tracking + budget guard.
        # Seed the best tracker with the baseline so the loop only
        # promotes a candidate on strict improvement over baseline.
        budget = _TrialBudget(metric, self.config.max_trials)
        budget.best_prompt = current_prompt
        budget.best_score = baseline_score

        # Cap steps at the global trial budget so the loop never
        # outruns the budget — _TrialBudget would raise on overflow,
        # but a pre-capped loop is clearer than exception control flow.
        max_steps = min(self.config.steps, self.config.max_trials)

        for _step in range(max_steps):
            # Score current prompt variable through the budget so it
            # lands in history and best tracking automatically.
            score = budget.evaluate(prompt_var.value)

            # Create loss variable (negative score — we maximize)
            loss = textgrad.Variable(
                f"Loss: {1.0 - score:.4f}",
                role_description="optimization loss to minimize",
            )

            # Backward pass to compute text gradients
            loss.backward()

            # Optimizer step to update prompt
            optimizer.step()

        return OptimizationResult(
            optimized_prompt=budget.best_prompt,
            baseline_score=baseline_score,
            optimized_score=budget.best_score,
            improvement=budget.best_score - baseline_score,
            num_trials=len(budget.history),
            trial_history=budget.history,
            optimizer_name="textgrad",
            config=self.config.dump_safe(),
        )
