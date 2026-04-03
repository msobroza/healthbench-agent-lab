"""TextGrad prompt optimizer adapter.

Wraps text-gradient descent for iterative prompt refinement. Uses the
``textgrad`` library to compute text-based gradients and update the prompt
over multiple steps — the dependency is optional and guarded by a
try/except at module level so the rest of the package works without
TextGrad installed.

Registered as ``"textgrad"`` in the optimizer registry.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..config import TextGradConfig
from ..optimizer import OptimizationResult, PromptOptimizer, TrialRecord
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
        meta-model, and a TGD optimizer. Runs ``config.steps`` iterations
        of: score the current prompt, compute a loss variable, backward
        pass to get text gradients, and optimizer step to update the prompt.

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset. Required — raises ValueError
                if None.
            metric: Callable that scores a prompt end-to-end. Required —
                raises ValueError if None.

        Returns:
            OptimizationResult with the best prompt found across all steps.

        Raises:
            ValueError: If samples or metric is None.
        """
        if samples is None:
            raise ValueError(
                "TextGradOptimizer requires samples for evaluation. "
                "Pass a non-empty list of HealthBenchSample."
            )
        if metric is None:
            raise ValueError(
                "TextGradOptimizer requires a metric for scoring. Pass an EndToEndMetric callable."
            )

        # Create TextGrad variable for the prompt
        prompt_var = textgrad.Variable(
            current_prompt,
            role_description="system prompt for a health assistant",
            requires_grad=True,
        )

        # Create engine and optimizer
        engine = textgrad.get_engine(f"{self.config.meta_provider}/{self.config.meta_model}")
        optimizer = textgrad.TGD(parameters=[prompt_var], engine=engine)

        # Score baseline
        baseline_score = metric(current_prompt)

        # Track best across iterations
        best_prompt = current_prompt
        best_score = baseline_score
        trial_history: list[TrialRecord] = []

        for step in range(self.config.steps):
            # Score current prompt variable
            score = metric(prompt_var.value)

            # Create loss variable (negative score — we maximize)
            loss = textgrad.Variable(
                f"Loss: {1.0 - score:.4f}",
                role_description="optimization loss to minimize",
            )

            # Backward pass to compute text gradients
            loss.backward()

            # Optimizer step to update prompt
            optimizer.step()

            # Record trial
            trial_history.append(
                TrialRecord(
                    trial_id=step + 1,
                    prompt=prompt_var.value,
                    score=score,
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            )

            # Track best (>= so gradient updates are preserved on ties)
            if score >= best_score:
                best_prompt = prompt_var.value
                best_score = score

        return OptimizationResult(
            optimized_prompt=best_prompt,
            baseline_score=baseline_score,
            optimized_score=best_score,
            improvement=best_score - baseline_score,
            num_trials=len(trial_history),
            trial_history=trial_history,
            optimizer_name="textgrad",
            config=self.config.model_dump(exclude={"google_api_key", "openai_api_key"}),
        )
