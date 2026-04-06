"""DSPy prompt optimizer adapter.

Wraps DSPy's COPRO and MIPROv2 teleprompters for instruction-only prompt
optimization. Uses the ``dspy`` library for compilation — the dependency
is optional and guarded by a try/except at module level so the rest of
the package works without DSPy installed.

Registered as ``"dspy"`` in the optimizer registry.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..config import DSPyConfig
from ..optimizer import OptimizationResult, PromptOptimizer, TrialRecord
from ..optimizer_registry import register_prompt_optimizer

if TYPE_CHECKING:
    from healthbench_agent.domain.dataset import HealthBenchSample

    from ..metric import EndToEndMetric

try:
    import dspy
except ImportError:
    dspy = None  # type: ignore[assignment]


class _BudgetExceededError(Exception):
    """Raised when the optimizer exceeds its trial budget.

    Used internally by the DSPy adapter to abort compilation early
    once ``max_trials`` distinct candidate prompts have been evaluated.
    """


def _check_dspy_installed() -> None:
    """Raise a helpful ImportError if dspy is not installed."""
    if dspy is None:
        raise ImportError(
            "DSPyOptimizer requires the 'optimization' extra. "
            "Install with: uv sync --extra optimization"
        )


@register_prompt_optimizer("dspy", DSPyConfig)
class DSPyOptimizer(PromptOptimizer):
    """Prompt optimizer using DSPy COPRO or MIPROv2 teleprompters.

    Builds a minimal ``dspy.Module`` with a single ``dspy.Predict`` step
    and compiles it using the selected teleprompter. The optimized
    instruction is extracted from the compiled module's signature.

    DSPy teleprompters expect a per-example metric with signature
    ``metric(example, prediction, trace) -> float``. The adapter wraps
    the supplied :class:`EndToEndMetric` (which scores per-instruction)
    in a closure that:

    1. Reads the *current* instruction from the module being compiled
       (``module.generate.signature.instructions``).
    2. Caches the end-to-end score per instruction so DSPy's repeated
       per-example calls only trigger one full evaluation per candidate.
    3. Tracks the best instruction seen so far so the result remains
       valid even if the budget is exhausted mid-compile.

    The ``max_trials`` setting from :class:`BaseOptimizationConfig` is
    enforced as an early-stop budget: once that many distinct candidates
    have been evaluated, the next cache miss raises :class:`_BudgetExceededError`,
    which the adapter catches to return the best prompt found so far.

    Requires both ``samples`` and ``metric`` — raises ``ValueError`` if
    either is missing.

    Attributes:
        config: DSPy-specific optimization configuration.
    """

    def __init__(self, config: DSPyConfig) -> None:
        self.config = config

    def optimize(
        self,
        current_prompt: str,
        samples: list[HealthBenchSample] | None,
        metric: EndToEndMetric | None,
    ) -> OptimizationResult:
        """Optimize a prompt using DSPy teleprompters.

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset. Required — raises ValueError
                if None.
            metric: Callable that scores a prompt end-to-end. Required —
                raises ValueError if None.

        Returns:
            OptimizationResult with the best prompt found by DSPy, or
            the best candidate seen so far if the trial budget is
            exhausted before compilation completes.

        Raises:
            ImportError: If the ``dspy`` package is not installed.
            ValueError: If samples or metric is None, or if
                ``config.dspy_optimizer`` is not ``"copro"`` or ``"miprov2"``.
        """
        _check_dspy_installed()
        if samples is None:
            raise ValueError(
                "DSPyOptimizer requires samples for evaluation. "
                "Pass a non-empty list of HealthBenchSample."
            )
        if metric is None:
            raise ValueError(
                "DSPyOptimizer requires a metric for scoring. Pass an EndToEndMetric callable."
            )

        # Configure DSPy language model
        language_model = dspy.LM(f"{self.config.meta_provider}/{self.config.meta_model}")
        dspy.configure(lm=language_model)

        # Build a minimal DSPy module with the current prompt as instruction
        signature = dspy.Signature("question -> answer").with_instructions(current_prompt)

        class HealthModule(dspy.Module):
            """Minimal DSPy module wrapping a single Predict step."""

            def __init__(self) -> None:
                super().__init__()
                self.generate = dspy.Predict(signature)

            def forward(self, question: str) -> Any:
                """Run prediction for a question.

                Args:
                    question: The input question text.

                Returns:
                    A DSPy Prediction with the answer.
                """
                return self.generate(question=question)

        module = HealthModule()

        # Build trainset from samples
        trainset = [
            dspy.Example(question=sample.prompt[0]["content"]).with_inputs("question")
            for sample in samples
        ]

        # Cached, budget-tracked, DSPy-compatible metric.
        # DSPy calls this once per (example, prediction) pair, so we cache
        # by instruction string to avoid re-running the agent for every
        # example sharing the same candidate.
        score_cache: dict[str, float] = {}
        trial_history: list[TrialRecord] = []
        best_prompt = current_prompt
        best_score = float("-inf")

        def evaluate_prompt(prompt: str) -> float:
            nonlocal best_prompt, best_score
            if prompt in score_cache:
                return score_cache[prompt]
            if len(trial_history) >= self.config.max_trials:
                raise _BudgetExceededError()
            score = metric(prompt)
            score_cache[prompt] = score
            trial_history.append(
                TrialRecord(
                    trial_id=len(trial_history) + 1,
                    prompt=prompt,
                    score=score,
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            )
            if score > best_score:
                best_score = score
                best_prompt = prompt
            return score

        def dspy_metric(_example: Any, _pred: Any, _trace: Any = None) -> float:
            """DSPy-compatible per-example metric.

            Reads the current instruction from the module being
            compiled and delegates to ``evaluate_prompt`` (which caches
            per-instruction).
            """
            current_instruction = module.generate.signature.instructions
            return evaluate_prompt(current_instruction)

        # Score the baseline up front so it appears in the result and
        # the budget always reflects at least one trial.
        baseline_score = evaluate_prompt(current_prompt)

        # Select and run the teleprompter
        if self.config.dspy_optimizer == "copro":
            teleprompter = dspy.COPRO(metric=dspy_metric, verbose=False)
        elif self.config.dspy_optimizer == "miprov2":
            teleprompter = dspy.MIPROv2(metric=dspy_metric, verbose=False)
        else:
            raise ValueError(
                f"Unsupported dspy_optimizer: {self.config.dspy_optimizer!r}. "
                f"Use 'copro' or 'miprov2'."
            )

        try:
            compiled_module = teleprompter.compile(
                module,
                trainset=trainset,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=0,
            )
            # Score the final compiled instruction so it lands in the
            # cache and best-tracking even if DSPy never re-called the
            # metric on it.
            final_instruction = compiled_module.generate.signature.instructions
            evaluate_prompt(final_instruction)
        except _BudgetExceededError:
            # Budget exhausted mid-compile — keep whatever we found.
            pass

        return OptimizationResult(
            optimized_prompt=best_prompt,
            baseline_score=baseline_score,
            optimized_score=best_score,
            improvement=best_score - baseline_score,
            num_trials=len(trial_history),
            trial_history=trial_history,
            optimizer_name="dspy",
            config=self.config.model_dump(exclude={"google_api_key", "openai_api_key"}),
        )
