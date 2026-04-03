"""DSPy prompt optimizer adapter.

Wraps DSPy's COPRO and MIPROv2 teleprompters for instruction-only prompt
optimization. Uses the ``dspy`` library for compilation — the dependency
is optional and guarded by a try/except at module level so the rest of
the package works without DSPy installed.

Registered as ``"dspy"`` in the optimizer registry.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

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


@register_prompt_optimizer("dspy", DSPyConfig)
class DSPyOptimizer(PromptOptimizer):
    """Prompt optimizer using DSPy COPRO or MIPROv2 teleprompters.

    Builds a minimal ``dspy.Module`` with a single ``dspy.Predict`` step
    and compiles it using the selected teleprompter. The optimized
    instruction is extracted from the compiled module's signature.

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

        Configures a ``dspy.LM`` with the meta-model, wraps the current
        prompt in a simple module, and compiles it with either COPRO or
        MIPROv2 depending on ``config.dspy_optimizer``.

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset. Required — raises ValueError
                if None.
            metric: Callable that scores a prompt end-to-end. Required —
                raises ValueError if None.

        Returns:
            OptimizationResult with the best prompt found by DSPy.

        Raises:
            ValueError: If samples or metric is None, or if
                ``config.dspy_optimizer`` is not ``"copro"`` or ``"miprov2"``.
        """
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
        signature = dspy.Signature("question -> answer")
        signature = signature.with_instructions(current_prompt)

        class HealthModule(dspy.Module):
            """Minimal DSPy module wrapping a single Predict step."""

            def __init__(self) -> None:
                super().__init__()
                self.generate = dspy.Predict(signature)

            def forward(self, question: str) -> dspy.Prediction:
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

        # Select and run the teleprompter
        if self.config.dspy_optimizer == "copro":
            teleprompter = dspy.COPRO(metric=metric, verbose=False)
        elif self.config.dspy_optimizer == "miprov2":
            teleprompter = dspy.MIPROv2(metric=metric, verbose=False)
        else:
            raise ValueError(
                f"Unsupported dspy_optimizer: {self.config.dspy_optimizer!r}. "
                f"Use 'copro' or 'miprov2'."
            )

        compiled_module = teleprompter.compile(
            module,
            trainset=trainset,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=0,
        )

        # Extract optimized instruction
        optimized_prompt = compiled_module.generate.signature.instructions

        # Score baseline and optimized prompts
        baseline_score = metric(current_prompt)
        optimized_score = metric(optimized_prompt)

        trial_history = [
            TrialRecord(
                trial_id=1,
                prompt=optimized_prompt,
                score=optimized_score,
                timestamp=datetime.now(tz=UTC).isoformat(),
            ),
        ]

        return OptimizationResult(
            optimized_prompt=optimized_prompt,
            baseline_score=baseline_score,
            optimized_score=optimized_score,
            improvement=optimized_score - baseline_score,
            num_trials=1,
            trial_history=trial_history,
            optimizer_name="dspy",
            config=self.config.model_dump(exclude={"google_api_key", "openai_api_key"}),
        )
