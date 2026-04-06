"""Prompt optimizer abstraction and result types.

Defines the contract that any prompt optimization backend (DSPy, TextGrad,
critique-refine) must implement. Result types are frozen dataclasses for
immutability and easy serialization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from healthbench_agent.domain.dataset import HealthBenchSample

    from .config import BaseOptimizationConfig
    from .metric import EndToEndMetric


def require_optional(module: object | None, optimizer_name: str) -> None:
    """Raise a helpful ImportError if an optional dependency is missing.

    Adapters lazy-import their backend (``dspy``, ``textgrad``, ...) so
    the rest of the package keeps working without the ``optimization``
    extra. This helper centralises the error message so every adapter
    points users to the same install command.

    Args:
        module: The optionally-imported module, or ``None`` if the
            import failed.
        optimizer_name: User-facing name of the optimizer requesting
            the dependency.

    Raises:
        ImportError: If ``module`` is ``None``.
    """
    if module is None:
        raise ImportError(
            f"{optimizer_name} requires the 'optimization' extra. "
            "Install with: uv sync --extra optimization"
        )


@dataclass(frozen=True)
class TrialRecord:
    """Single optimization trial.

    Attributes:
        trial_id: Sequential trial number.
        prompt: The candidate prompt evaluated in this trial.
        score: Evaluation score, or None in mutation-only mode.
        timestamp: ISO 8601 timestamp of when the trial completed.
    """

    trial_id: int
    prompt: str
    score: float | None
    timestamp: str


@dataclass(frozen=True)
class OptimizationResult:
    """Result of a prompt optimization run.

    Attributes:
        optimized_prompt: The best prompt found during optimization.
        baseline_score: Score of the original prompt before optimization.
        optimized_score: Score of the best prompt found.
        improvement: Score delta (optimized_score - baseline_score).
        num_trials: Total number of candidate prompts evaluated.
        trial_history: Per-trial details for reproducibility.
        optimizer_name: Identifier of the optimizer used.
        config: Serialized optimizer configuration for reproducibility.
    """

    optimized_prompt: str
    baseline_score: float
    optimized_score: float
    improvement: float
    num_trials: int
    trial_history: list[TrialRecord]
    optimizer_name: str
    config: dict[str, Any]


class PromptOptimizer(ABC):
    """Abstract base for prompt optimizers.

    Subclasses implement the optimization logic for a specific framework
    (DSPy, TextGrad, critique-refine) and return an OptimizationResult.
    """

    def __init__(self, config: BaseOptimizationConfig) -> None:
        """Accept an optimizer configuration.

        Subclasses are expected to assign ``self.config`` with the
        concrete config subclass; this base ``__init__`` only declares
        the constructor signature so the registry factory can call
        ``optimizer_class(config)`` polymorphically.

        Args:
            config: Optimizer configuration; concrete subclass varies.
        """

    @abstractmethod
    def optimize(
        self,
        current_prompt: str,
        samples: list[HealthBenchSample] | None,
        metric: EndToEndMetric | None,
    ) -> OptimizationResult:
        """Optimize a prompt against a scoring metric.

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset. Required for DSPy/TextGrad,
                optional for critique-refine (mutation-only mode).
            metric: Callable that scores a prompt end-to-end.
                Required for DSPy/TextGrad, optional for critique-refine.

        Returns:
            OptimizationResult with the best prompt and trial history.
        """
        ...

    @staticmethod
    def _require_samples_and_metric(
        samples: list[HealthBenchSample] | None,
        metric: EndToEndMetric | None,
        optimizer_name: str,
    ) -> tuple[list[HealthBenchSample], EndToEndMetric]:
        """Validate and return the mandatory ``samples`` and ``metric``.

        Centralises the input check shared by every optimizer that
        needs end-to-end scoring (DSPy, TextGrad). Critique-refine has
        a mutation-only mode and does not call this helper. Returning
        the validated values lets callers rebind their locals so the
        type checker sees the narrowed types without ``assert`` noise.

        Args:
            samples: Evaluation dataset.
            metric: End-to-end scoring callable.
            optimizer_name: User-facing name shown in the error message.

        Returns:
            Tuple ``(samples, metric)`` with both values guaranteed non-None.

        Raises:
            ValueError: If ``samples`` or ``metric`` is ``None``.
        """
        if samples is None:
            raise ValueError(
                f"{optimizer_name} requires samples for evaluation. "
                "Pass a non-empty list of HealthBenchSample."
            )
        if metric is None:
            raise ValueError(
                f"{optimizer_name} requires a metric for scoring. Pass an EndToEndMetric callable."
            )
        return samples, metric


class _BudgetExceededError(Exception):
    """Raised by :class:`_TrialBudget` when ``max_trials`` is exhausted.

    Adapters whose backend calls the metric in an unpredictable pattern
    (e.g. DSPy teleprompters) catch this to abort compilation early
    while keeping whatever best prompt has already been found.
    """


class _TrialBudget:
    """Trial cache, history, best-tracker, and budget guard in one place.

    Centralises the bookkeeping every adapter needs around the user-supplied
    :class:`EndToEndMetric`:

    * **Cache** -- repeated calls with the same prompt return the cached
      score instead of re-running the metric. Essential for DSPy, where
      teleprompters call the metric per training example sharing one
      candidate instruction.
    * **History** -- every distinct prompt becomes a :class:`TrialRecord`
      with auto-assigned ``trial_id`` and timestamp.
    * **Best tracking** -- ``best_prompt`` and ``best_score`` are updated
      on strict improvement so the best result is always available, even
      if the budget aborts mid-run.
    * **Budget guard** -- once ``len(history) >= max_trials``, the next
      cache miss raises :class:`_BudgetExceededError`. Cache hits are
      always free.

    The two "best" attributes are public so callers can seed them with
    a baseline score that was computed outside the budget (e.g. TextGrad
    scores its baseline before entering the optimization loop).

    Attributes:
        metric: The end-to-end scoring callable.
        max_trials: Maximum number of distinct prompts to evaluate.
        history: Per-trial records, in order of first evaluation.
        best_prompt: The highest-scoring prompt seen so far.
        best_score: The score of ``best_prompt``.
    """

    def __init__(self, metric: EndToEndMetric, max_trials: int) -> None:
        """Initialise an empty budget.

        Args:
            metric: End-to-end scoring callable.
            max_trials: Maximum number of distinct prompts to evaluate.
        """
        self.metric = metric
        self.max_trials = max_trials
        self._cache: dict[str, float] = {}
        self.history: list[TrialRecord] = []
        self.best_prompt: str = ""
        self.best_score: float = float("-inf")

    def evaluate(self, prompt: str) -> float:
        """Score a prompt, hitting the cache if it has been seen before.

        Updates ``history`` and ``best_prompt``/``best_score`` on the
        first evaluation of a new prompt. Cache hits never count toward
        the budget.

        Args:
            prompt: The candidate prompt to score.

        Returns:
            The score for ``prompt``.

        Raises:
            _BudgetExceededError: On a cache miss when ``max_trials``
                trials have already been recorded.
        """
        if prompt in self._cache:
            return self._cache[prompt]
        if len(self.history) >= self.max_trials:
            raise _BudgetExceededError
        score = self.metric(prompt)
        self._cache[prompt] = score
        self.history.append(
            TrialRecord(
                trial_id=len(self.history) + 1,
                prompt=prompt,
                score=score,
                timestamp=datetime.now(tz=UTC).isoformat(),
            )
        )
        if score > self.best_score:
            self.best_score = score
            self.best_prompt = prompt
        return score
