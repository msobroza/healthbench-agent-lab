"""Prompt optimizer abstraction and result types.

Defines the contract that any prompt optimization backend (DSPy, TextGrad,
critique-refine) must implement. Result types are frozen dataclasses for
immutability and easy serialization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from healthbench_agent.domain.dataset import HealthBenchSample

    from .metric import EndToEndMetric


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
