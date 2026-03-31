"""Evaluation result types and the Eval base class.

Mirrors simple-evals SingleEvalResult / EvalResult / Eval. Depends only on
conversation types within this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .conversation import MessageList
from .sampler import SamplerBase


@dataclass
class CriterionVerdict:
    """Typed LLM-judge verdict for a single rubric criterion.

    Field names mirror the grading dict keys used in simple-evals
    (criteria_met, explanation) for easy interop.

    Attributes:
        criterion: The criterion text that was graded.
        criteria_met: Whether the criterion was satisfied by the agent response.
        explanation: Grader explanation for the verdict.
        confidence: Majority-vote confidence from multi-sample scoring.
            Range: [0.0, 1.0]. Defaults to 1.0 for single-sample judges.
    """

    criterion: str
    criteria_met: bool
    explanation: str = ""
    confidence: float = 1.0


@dataclass
class SingleEvalResult:
    """Evaluation outcome for a single conversation sample.

    Mirrors simple-evals SingleEvalResult. HealthBench-specific data such as
    rubric verdicts, prompt_id, and token usage is stored in
    example_level_metadata.

    Attributes:
        score: HealthBench score in (-inf, 1.0]. None if scoring failed.
            Can be negative when penalty criteria dominate.
        metrics: Per-label scores for stratified analysis keyed by tag
            (e.g. 'overall_score', 'accuracy', 'emergency_referral').
        html: Optional HTML rendering of this sample's evaluation.
        convo: The full conversation including the sampled assistant response.
        example_level_metadata: Per-sample data such as verdicts, prompt_id,
            completion_id, usage stats, and raw rubric grades.
    """

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None
    example_level_metadata: dict[str, Any] | None = None


@dataclass
class EvalResult:
    """Aggregate result of running an evaluation over many conversations.

    Mirrors simple-evals EvalResult. One EvalResult is produced per
    agent × prompt-version × dataset-subset run.

    Attributes:
        score: Overall benchmark score (clipped mean across samples). None if
            no samples were scored.
        metrics: Aggregate stratified scores keyed by tag, plus bootstrap
            standard deviations and sample counts (e.g. 'accuracy:bootstrap_std',
            'accuracy:n_samples'). None when no stratification was requested.
        htmls: Per-sample HTML renderings, one per conversation evaluated.
        convos: Sampled conversations, one MessageList per conversation.
        metadata: Run-level metadata: agent_name, prompt_version, model,
            sample_size, dataset_subset, and per-sample metadata list.
    """

    score: float | None
    metrics: dict[str, float] | None
    htmls: list[str]
    convos: list[MessageList]
    metadata: dict[str, Any] | None


class Eval:
    """Abstract base class for a HealthBench evaluation.

    Subclasses implement __call__ to score a sampler against a dataset
    and return an aggregate EvalResult.
    """

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """Run the evaluation against the provided sampler.

        Args:
            sampler: Model sampler to evaluate.

        Returns:
            Aggregate EvalResult with overall score and per-sample data.
        """
        raise NotImplementedError
