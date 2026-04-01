"""Evaluation runner orchestrating batch and async LLM-as-judge execution.

Supports two modes:
- async (ThreadPoolExecutor): for development iterations, ≤500 samples
- batch (OpenAI Batch API): for full benchmark runs, 50% cost reduction

See SPEC §5.7 and AGENT_DECISIONS.md §13.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

from jinja2 import Template

from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.evaluation import CriterionVerdict, SingleEvalResult
from healthbench_agent.domain.sampler import SamplerBase
from healthbench_agent.domain.scoring import calculate_score

from .grader import (
    GRADER_TEMPLATE,
    format_conversation,
    grade_sample,
    parse_grading_response,
)


class EvalRunner:
    """Orchestrates LLM-as-judge evaluation across multiple samples.

    Attributes:
        sampler: Model sampler for grading (OpenAI or Gemini).
        template: Jinja2 grader prompt template.
        max_workers: ThreadPool concurrency for async mode.
        mode: Execution mode — "async" or "batch".
    """

    def __init__(
        self,
        sampler: SamplerBase,
        template: Template | None = None,
        max_workers: int = 120,
        mode: Literal["async", "batch"] = "async",
    ) -> None:
        self.sampler = sampler
        self.template = template or Template(GRADER_TEMPLATE)
        self.max_workers = max_workers
        self.mode = mode

    def evaluate_sample(
        self,
        sample: HealthBenchSample,
        response_text: str,
    ) -> SingleEvalResult:
        """Evaluate a single sample's agent response against its rubric.

        Appends the response as an assistant turn, grades each rubric item,
        and computes the HealthBench score.

        Args:
            sample: The HealthBench sample with prompt and rubric.
            response_text: The agent's response text.

        Returns:
            SingleEvalResult with score, per-tag metrics, and verdicts.
        """
        # Build full conversation with agent response
        conversation = list(sample.prompt) + [
            {"role": "assistant", "content": response_text}
        ]

        # Grade each rubric item
        verdicts = grade_sample(
            sampler=self.sampler,
            conversation=conversation,
            rubric_items=sample.rubrics,
            template=self.template,
        )

        # Compute score
        score = calculate_score(sample.rubrics, verdicts)

        # Compute per-tag metrics
        metrics = _compute_tag_metrics(sample, verdicts)

        return SingleEvalResult(
            score=score,
            metrics=metrics,
            convo=conversation,
            example_level_metadata={
                "prompt_id": sample.prompt_id,
                "verdicts": [
                    {
                        "criterion": v.criterion,
                        "criteria_met": v.criteria_met,
                        "explanation": v.explanation,
                    }
                    for v in verdicts
                ],
                "example_tags": sample.example_tags,
            },
        )

    def run_async(
        self,
        samples: list[HealthBenchSample],
        responses: list[str],
    ) -> list[SingleEvalResult]:
        """Evaluate samples concurrently using ThreadPoolExecutor.

        Args:
            samples: HealthBench samples to evaluate.
            responses: Agent responses, one per sample in the same order.

        Returns:
            List of SingleEvalResult in the same order as input samples.
        """
        results: list[SingleEvalResult | None] = [None] * len(samples)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.evaluate_sample, sample, response): i
                for i, (sample, response) in enumerate(zip(samples, responses))
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()

        return results  # type: ignore[return-value]

    def run(
        self,
        samples: list[HealthBenchSample],
        responses: list[str],
    ) -> list[SingleEvalResult]:
        """Evaluate samples using the configured mode.

        Args:
            samples: HealthBench samples to evaluate.
            responses: Agent responses, one per sample in the same order.

        Returns:
            List of SingleEvalResult in the same order as input samples.

        Raises:
            NotImplementedError: If mode is "batch" (not yet implemented).
        """
        if self.mode == "async":
            return self.run_async(samples, responses)
        raise NotImplementedError(  # pragma: no cover
            "Batch mode (OpenAI Batch API) is not yet implemented. "
            "Use mode='async' for now."
        )


def _compute_tag_metrics(
    sample: HealthBenchSample,
    verdicts: list[CriterionVerdict],
) -> dict[str, float]:
    """Compute per-tag scores for a single sample.

    Groups rubric items by their tags, applies calculate_score to each
    group, and includes example_tags as additional metric keys.

    Args:
        sample: The HealthBench sample.
        verdicts: Grading verdicts, one per rubric item.

    Returns:
        Dict mapping tag names to scores.
    """
    from healthbench_agent.domain.rubric import RubricItem

    metrics: dict[str, float] = {}

    # Group rubric items by tag
    tag_groups: dict[str, list[tuple[RubricItem, CriterionVerdict]]] = {}
    for item, verdict in zip(sample.rubrics, verdicts):
        for tag in item.tags:
            tag_groups.setdefault(tag, []).append((item, verdict))

    # Compute score per tag
    for tag, pairs in tag_groups.items():
        items = [p[0] for p in pairs]
        tag_verdicts = [p[1] for p in pairs]
        tag_score = calculate_score(items, tag_verdicts)
        if tag_score is not None:
            metrics[tag] = tag_score

    return metrics
