"""Evaluation runner orchestrating batch and async LLM-as-judge execution.

Supports two modes:
- async (ThreadPoolExecutor): for development iterations, ≤500 samples
- batch (OpenAI Batch API): for full benchmark runs, 50% cost reduction

See SPEC §5.7.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from healthbench_agent.agent import AgentPipeline
from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.evaluation import CriterionVerdict, SingleEvalResult
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.domain.scoring import calculate_score

from .config_grader import EvalMode, JudgeConfig
from .grader import create_judge


class EvalRunner:
    """Orchestrates LLM-as-judge evaluation across multiple samples.

    Depends on the ``JudgeGrader`` abstraction (Dependency Inversion),
    so any grader implementation — LLM-based, rule-based, or human —
    can be injected. Operational settings (mode, concurrency) are read
    from ``JudgeConfig``.

    Attributes:
        judge: Grader that evaluates conversations against rubrics.
        config: Judge configuration controlling mode and concurrency.
    """

    def __init__(
        self,
        judge: JudgeGrader,
        config: JudgeConfig | None = None,
    ) -> None:
        self.judge = judge
        self.config = config or JudgeConfig()

    @classmethod
    def from_config(cls, config: JudgeConfig | None = None) -> EvalRunner:
        """Create an EvalRunner from a JudgeConfig.

        Convenience factory that builds the judge from config, avoiding
        the need for callers to construct the judge themselves.

        Args:
            config: Judge configuration. Uses defaults if None.

        Returns:
            A fully configured EvalRunner.
        """
        cfg = config or JudgeConfig()
        judge = create_judge(cfg)
        return cls(judge=judge, config=cfg)

    def evaluate_sample(
        self,
        sample: HealthBenchSample,
        response_text: str,
    ) -> SingleEvalResult:
        """Evaluate a single sample's agent response against its rubric.

        Appends the response as an assistant turn, grades each rubric item
        via the judge, and computes the HealthBench score.

        Args:
            sample: The HealthBench sample with prompt and rubric.
            response_text: The agent's response text.

        Returns:
            SingleEvalResult with score, per-tag metrics, and verdicts.
        """
        conversation: MessageList = list(sample.prompt) + [
            {"role": "assistant", "content": response_text}
        ]

        verdicts = self.judge.grade(conversation, sample.rubrics)

        score = calculate_score(sample.rubrics, verdicts)
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

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
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
            NotImplementedError: If mode is ``BATCH`` (not yet implemented).
        """
        if self.config.mode == EvalMode.ASYNC:
            return self.run_async(samples, responses)
        raise NotImplementedError(  # pragma: no cover
            "Batch mode (OpenAI Batch API) is not yet implemented. Use mode='async' for now."
        )

    async def evaluate_pipeline(
        self,
        pipeline: AgentPipeline,
        samples: list[HealthBenchSample],
    ) -> list[SingleEvalResult]:
        """Generate responses via an agent pipeline, then evaluate them.

        Each sample's conversation is sent to the pipeline to produce a
        response. The response is appended to the conversation and then
        graded by the judge.

        Args:
            pipeline: Agent pipeline that generates responses asynchronously.
            samples: HealthBench samples to evaluate.

        Returns:
            List of SingleEvalResult in the same order as input samples.
        """
        responses = await asyncio.gather(*[pipeline.generate(sample.prompt) for sample in samples])
        return self.run(samples, list(responses))


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
    metrics: dict[str, float] = {}

    tag_groups: dict[str, list[tuple[RubricItem, CriterionVerdict]]] = {}
    for item, verdict in zip(sample.rubrics, verdicts):
        for tag in item.tags:
            tag_groups.setdefault(tag, []).append((item, verdict))

    for tag, pairs in tag_groups.items():
        items = [p[0] for p in pairs]
        tag_verdicts = [p[1] for p in pairs]
        tag_score = calculate_score(items, tag_verdicts)
        if tag_score is not None:
            metrics[tag] = tag_score

    return metrics
