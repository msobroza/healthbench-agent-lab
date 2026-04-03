"""End-to-end evaluation metric for prompt optimization.

Bridges the optimizer to the existing agent pipeline and LLM judge by
wrapping the full evaluation flow (generate responses, grade, score)
into a single callable: ``metric(prompt) -> float``.
"""

from __future__ import annotations

import asyncio

from healthbench_agent.agent import RootAgentPipelineConfig, create_pipeline
from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.evaluation import SingleEvalResult
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.scoring import aggregate_scores, calculate_score


class EndToEndMetric:
    """Scores a prompt by running agent generation + LLM judge grading.

    Stateless with respect to the prompt being tested — builds a fresh
    pipeline per call so the prompt change takes effect. Reuses the same
    judge instance across calls to avoid re-initialization overhead.

    Attributes:
        agent_config: Base agent config to copy and patch with candidate prompts.
        judge: Grader that evaluates conversations against rubrics.
        samples: Fixed evaluation dataset for fair comparison across trials.
    """

    def __init__(
        self,
        agent_config: RootAgentPipelineConfig,
        judge: JudgeGrader,
        samples: list[HealthBenchSample],
    ) -> None:
        self.agent_config = agent_config
        self.judge = judge
        self.samples = samples

    def __call__(self, prompt: str) -> float:
        """Evaluate a candidate prompt end-to-end.

        Creates a copy of the agent config with the candidate prompt
        injected, builds a fresh pipeline, generates responses for all
        samples, grades them via the judge, and returns the aggregate
        score.

        Args:
            prompt: The candidate prompt text to evaluate.

        Returns:
            Aggregate HealthBench score in [0.0, 1.0].
        """
        patched_config = self.agent_config.model_copy(update={"instruction_override": prompt})
        pipeline = create_pipeline(patched_config)

        async def _generate_all() -> list[str]:
            return await asyncio.gather(
                *[pipeline.generate(sample.prompt) for sample in self.samples]
            )

        responses = asyncio.run(_generate_all())

        results = []
        for sample, response in zip(self.samples, responses):
            conversation = list(sample.prompt) + [{"role": "assistant", "content": response}]
            verdicts = self.judge.grade(conversation, sample.rubrics)
            score = calculate_score(sample.rubrics, verdicts)
            results.append(SingleEvalResult(score=score, metrics={}))

        return aggregate_scores(results)
