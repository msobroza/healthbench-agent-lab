"""End-to-end evaluation metric for prompt optimization.

Bridges the optimizer to the existing agent pipeline and LLM judge by
wrapping the full evaluation flow (generate responses, grade, score)
into a single callable: ``metric(prompt) -> float``.

The actual evaluation work — running the pipeline over each sample,
grading via the judge, computing per-sample and per-tag scores — is
delegated to :class:`healthbench_agent.llm_eval.runner.EvalRunner`,
which is already the project's single source of truth for the
end-to-end agent + judge loop. This module only contributes the
optimization-specific concern: rebuilding the pipeline with the
candidate prompt injected via ``instruction_override``.
"""

from __future__ import annotations

import asyncio

from healthbench_agent.agent import RootAgentPipelineConfig, create_pipeline
from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.scoring import aggregate_scores
from healthbench_agent.llm_eval.runner import EvalRunner


class EndToEndMetric:
    """Scores a candidate prompt by running agent generation + LLM judge grading.

    Builds a fresh :class:`AgentPipeline` per call (so the candidate
    prompt actually takes effect via ``instruction_override``) and then
    delegates the entire generate-grade-score loop to
    :class:`EvalRunner.evaluate_pipeline`. The aggregate score is the
    clipped mean across samples, identical to how the standard
    HealthBench evaluation reports its overall score.

    Reuses the same :class:`EvalRunner` instance across calls so the
    judge does not get re-initialised on every trial.

    Attributes:
        agent_config: Base agent config to copy and patch with candidate
            prompts via ``instruction_override``.
        runner: The shared eval runner that owns the judge.
        samples: Fixed evaluation dataset for fair comparison across trials.
    """

    def __init__(
        self,
        agent_config: RootAgentPipelineConfig,
        judge: JudgeGrader,
        samples: list[HealthBenchSample],
    ) -> None:
        self.agent_config = agent_config
        self.runner = EvalRunner(judge=judge)
        self.samples = samples

    def __call__(self, prompt: str) -> float:
        """Evaluate a candidate prompt end-to-end.

        Patches the agent config with ``instruction_override`` set to
        the candidate, builds a fresh pipeline, and delegates to
        :meth:`EvalRunner.evaluate_pipeline` for response generation,
        grading, and per-sample scoring. Returns the aggregate
        HealthBench score (clipped mean across samples).

        Args:
            prompt: The candidate prompt text to evaluate.

        Returns:
            Aggregate HealthBench score in [0.0, 1.0].
        """
        patched_config = self.agent_config.model_copy(update={"instruction_override": prompt})
        pipeline = create_pipeline(patched_config)
        results = asyncio.run(self.runner.evaluate_pipeline(pipeline, self.samples))
        return aggregate_scores(results)
