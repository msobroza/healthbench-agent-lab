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
candidate prompt injected via ``instruction_override`` on a chosen
node of the agent tree.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from healthbench_agent.agent import (
    AgentNodeConfig,
    RootAgentPipelineConfig,
    create_pipeline,
)
from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.domain.scoring import aggregate_scores
from healthbench_agent.llm_eval.meta_eval import EmptyFilterError, run_meta_eval
from healthbench_agent.llm_eval.runner import EvalRunner

if TYPE_CHECKING:
    from healthbench_agent.llm_eval.grading.config import JudgeConfig


def find_agent_node(root: AgentNodeConfig, name: str) -> AgentNodeConfig | None:
    """Recursively search an agent tree for a node by name.

    Args:
        root: Root of the agent tree to search.
        name: Target agent ``name`` to look for.

    Returns:
        The matching node, or ``None`` if no node with that name exists.
    """
    if root.name == name:
        return root
    for child in root.sub_agents:
        found = find_agent_node(child, name)
        if found is not None:
            return found
    return None


def list_agent_names(root: AgentNodeConfig) -> list[str]:
    """List all agent names in a tree in pre-order traversal.

    Args:
        root: Root of the agent tree.

    Returns:
        Flat list of every ``name`` in the tree, including the root.
    """
    names = [root.name]
    for child in root.sub_agents:
        names.extend(list_agent_names(child))
    return names


def locate_target(
    root: AgentNodeConfig,
    name: str,
) -> tuple[AgentNodeConfig, str]:
    """Find a target node and resolve its effective prompt_path.

    Walks the tree from ``root`` tracking the inherited ``prompt_path``
    (a child with an empty ``prompt_path`` inherits its parent's value,
    matching the rule used at build time in the ADK adapter). The
    returned path is the one that would actually be loaded when
    instantiating the target node.

    Args:
        root: Root of the agent tree.
        name: Target agent ``name``.

    Returns:
        A tuple ``(node, effective_prompt_path)`` for the target.

    Raises:
        ValueError: If no node in the tree has the given name.
    """
    result = _walk_with_path(root, name, parent_prompt_path="")
    if result is None:
        available = ", ".join(list_agent_names(root))
        raise ValueError(f"Agent '{name}' not found in pipeline. Available agents: {available}")
    return result


def accepts_instruction_override(node: AgentNodeConfig) -> bool:
    """Return True when a node becomes an ``LlmAgent`` at build time.

    The ADK adapter only calls ``_resolve_instruction`` (which honors
    ``instruction_override``) for nodes that are built as
    ``LlmAgent``:

    * leaf agents (``sub_agents`` empty), or
    * non-leaf agents with ``orchestration == "routing"``.

    Nodes built as pure composites (``SequentialAgent``, ``LoopAgent``,
    ``ParallelAgent``) have no ``instruction`` field, so an override on
    them is silently dropped. Use this guard before setting
    ``instruction_override`` to fail loudly instead.

    Args:
        node: The agent node to inspect.

    Returns:
        ``True`` when the node's ``instruction_override`` would take
        effect, ``False`` when it would be silently dropped.
    """
    if not node.sub_agents:
        return True
    return node.orchestration == "routing"


def _walk_with_path(
    node: AgentNodeConfig,
    target: str,
    parent_prompt_path: str,
) -> tuple[AgentNodeConfig, str] | None:
    """Recursive helper that carries inherited prompt_path through the tree.

    Args:
        node: Current tree node being visited.
        target: Target agent name to match.
        parent_prompt_path: Prompt path inherited from the parent.

    Returns:
        ``(node, effective_prompt_path)`` when the target is found,
        or ``None`` otherwise.
    """
    effective_path = node.prompt_path or parent_prompt_path
    if node.name == target:
        return node, effective_path
    for child in node.sub_agents:
        found = _walk_with_path(child, target, effective_path)
        if found is not None:
            return found
    return None


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

    When ``target_agent_name`` is set, the candidate prompt is injected
    on a specific sub-agent (identified by name) rather than on the
    root. This is the only correct way to optimize a single prompt
    inside a multi-agent pipeline where several sub-agents have their
    own ``prompt_key`` — the other sub-agents keep loading their own
    instructions from YAML, so one run at a time is optimized.

    Attributes:
        agent_config: Base agent config to copy and patch with candidate
            prompts via ``instruction_override``.
        runner: The shared eval runner that owns the judge.
        samples: Fixed evaluation dataset for fair comparison across trials.
        target_agent_name: Optional name of a sub-agent whose instruction
            should receive the candidate prompt. When ``None``, the
            root's ``instruction_override`` is patched instead.
        sample_filter: Optional predicate used to restrict ``samples``
            before each trial. Samples for which it returns ``False``
            are dropped.
        rubric_filter: Optional predicate used to restrict rubric items
            within each surviving sample. Samples with no surviving
            rubric items are dropped.
    """

    def __init__(
        self,
        agent_config: RootAgentPipelineConfig,
        judge: JudgeGrader,
        samples: list[HealthBenchSample],
        target_agent_name: str | None = None,
        sample_filter: Callable[[HealthBenchSample], bool] | None = None,
        rubric_filter: Callable[[RubricItem], bool] | None = None,
    ) -> None:
        if target_agent_name is not None:
            target_node = find_agent_node(agent_config, target_agent_name)
            if target_node is None:
                available = ", ".join(list_agent_names(agent_config))
                raise ValueError(
                    f"Target agent '{target_agent_name}' not found in pipeline. "
                    f"Available agents: {available}"
                )
            if not accepts_instruction_override(target_node):
                raise ValueError(
                    f"Target agent '{target_agent_name}' has orchestration "
                    f"'{target_node.orchestration}' with sub_agents, which "
                    "builds a pure composite (SequentialAgent/LoopAgent/"
                    "ParallelAgent). Composite agents have no instruction "
                    "field, so instruction_override would be silently "
                    "dropped. Target a leaf agent or an agent with "
                    "orchestration='routing' instead."
                )
        self.agent_config = agent_config
        self.runner = EvalRunner(judge=judge)
        self.samples = samples
        self.target_agent_name = target_agent_name
        self.sample_filter = sample_filter
        self.rubric_filter = rubric_filter

    def __call__(self, prompt: str) -> float:
        """Evaluate a candidate prompt end-to-end.

        Patches the agent config with ``instruction_override`` set to
        the candidate (on the root or on ``target_agent_name`` when
        set), builds a fresh pipeline, and delegates to
        :meth:`EvalRunner.evaluate_pipeline` for response generation,
        grading, and per-sample scoring. Returns the aggregate
        HealthBench score (clipped mean across samples).

        Args:
            prompt: The candidate prompt text to evaluate.

        Returns:
            Aggregate HealthBench score in [0.0, 1.0].
        """
        kept: list[HealthBenchSample] = (
            [s for s in self.samples if self.sample_filter(s)]
            if self.sample_filter is not None
            else list(self.samples)
        )
        if not kept:
            raise EmptyFilterError(
                sample_filter=self.sample_filter,
                rubric_filter=self.rubric_filter,
            )
        if self.rubric_filter is not None:
            patched: list[HealthBenchSample] = []
            for sample in kept:
                surviving = [r for r in sample.rubrics if self.rubric_filter(r)]
                if not surviving:
                    continue
                clone = sample.__class__(**{**sample.__dict__, "rubrics": surviving})
                patched.append(clone)
            kept = patched
            if not kept:
                raise EmptyFilterError(
                    sample_filter=self.sample_filter,
                    rubric_filter=self.rubric_filter,
                )
        if self.target_agent_name is None:
            patched_config = self.agent_config.model_copy(update={"instruction_override": prompt})
        else:
            patched_config = self.agent_config.model_copy(deep=True)
            target_node = find_agent_node(patched_config, self.target_agent_name)
            # Validated in __init__, so target_node is guaranteed to exist.
            assert target_node is not None
            target_node.instruction_override = prompt
        pipeline = create_pipeline(patched_config)
        results = asyncio.run(self.runner.evaluate_pipeline(pipeline, kept))
        return aggregate_scores(results)


class JudgeAgreementMetric:
    """Score a candidate grader prompt via meta-evaluation against a labelled set.

    Drop-in fitness function for :class:`PromptOptimizer` backends, but
    instead of running an agent pipeline (like :class:`EndToEndMetric`)
    it runs the candidate grader prompt against a fixed set of
    :class:`LabelledSample` objects and returns a scalar agreement
    score from :func:`run_meta_eval`.

    Satisfies the :class:`OptimizationMetric` Protocol structurally.

    Subclasses can override :meth:`_build_judge` to inject a fake judge
    for tests; the default implementation constructs an
    :class:`LLMJudgeGrader` from ``judge_config`` with the candidate
    template swapped in.

    Attributes:
        judge_config: Source :class:`JudgeConfig` cloned per trial.
            May be ``None`` when :meth:`_build_judge` is overridden.
        labelled: Fixed labelled set the judge is graded against.
        dimension_extractor: Extracts a dimension key from each
            :class:`RubricItem` (e.g. an axis or category).
        n_samples: Number of judge samples to draw per rubric item.
        fitness: Name of the metric whose scalar value is returned.
        sample_filter: Optional predicate restricting the labelled set.
        rubric_filter: Optional predicate restricting rubric items.
    """

    def __init__(
        self,
        judge_config: JudgeConfig | None,
        labelled: list[LabelledSample],
        dimension_extractor: Callable[[RubricItem], str | None],
        n_samples: int = 3,
        fitness: str = "gold_score",
        sample_filter: Callable[[LabelledSample], bool] | None = None,
        rubric_filter: Callable[[RubricItem], bool] | None = None,
    ) -> None:
        """Initialise the metric with a fixed labelled set and judge config.

        Args:
            judge_config: Source :class:`JudgeConfig`. May be ``None``
                only when :meth:`_build_judge` is overridden.
            labelled: Labelled samples to grade.
            dimension_extractor: Extracts a dimension key per rubric item.
            n_samples: Number of judge samples per rubric item.
            fitness: Name of the metric returned by ``__call__``.
            sample_filter: Optional predicate restricting samples.
            rubric_filter: Optional predicate restricting rubric items.
        """
        self.judge_config = judge_config
        self.labelled = labelled
        self.dimension_extractor = dimension_extractor
        self.n_samples = n_samples
        self.fitness = fitness
        self.sample_filter = sample_filter
        self.rubric_filter = rubric_filter

    def _build_judge(self, candidate_template: str) -> tuple[JudgeGrader, str, str]:
        """Construct a judge from the candidate grader template.

        Override this in tests to inject a fake judge. The default
        implementation clones ``self.judge_config`` and builds an
        :class:`LLMJudgeGrader` whose Jinja2 template has been replaced
        with ``candidate_template``. Returns the judge together with a
        model fingerprint and the sha256 of the template.

        Args:
            candidate_template: The candidate grader prompt template.

        Returns:
            Tuple ``(judge, model_fingerprint, prompt_sha)``.

        Raises:
            RuntimeError: If ``judge_config`` is ``None`` and this
                method was not overridden.
        """
        import hashlib

        from healthbench_agent.llm_eval.clients import create_llm_client
        from healthbench_agent.llm_eval.grading.judge import LLMJudgeGrader, make_template

        if self.judge_config is None:
            raise RuntimeError("judge_config is None and _build_judge was not overridden")
        cfg = self.judge_config.model_copy()
        llm_client = create_llm_client(cfg)
        judge = LLMJudgeGrader(
            llm_client=llm_client,
            template=make_template(candidate_template),
            max_workers=cfg.grader_max_workers,
        )
        fingerprint = f"{cfg.provider}/{cfg.model}@{cfg.temperature}"
        prompt_sha = hashlib.sha256(candidate_template.encode("utf-8")).hexdigest()
        return judge, fingerprint, prompt_sha

    def __call__(self, candidate_template: str) -> float:
        """Evaluate a candidate grader template and return a scalar score.

        Delegates to :func:`run_meta_eval` and returns the scalar
        value of ``scores[self.fitness]``.

        Args:
            candidate_template: The grader prompt template under test.

        Returns:
            Scalar agreement score; higher is better.
        """
        judge, fingerprint, _ = self._build_judge(candidate_template)
        view = run_meta_eval(
            judge=judge,
            labelled=self.labelled,
            dimension_extractor=self.dimension_extractor,
            metric_names=[self.fitness],
            n_samples=self.n_samples,
            sample_filter=self.sample_filter,
            rubric_filter=self.rubric_filter,
            judge_metadata={"judge_model": fingerprint},
            progress=False,
        )
        return float(view.results.scores[self.fitness])
