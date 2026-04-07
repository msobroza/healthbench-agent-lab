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

from healthbench_agent.agent import (
    AgentNodeConfig,
    RootAgentPipelineConfig,
    create_pipeline,
)
from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.scoring import aggregate_scores
from healthbench_agent.llm_eval.runner import EvalRunner


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
    """

    def __init__(
        self,
        agent_config: RootAgentPipelineConfig,
        judge: JudgeGrader,
        samples: list[HealthBenchSample],
        target_agent_name: str | None = None,
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
        if self.target_agent_name is None:
            patched_config = self.agent_config.model_copy(update={"instruction_override": prompt})
        else:
            patched_config = self.agent_config.model_copy(deep=True)
            target_node = find_agent_node(patched_config, self.target_agent_name)
            # Validated in __init__, so target_node is guaranteed to exist.
            assert target_node is not None
            target_node.instruction_override = prompt
        pipeline = create_pipeline(patched_config)
        results = asyncio.run(self.runner.evaluate_pipeline(pipeline, self.samples))
        return aggregate_scores(results)
