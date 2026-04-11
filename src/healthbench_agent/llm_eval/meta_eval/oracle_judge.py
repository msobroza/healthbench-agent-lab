"""Deterministic oracle judge used as a baseline evaluation strategy.

Provides :class:`OracleJudge`, a :class:`JudgeGrader` implementation
that never touches the network. It is the "known-answer" baseline
evaluator for meta-evaluation smoke tests, documentation snippets, and
offline runs where real LLM grading would be cost-prohibitive or
non-reproducible.
"""

from __future__ import annotations

from collections.abc import Callable

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.rubric import RubricItem


class OracleJudge(JudgeGrader):
    """Deterministic oracle JudgeGrader for meta-evaluation smoke tests and demos.

    An oracle judge scores rubric items against a deterministic strategy
    rather than calling an LLM. Used as the baseline "known-answer"
    evaluator in meta-evaluation smoke tests, documentation snippets,
    and offline runs where real LLM grading would be cost-prohibitive
    or non-reproducible.

    Strategies:
        - ``"always_met"`` / ``"always_fail"`` / ``"alternating"`` —
          built-in patterns.
        - ``dict[str, bool]`` keyed by criterion text — scripted
          per-rubric verdicts.
        - ``Callable[[RubricItem], bool]`` — arbitrary custom logic.
    """

    def __init__(
        self,
        strategy: str | dict[str, bool] | Callable[[RubricItem], bool] = "always_met",
    ) -> None:
        self.strategy = strategy

    def grade(
        self,
        conversation: MessageList,
        rubric_items: list[RubricItem],
    ) -> list[CriterionVerdict]:
        """Return a deterministic verdict for every rubric item.

        Args:
            conversation: Conversation under grading. Ignored by the
                oracle — retained for :class:`JudgeGrader` compatibility.
            rubric_items: Rubric items to score.

        Returns:
            One :class:`CriterionVerdict` per rubric item, in input order.
        """
        verdicts: list[CriterionVerdict] = []
        for idx, item in enumerate(rubric_items):
            met = self._verdict_for(item, idx)
            verdicts.append(
                CriterionVerdict(criterion=item.criterion, criteria_met=met, explanation="oracle")
            )
        return verdicts

    def _verdict_for(self, item: RubricItem, idx: int) -> bool:
        strategy = self.strategy
        if callable(strategy):
            return bool(strategy(item))
        if isinstance(strategy, dict):
            return bool(strategy.get(item.criterion, False))
        if strategy == "always_met":
            return True
        if strategy == "always_fail":
            return False
        if strategy == "alternating":
            return idx % 2 == 0
        raise ValueError(f"Unknown OracleJudge strategy: {strategy!r}")
