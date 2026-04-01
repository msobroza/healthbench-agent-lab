"""Judge grader abstraction for evaluating agent responses.

Defines the contract that any grader (LLM-based, rule-based, human) must
implement to be usable by the evaluation runner. Mirrors the SamplerBase
pattern: a pure abstraction in the domain layer, with concrete
implementations in the llm_eval layer.

Nothing in this module imports from the rest of the project except
domain types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .conversation import MessageList
from .evaluation import CriterionVerdict
from .rubric import RubricItem


class JudgeGrader(ABC):
    """Abstract base for graders that evaluate agent responses against rubrics.

    Subclasses implement the grading logic — whether via an LLM call, a
    rule-based system, or human annotation — and return structured verdicts.
    """

    @abstractmethod
    def grade(
        self,
        conversation: MessageList,
        rubric_items: list[RubricItem],
    ) -> list[CriterionVerdict]:
        """Grade a conversation against a list of rubric items.

        Args:
            conversation: Full conversation including the agent's response
                as the last assistant turn.
            rubric_items: Rubric items to grade against.

        Returns:
            List of CriterionVerdict, one per rubric item in the same order.
        """
        ...
