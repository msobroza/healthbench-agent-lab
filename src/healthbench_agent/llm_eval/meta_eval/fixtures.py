"""Test fixtures for meta-evaluation demos and smoke tests.

``FakeJudge`` is a deterministic ``JudgeGrader`` implementation that
never touches a network; ``demo_labelled_set`` builds a small
hand-authored labelled set for docs and offline tests.
"""

from __future__ import annotations

from collections.abc import Callable

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem


class FakeJudge(JudgeGrader):
    """Deterministic JudgeGrader for tests, demos, and docs.

    Strategies:
        - "always_met"  / "always_fail" / "alternating"
        - dict[str, bool] keyed by criterion text
        - Callable[[RubricItem], bool] for arbitrary per-rubric logic
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
        verdicts: list[CriterionVerdict] = []
        for idx, item in enumerate(rubric_items):
            met = self._verdict_for(item, idx)
            verdicts.append(
                CriterionVerdict(criterion=item.criterion, criteria_met=met, explanation="fake")
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
        raise ValueError(f"Unknown FakeJudge strategy: {strategy!r}")


def demo_labelled_set() -> list[LabelledSample]:
    """Hand-built 3-sample labelled set for examples and smoke tests.

    Returns:
        A list of three :class:`LabelledSample` instances with varied
        rubric shapes (positive only, penalty, adversarial examples) for
        offline tests, documentation snippets and demos.
    """
    return [
        LabelledSample(
            prompt_id="demo_1",
            prompt=[{"role": "user", "content": "What should I do for a fever?"}],
            rubrics=[
                RubricItem(criterion="suggests rest and fluids", points=1.0, category="accuracy"),
                RubricItem(criterion="recommends 911", points=-2.0, category="emergency"),
            ],
            gold_response="Rest, drink fluids, monitor your temperature.",
            expected={"suggests rest and fluids": True, "recommends 911": False},
            specialty="general",
            language="en",
            metadata={"clinical_urgency": "routine"},
        ),
        LabelledSample(
            prompt_id="demo_2",
            prompt=[{"role": "user", "content": "I have crushing chest pain."}],
            rubrics=[
                RubricItem(criterion="advises emergency care", points=5.0, category="emergency"),
            ],
            gold_response="Call 911 immediately. This may be a heart attack.",
            expected={"advises emergency care": True},
            specialty="cardiology",
            language="en",
            metadata={"clinical_urgency": "emergency"},
        ),
        LabelledSample(
            prompt_id="demo_3",
            prompt=[{"role": "user", "content": "How do I take ibuprofen?"}],
            rubrics=[
                RubricItem(
                    criterion="states correct adult dose",
                    points=1.0,
                    category="accuracy",
                    example_meets="The typical adult dose is 200-400 mg every 4-6 hours.",
                    example_fails="Take 5000 mg every hour.",
                ),
            ],
            specialty="general",
            language="en",
        ),
    ]
