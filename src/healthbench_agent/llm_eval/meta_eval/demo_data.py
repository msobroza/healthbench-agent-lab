"""Hand-built ``LabelledSample`` instances used as example data.

Provides :func:`demo_labelled_set`, a tiny hand-authored labelled set
for documentation snippets and offline smoke tests. Ships with varied
rubric shapes (positive only, penalty, adversarial examples) so that
example snippets can exercise every code path without loading the
full HealthBench dataset.
"""

from __future__ import annotations

from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem


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
