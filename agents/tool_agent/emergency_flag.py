"""Emergency flag tool — keyword-based urgency classification.

Provides emergency_flag() for classifying clinical descriptions by
urgency using tiered keyword matching. Registered in the tool registry
as ``"emergency_flag"``.

See AGENT_DECISIONS.md §5 for design rationale.
"""

from __future__ import annotations

from healthbench_agent.agent import register_tool

EMERGENCY_KEYWORDS: list[str] = [
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "can't breathe",
    "cannot breathe",
    "unconscious",
    "unresponsive",
    "not breathing",
    "cardiac arrest",
    "heart attack",
    "stroke",
    "seizure",
    "severe bleeding",
    "hemorrhage",
    "anaphylaxis",
    "allergic reaction swelling",
    "suicidal",
    "suicide",
    "self-harm",
    "want to die",
    "overdose",
    "poisoning",
    "choking",
    "drowning",
    "severe burn",
    "gunshot",
    "stab wound",
    "head injury",
    "loss of consciousness",
    "coughing blood",
    "vomiting blood",
    "severe abdominal pain",
    "sudden vision loss",
    "sudden weakness one side",
    "slurred speech",
    "facial drooping",
    "baby not breathing",
    "infant turning blue",
    "meningitis",
    "sepsis",
]

URGENT_KEYWORDS: list[str] = [
    "high fever",
    "fever over 103",
    "fever over 39",
    "persistent vomiting",
    "blood in stool",
    "blood in urine",
    "severe headache",
    "worst headache",
    "dehydration",
    "unable to eat",
    "unable to drink",
    "chest tightness",
    "palpitations",
    "racing heart",
    "swollen leg",
    "deep vein",
    "severe pain",
    "fracture",
    "broken bone",
    "concussion",
    "diabetic emergency",
    "low blood sugar",
    "high blood sugar",
    "asthma attack",
    "panic attack",
    "severe anxiety",
    "rash spreading",
    "difficulty swallowing",
    "ear infection child",
    "urinary tract infection",
]


@register_tool("emergency_flag")
def emergency_flag(description: str) -> dict:
    """Classify a clinical description for emergency urgency.

    Uses keyword-based classification with confidence tiers. Designed for
    high sensitivity (low false-negative rate) across diverse clinical
    contexts including global health, context-seeking, and emergency
    referral scenarios.

    Args:
        description: Patient's description of symptoms or situation.

    Returns:
        dict with urgency level (emergency, urgent, or routine),
        confidence score, matched keywords, and recommendation.
    """
    description_lower = description.lower()

    emergency_matches = [k for k in EMERGENCY_KEYWORDS if k in description_lower]
    urgent_matches = [k for k in URGENT_KEYWORDS if k in description_lower]

    if emergency_matches:
        return {
            "status": "success",
            "urgency": "emergency",
            "confidence": min(1.0, len(emergency_matches) * 0.3 + 0.4),
            "matched_keywords": emergency_matches,
            "recommendation": (
                "This situation may require immediate emergency medical care. "
                "Call emergency services (911 or local equivalent) or go to "
                "the nearest emergency department immediately."
            ),
        }

    if urgent_matches:
        return {
            "status": "success",
            "urgency": "urgent",
            "confidence": min(1.0, len(urgent_matches) * 0.25 + 0.3),
            "matched_keywords": urgent_matches,
            "recommendation": (
                "This situation may require prompt medical attention. "
                "Contact your healthcare provider today or visit an "
                "urgent care facility."
            ),
        }

    return {
        "status": "success",
        "urgency": "routine",
        "confidence": 0.5,
        "matched_keywords": [],
        "recommendation": (
            "Based on the description, this does not appear to be an emergency. "
            "However, if symptoms worsen or new concerning symptoms develop, "
            "seek medical attention promptly."
        ),
    }
