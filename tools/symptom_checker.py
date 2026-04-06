"""Symptom checker tool — symptom-to-condition mapping with urgency levels.

Provides symptom_checker() for checking symptoms against a curated
medical knowledge base. Registered in the tool registry as
``"symptom_checker"``.

See AGENT_DECISIONS.md §6 for design rationale.
"""

from __future__ import annotations

from healthbench_agent.agent import register_tool

_SYMPTOM_DATABASE: dict[str, list[dict]] = {
    "chest pain": [
        {
            "condition": "Angina / Acute Coronary Syndrome",
            "urgency": "emergency",
            "description": "Chest pain or pressure that may indicate a heart attack.",
        },
        {
            "condition": "Costochondritis",
            "urgency": "routine",
            "description": "Inflammation of cartilage connecting ribs to sternum.",
        },
        {
            "condition": "Gastroesophageal reflux (GERD)",
            "urgency": "routine",
            "description": "Acid reflux causing burning chest pain, often worse after eating.",
        },
    ],
    "headache": [
        {
            "condition": "Tension headache",
            "urgency": "routine",
            "description": "Most common type; band-like pressure around the head.",
        },
        {
            "condition": "Migraine",
            "urgency": "routine",
            "description": "Throbbing, often unilateral, with nausea or light sensitivity.",
        },
        {
            "condition": "Subarachnoid hemorrhage",
            "urgency": "emergency",
            "description": (
                "Sudden 'thunderclap' worst-ever headache; requires immediate evaluation."
            ),
        },
    ],
    "fever": [
        {
            "condition": "Viral infection",
            "urgency": "routine",
            "description": "Most common cause; self-limiting, supportive care.",
        },
        {
            "condition": "Bacterial infection",
            "urgency": "urgent",
            "description": "May require antibiotics; evaluate source.",
        },
        {
            "condition": "Sepsis",
            "urgency": "emergency",
            "description": "Fever with altered mental status, hypotension, or tachycardia.",
        },
    ],
    "shortness of breath": [
        {
            "condition": "Asthma exacerbation",
            "urgency": "urgent",
            "description": "Wheezing, cough, chest tightness; may need bronchodilator.",
        },
        {
            "condition": "Pulmonary embolism",
            "urgency": "emergency",
            "description": "Sudden onset, often with pleuritic chest pain and tachycardia.",
        },
        {
            "condition": "Heart failure",
            "urgency": "urgent",
            "description": "Dyspnea on exertion, orthopnea, peripheral edema.",
        },
    ],
    "abdominal pain": [
        {
            "condition": "Gastroenteritis",
            "urgency": "routine",
            "description": "Nausea, vomiting, diarrhea; usually viral and self-limiting.",
        },
        {
            "condition": "Appendicitis",
            "urgency": "emergency",
            "description": (
                "Right lower quadrant pain, fever, guarding; requires surgical evaluation."
            ),
        },
        {
            "condition": "Peptic ulcer",
            "urgency": "urgent",
            "description": "Epigastric pain, worse with meals or on empty stomach.",
        },
    ],
    "fatigue": [
        {
            "condition": "Iron deficiency anemia",
            "urgency": "routine",
            "description": "Fatigue, pallor, weakness; check CBC and iron studies.",
        },
        {
            "condition": "Hypothyroidism",
            "urgency": "routine",
            "description": "Fatigue, weight gain, cold intolerance; check TSH.",
        },
        {
            "condition": "Depression",
            "urgency": "routine",
            "description": "Persistent fatigue with mood changes, sleep disturbance.",
        },
    ],
    "dizziness": [
        {
            "condition": "Benign positional vertigo (BPPV)",
            "urgency": "routine",
            "description": "Brief vertigo triggered by head position changes.",
        },
        {
            "condition": "Orthostatic hypotension",
            "urgency": "routine",
            "description": "Dizziness on standing; check medications and hydration.",
        },
        {
            "condition": "Stroke / TIA",
            "urgency": "emergency",
            "description": "Sudden dizziness with focal neurological symptoms.",
        },
    ],
}


@register_tool("symptom_checker")
def symptom_checker(symptoms: str) -> dict:
    """Check symptoms against a medical knowledge base for possible conditions.

    Args:
        symptoms: Comma-separated list of symptoms the patient is experiencing.

    Returns:
        dict with possible conditions, urgency levels, and a disclaimer
        that this is not a diagnosis.
    """
    symptom_list = [s.strip().lower() for s in symptoms.split(",")]
    all_conditions: list[dict] = []
    matched_symptoms: list[str] = []

    for symptom in symptom_list:
        for known_symptom, conditions in _SYMPTOM_DATABASE.items():
            if known_symptom in symptom or symptom in known_symptom:
                matched_symptoms.append(known_symptom)
                all_conditions.extend(conditions)

    if not all_conditions:
        return {
            "status": "success",
            "symptoms_analyzed": symptom_list,
            "matched_symptoms": [],
            "possible_conditions": [],
            "disclaimer": (
                "No matching conditions found in database. "
                "This does not mean the symptoms are not significant. "
                "Please consult a healthcare provider for proper evaluation."
            ),
        }

    seen = set()
    unique_conditions = []
    for condition in all_conditions:
        if condition["condition"] not in seen:
            seen.add(condition["condition"])
            unique_conditions.append(condition)

    urgency_order = {"emergency": 0, "urgent": 1, "routine": 2}
    unique_conditions.sort(key=lambda c: urgency_order.get(c["urgency"], 3))

    return {
        "status": "success",
        "symptoms_analyzed": symptom_list,
        "matched_symptoms": matched_symptoms,
        "possible_conditions": unique_conditions,
        "disclaimer": (
            "This is not a diagnosis. These are possible conditions based on "
            "reported symptoms. Please consult a healthcare provider for "
            "proper evaluation and diagnosis."
        ),
    }
