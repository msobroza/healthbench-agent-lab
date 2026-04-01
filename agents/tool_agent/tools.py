"""Medical reference tools for the tool-augmented agent (Architecture B).

Three mock tools with curated medical reference data. Each returns a dict
with a "status" field following ADK conventions. The data is intentionally
limited — the tools are demonstrations of tool-augmented agent architecture,
not a production medical database.

See AGENT_DECISIONS.md §4-6 for design rationale.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Emergency keyword tiers (Decision 5 — rule-based with confidence tiers)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Drug reference database (Decision 4 — plain dict with status field)
# ---------------------------------------------------------------------------

_DRUG_DATABASE: dict[str, dict] = {
    "metformin": {
        "generic_name": "metformin",
        "brand_names": ["Glucophage", "Fortamet", "Glumetza"],
        "drug_class": "biguanide (antidiabetic)",
        "common_dosage": "500-2000 mg daily in divided doses",
        "indications": ["type 2 diabetes mellitus"],
        "contraindications": [
            "severe renal impairment (eGFR <30 mL/min)",
            "metabolic acidosis",
            "diabetic ketoacidosis",
        ],
        "common_side_effects": ["nausea", "diarrhea", "abdominal discomfort", "metallic taste"],
        "serious_side_effects": ["lactic acidosis (rare but serious)"],
        "interactions": [
            "alcohol (increased risk of lactic acidosis)",
            "iodinated contrast agents (hold 48h before/after)",
            "carbonic anhydrase inhibitors",
        ],
    },
    "lisinopril": {
        "generic_name": "lisinopril",
        "brand_names": ["Prinivil", "Zestril"],
        "drug_class": "ACE inhibitor (antihypertensive)",
        "common_dosage": "10-40 mg once daily",
        "indications": ["hypertension", "heart failure", "post-myocardial infarction"],
        "contraindications": [
            "history of angioedema",
            "pregnancy",
            "bilateral renal artery stenosis",
        ],
        "common_side_effects": ["dry cough", "dizziness", "headache", "hyperkalemia"],
        "serious_side_effects": ["angioedema", "renal failure", "hyperkalemia"],
        "interactions": [
            "potassium supplements (hyperkalemia risk)",
            "NSAIDs (reduced antihypertensive effect)",
            "lithium (increased lithium levels)",
        ],
    },
    "ibuprofen": {
        "generic_name": "ibuprofen",
        "brand_names": ["Advil", "Motrin"],
        "drug_class": "NSAID (nonsteroidal anti-inflammatory drug)",
        "common_dosage": "200-800 mg every 4-6 hours, max 3200 mg/day",
        "indications": ["pain", "inflammation", "fever", "dysmenorrhea"],
        "contraindications": [
            "active GI bleeding",
            "severe renal impairment",
            "third trimester pregnancy",
            "CABG perioperative period",
        ],
        "common_side_effects": ["nausea", "dyspepsia", "GI discomfort", "headache"],
        "serious_side_effects": [
            "GI bleeding",
            "cardiovascular events",
            "renal impairment",
            "Stevens-Johnson syndrome (rare)",
        ],
        "interactions": [
            "anticoagulants (increased bleeding risk)",
            "ACE inhibitors (reduced effect)",
            "lithium (increased lithium levels)",
            "methotrexate (increased toxicity)",
        ],
    },
    "amoxicillin": {
        "generic_name": "amoxicillin",
        "brand_names": ["Amoxil", "Trimox"],
        "drug_class": "penicillin antibiotic",
        "common_dosage": "250-500 mg every 8 hours or 500-875 mg every 12 hours",
        "indications": [
            "bacterial infections",
            "otitis media",
            "sinusitis",
            "urinary tract infection",
            "H. pylori (with other agents)",
        ],
        "contraindications": ["penicillin allergy", "severe hypersensitivity to beta-lactams"],
        "common_side_effects": ["diarrhea", "nausea", "rash", "vomiting"],
        "serious_side_effects": [
            "anaphylaxis",
            "Clostridioides difficile colitis",
            "Stevens-Johnson syndrome (rare)",
        ],
        "interactions": [
            "warfarin (increased INR monitoring needed)",
            "methotrexate (increased toxicity)",
            "oral contraceptives (possibly reduced efficacy)",
        ],
    },
    "atorvastatin": {
        "generic_name": "atorvastatin",
        "brand_names": ["Lipitor"],
        "drug_class": "HMG-CoA reductase inhibitor (statin)",
        "common_dosage": "10-80 mg once daily",
        "indications": [
            "hyperlipidemia",
            "cardiovascular risk reduction",
            "familial hypercholesterolemia",
        ],
        "contraindications": [
            "active liver disease",
            "unexplained persistent transaminase elevation",
            "pregnancy",
            "breastfeeding",
        ],
        "common_side_effects": ["myalgia", "arthralgia", "diarrhea", "nasopharyngitis"],
        "serious_side_effects": [
            "rhabdomyolysis",
            "hepatotoxicity",
            "immune-mediated necrotizing myopathy",
        ],
        "interactions": [
            "cyclosporine (max 10 mg atorvastatin)",
            "strong CYP3A4 inhibitors (clarithromycin, itraconazole)",
            "gemfibrozil (increased myopathy risk)",
            "grapefruit juice (large quantities)",
        ],
    },
    "levothyroxine": {
        "generic_name": "levothyroxine",
        "brand_names": ["Synthroid", "Levoxyl", "Tirosint"],
        "drug_class": "thyroid hormone replacement",
        "common_dosage": "25-200 mcg once daily on empty stomach",
        "indications": ["hypothyroidism", "thyroid cancer (TSH suppression)"],
        "contraindications": [
            "uncorrected adrenal insufficiency",
            "acute myocardial infarction (relative)",
        ],
        "common_side_effects": [
            "dose-dependent: tachycardia, tremor, weight loss, insomnia (signs of overreplacement)"
        ],
        "serious_side_effects": [
            "cardiac arrhythmias (if over-replaced)",
            "bone density loss (long-term overreplacement)",
        ],
        "interactions": [
            "calcium supplements (separate by 4 hours)",
            "iron supplements (separate by 4 hours)",
            "proton pump inhibitors (may reduce absorption)",
            "warfarin (may increase anticoagulant effect)",
        ],
    },
}

# ---------------------------------------------------------------------------
# Symptom database (Decision 6 — dict lookup with curated data)
# ---------------------------------------------------------------------------

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
                "Sudden 'thunderclap' worst-ever headache; "
                "requires immediate evaluation."
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
                "Right lower quadrant pain, fever, guarding; "
                "requires surgical evaluation."
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


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def drug_reference(drug_name: str) -> dict:
    """Look up drug information including dosage, interactions, and contraindications.

    Args:
        drug_name: The name of the drug to look up (generic or brand name).

    Returns:
        dict with drug information including dosage, interactions,
        contraindications, and side effects. Returns an error dict
        if the drug is not found in the reference database.
    """
    key = drug_name.lower().strip()

    # Check by generic name first
    if key in _DRUG_DATABASE:
        return {"status": "success", **_DRUG_DATABASE[key]}

    # Check by brand name
    for drug_data in _DRUG_DATABASE.values():
        if key in [b.lower() for b in drug_data["brand_names"]]:
            return {"status": "success", **drug_data}

    return {
        "status": "error",
        "error_message": (
            f"Drug '{drug_name}' not found in reference database. "
            "Recommend consulting a pharmacist or official drug reference."
        ),
    }


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

    # Deduplicate by condition name
    seen = set()
    unique_conditions = []
    for condition in all_conditions:
        if condition["condition"] not in seen:
            seen.add(condition["condition"])
            unique_conditions.append(condition)

    # Sort by urgency: emergency > urgent > routine
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
