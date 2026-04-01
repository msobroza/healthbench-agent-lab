"""Drug reference tool — curated medication information lookup.

Provides drug_reference() for looking up medication data including
dosage, interactions, and contraindications. Registered in the tool
registry as ``"drug_reference"``.

See AGENT_DECISIONS.md §4 for design rationale.
"""

from __future__ import annotations

from healthbench_agent.agent import register_tool

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


@register_tool("drug_reference")
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

    if key in _DRUG_DATABASE:
        return {"status": "success", **_DRUG_DATABASE[key]}

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
