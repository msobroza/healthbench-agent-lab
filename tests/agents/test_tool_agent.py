"""Tests for agents/tool_agent/ — Architecture B.

Tests tool functions (drug_reference, symptom_checker, emergency_flag),
prompt loading, tool registry, and the root_agent configuration.
Does NOT test LLM inference (no network calls).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agents.tool_pipeline import _pipeline, root_agent
from healthbench_agent.agent import (
    AgentPipeline,
    RootAgentPipelineConfig,
    get_tools,
    load_instruction,
    registered_tools,
)
from healthbench_agent.agent.adapters.adk_adapter import ADKAgentPipeline
from tools.drug_reference import _DRUG_DATABASE, drug_reference
from tools.emergency_flag import (
    EMERGENCY_KEYWORDS,
    URGENT_KEYWORDS,
    emergency_flag,
)
from tools.symptom_checker import _SYMPTOM_DATABASE, symptom_checker

_PROMPT_PATH = Path(_pipeline.config.prompt_path)


# ---------------------------------------------------------------------------
# Prompt YAML tests
# ---------------------------------------------------------------------------


class TestClinicalPromptYaml:
    """Tests for prompts/v1_clinical.yaml structure and content."""

    def test_prompt_file_exists(self):
        assert _PROMPT_PATH.exists()

    def test_prompt_yaml_has_required_keys(self):
        with open(_PROMPT_PATH) as f:
            data = yaml.safe_load(f)
        assert "version" in data
        assert "instruction" in data
        assert "rationale" in data

    def test_prompt_instruction_references_tools(self):
        instruction = load_instruction(_PROMPT_PATH)
        assert "drug_reference" in instruction
        assert "symptom_checker" in instruction
        assert "emergency_flag" in instruction

    def test_prompt_instruction_contains_conversation_placeholder(self):
        with open(_PROMPT_PATH) as f:
            raw = yaml.safe_load(f)["instruction"]
        assert "{{ conversation }}" in raw


# ---------------------------------------------------------------------------
# Tool Registry tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """Tests for the tool registry integration."""

    def test_all_tools_registered(self):
        registry = registered_tools()
        assert "drug_reference" in registry
        assert "symptom_checker" in registry
        assert "emergency_flag" in registry

    def test_get_tools_resolves_by_name(self):
        tools = get_tools(["drug_reference", "symptom_checker", "emergency_flag"])
        assert len(tools) == 3
        assert tools[0] is drug_reference
        assert tools[1] is symptom_checker
        assert tools[2] is emergency_flag

    def test_get_tools_unknown_name_raises(self):
        with pytest.raises(KeyError, match="nonexistent"):
            get_tools(["nonexistent"])


# ---------------------------------------------------------------------------
# Tool Agent configuration tests
# ---------------------------------------------------------------------------


class TestToolAgentConfig:
    """Tests for the root_agent exported by the tool agent module."""

    def test_root_agent_name(self):
        assert root_agent.name == "tool_agent"

    def test_root_agent_model(self):
        assert root_agent.model == "gemini-2.5-flash"

    def test_root_agent_has_three_tools(self):
        assert len(root_agent.tools) == 3

    def test_root_agent_has_no_sub_agents(self):
        assert not root_agent.sub_agents

    def test_config_tools_list(self):
        assert _pipeline.config.tools == [
            "drug_reference",
            "symptom_checker",
            "emergency_flag",
        ]


# ---------------------------------------------------------------------------
# Pipeline class tests
# ---------------------------------------------------------------------------


class TestToolPipeline:
    """Tests for the tool ADKAgentPipeline concrete class."""

    def test_is_agent_pipeline_subclass(self):
        assert isinstance(_pipeline, AgentPipeline)

    def test_config_is_tool_agent_pipeline_config(self):
        assert isinstance(_pipeline.config, RootAgentPipelineConfig)

    def test_from_config_creates_pipeline(self):
        pipeline = ADKAgentPipeline.from_config("config/agents/tool_agent.yaml")
        assert isinstance(pipeline, ADKAgentPipeline)
        assert pipeline.agent.name == "tool_agent"

    def test_description_from_config(self):
        assert _pipeline.config.description == root_agent.description


# ---------------------------------------------------------------------------
# drug_reference() tests — ZOMBIES
# ---------------------------------------------------------------------------


class TestDrugReference:
    """Tests for the drug_reference() tool function."""

    def test_empty_drug_name_returns_error(self):
        result = drug_reference("")
        assert result["status"] == "error"

    def test_known_generic_drug_returns_success(self):
        result = drug_reference("metformin")
        assert result["status"] == "success"
        assert result["generic_name"] == "metformin"
        assert "common_dosage" in result
        assert "contraindications" in result
        assert "interactions" in result

    def test_known_brand_name_returns_success(self):
        result = drug_reference("Advil")
        assert result["status"] == "success"
        assert result["generic_name"] == "ibuprofen"

    @pytest.mark.parametrize("drug_name", list(_DRUG_DATABASE.keys()))
    def test_all_database_drugs_return_success(self, drug_name):
        result = drug_reference(drug_name)
        assert result["status"] == "success"

    def test_case_insensitive_lookup(self):
        result = drug_reference("METFORMIN")
        assert result["status"] == "success"

    def test_whitespace_stripped(self):
        result = drug_reference("  lisinopril  ")
        assert result["status"] == "success"

    def test_unknown_drug_returns_error(self):
        result = drug_reference("nonexistent_drug_xyz")
        assert result["status"] == "error"
        assert "error_message" in result
        assert "nonexistent_drug_xyz" in result["error_message"]

    def test_response_has_required_fields(self):
        result = drug_reference("atorvastatin")
        assert result["status"] == "success"
        required_fields = [
            "generic_name",
            "brand_names",
            "drug_class",
            "common_dosage",
            "indications",
            "contraindications",
            "common_side_effects",
            "serious_side_effects",
            "interactions",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_brand_name_lookup_lipitor(self):
        result = drug_reference("Lipitor")
        assert result["status"] == "success"
        assert result["generic_name"] == "atorvastatin"

    def test_brand_name_lookup_synthroid(self):
        result = drug_reference("Synthroid")
        assert result["status"] == "success"
        assert result["generic_name"] == "levothyroxine"


# ---------------------------------------------------------------------------
# symptom_checker() tests — ZOMBIES
# ---------------------------------------------------------------------------


class TestSymptomChecker:
    """Tests for the symptom_checker() tool function."""

    def test_unrecognized_symptoms_returns_empty_conditions(self):
        result = symptom_checker("blue fingernails, itchy elbow")
        assert result["status"] == "success"
        assert result["possible_conditions"] == []
        assert "disclaimer" in result

    def test_single_known_symptom_returns_conditions(self):
        result = symptom_checker("chest pain")
        assert result["status"] == "success"
        assert len(result["possible_conditions"]) > 0
        assert "chest pain" in result["matched_symptoms"]

    def test_multiple_symptoms_returns_combined_conditions(self):
        result = symptom_checker("headache, fever")
        assert result["status"] == "success"
        assert len(result["possible_conditions"]) > 2

    def test_conditions_sorted_by_urgency(self):
        result = symptom_checker("chest pain")
        urgencies = [c["urgency"] for c in result["possible_conditions"]]
        urgency_order = {"emergency": 0, "urgent": 1, "routine": 2}
        ordered = sorted(urgencies, key=lambda u: urgency_order.get(u, 3))
        assert urgencies == ordered

    def test_no_duplicate_conditions(self):
        result = symptom_checker("headache, headache, headache")
        condition_names = [c["condition"] for c in result["possible_conditions"]]
        assert len(condition_names) == len(set(condition_names))

    def test_response_has_required_keys(self):
        result = symptom_checker("fatigue")
        assert "status" in result
        assert "symptoms_analyzed" in result
        assert "matched_symptoms" in result
        assert "possible_conditions" in result
        assert "disclaimer" in result

    def test_disclaimer_always_present(self):
        result = symptom_checker("fatigue")
        disclaimer = result["disclaimer"].lower()
        assert "not a diagnosis" in disclaimer or "consult" in disclaimer

    def test_symptoms_parsed_as_list(self):
        result = symptom_checker("headache, fever, fatigue")
        assert result["symptoms_analyzed"] == ["headache", "fever", "fatigue"]

    @pytest.mark.parametrize("symptom", list(_SYMPTOM_DATABASE.keys()))
    def test_all_database_symptoms_return_conditions(self, symptom):
        result = symptom_checker(symptom)
        assert result["status"] == "success"
        assert len(result["possible_conditions"]) > 0


# ---------------------------------------------------------------------------
# emergency_flag() tests — ZOMBIES
# ---------------------------------------------------------------------------


class TestEmergencyFlag:
    """Tests for the emergency_flag() tool function."""

    def test_empty_description_returns_routine(self):
        result = emergency_flag("")
        assert result["status"] == "success"
        assert result["urgency"] == "routine"

    def test_single_emergency_keyword_returns_emergency(self):
        result = emergency_flag("I have severe chest pain")
        assert result["status"] == "success"
        assert result["urgency"] == "emergency"
        assert "chest pain" in result["matched_keywords"]

    def test_single_urgent_keyword_returns_urgent(self):
        result = emergency_flag("I have a very high fever")
        assert result["status"] == "success"
        assert result["urgency"] == "urgent"

    def test_multiple_emergency_keywords_high_confidence(self):
        result = emergency_flag("chest pain and difficulty breathing and seizure")
        assert result["urgency"] == "emergency"
        assert result["confidence"] > 0.7
        assert len(result["matched_keywords"]) >= 3

    def test_confidence_capped_at_one(self):
        all_keywords = " ".join(EMERGENCY_KEYWORDS)
        result = emergency_flag(all_keywords)
        assert result["confidence"] <= 1.0

    def test_emergency_overrides_urgent(self):
        result = emergency_flag("chest pain and high fever")
        assert result["urgency"] == "emergency"

    def test_routine_has_empty_matched_keywords(self):
        result = emergency_flag("I have a mild cold and runny nose")
        assert result["urgency"] == "routine"
        assert result["matched_keywords"] == []

    def test_response_has_required_keys(self):
        result = emergency_flag("some description")
        required_keys = ["status", "urgency", "confidence", "matched_keywords", "recommendation"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    @pytest.mark.parametrize(
        "description,expected_urgency",
        [
            ("chest pain", "emergency"),
            ("high fever", "urgent"),
            ("mild headache", "routine"),
        ],
    )
    def test_urgency_values_are_valid(self, description, expected_urgency):
        result = emergency_flag(description)
        assert result["urgency"] == expected_urgency

    def test_case_insensitive_matching(self):
        result = emergency_flag("CHEST PAIN AND DIFFICULTY BREATHING")
        assert result["urgency"] == "emergency"

    @pytest.mark.parametrize(
        "description",
        ["chest pain", "high fever", "mild cough"],
    )
    def test_recommendation_always_present(self, description):
        result = emergency_flag(description)
        assert len(result["recommendation"]) > 0

    @pytest.mark.parametrize(
        "emergency_scenario",
        [
            "suicidal thoughts",
            "difficulty breathing",
            "cardiac arrest",
            "stroke symptoms",
            "severe bleeding",
            "anaphylaxis",
            "overdose",
            "seizure",
        ],
    )
    def test_critical_emergencies_detected(self, emergency_scenario):
        result = emergency_flag(emergency_scenario)
        assert result["urgency"] == "emergency", f"Failed to detect emergency: {emergency_scenario}"

    def test_global_health_emergency_detected(self):
        result = emergency_flag("patient with severe bleeding from tropical disease")
        assert result["urgency"] == "emergency"

    def test_context_seeking_emergency_detected(self):
        result = emergency_flag("my baby is not breathing and turning blue")
        assert result["urgency"] == "emergency"


# ---------------------------------------------------------------------------
# Keyword database quality tests
# ---------------------------------------------------------------------------


class TestKeywordDatabases:
    """Validate keyword lists are well-formed and non-overlapping."""

    def test_emergency_keywords_are_lowercase(self):
        for keyword in EMERGENCY_KEYWORDS:
            assert keyword == keyword.lower()

    def test_urgent_keywords_are_lowercase(self):
        for keyword in URGENT_KEYWORDS:
            assert keyword == keyword.lower()

    def test_no_duplicate_emergency_keywords(self):
        assert len(EMERGENCY_KEYWORDS) == len(set(EMERGENCY_KEYWORDS))

    def test_no_duplicate_urgent_keywords(self):
        assert len(URGENT_KEYWORDS) == len(set(URGENT_KEYWORDS))

    def test_no_overlap_between_emergency_and_urgent(self):
        overlap = set(EMERGENCY_KEYWORDS) & set(URGENT_KEYWORDS)
        assert not overlap, f"Overlapping keywords: {overlap}"
