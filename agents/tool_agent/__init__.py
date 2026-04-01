"""Tool-augmented agent package — Architecture B.

Importing this package triggers @register_tool registration for all
medical reference tools, making them available via the tool registry.
"""

# Import tool modules to trigger @register_tool registration.
import agents.tool_agent.drug_reference  # noqa: F401
import agents.tool_agent.emergency_flag  # noqa: F401
import agents.tool_agent.symptom_checker  # noqa: F401
