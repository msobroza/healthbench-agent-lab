"""Medical reference tools package.

Importing this package triggers @register_tool registration for all
medical reference tools, making them available via the tool registry.
"""

# Import tool modules to trigger @register_tool registration.
import tools.drug_reference  # noqa: F401
import tools.emergency_flag  # noqa: F401
import tools.symptom_checker  # noqa: F401
