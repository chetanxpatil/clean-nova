# growth/language.py
"""
Step 6: Language Expression

This module translates the GrowthMind's internal, abstract
decision (like 'merge' or 'branch') into a human-readable
statement of logic.
"""

from typing import Dict

# Defines the "voice" of the AI.
# We can make these as simple or as complex as we want.
RULE_TO_EXPRESSION_MAP: Dict[str, str] = {
    "merge": "Conclusion: The hypothesis is **entailed** by the premise.",
    "branch": "Conclusion: The hypothesis **contradicts** the premise.",
    "stabilize": "Conclusion: The relationship is **neutral**; the premise neither supports nor contradicts the hypothesis.",
    "revert": "Conclusion: Reverting to a prior state as the relationship is inconclusive.",
}

DEFAULT_EXPRESSION = "Conclusion: The logical outcome is undefined."


def express_decision(rule: str, phi: float) -> str:
    """
    Translates a chosen rule and its phi value into a
    natural language sentence.

    Args:
        rule (str): The final rule chosen by the GrowthMind
                    (e.g., 'merge', 'branch').
        phi (float): The final phi value that led to this decision.

    Returns:
        str: A human-readable explanation of the decision.
    """
    # Find the base sentence from our map
    base_sentence = RULE_TO_EXPRESSION_MAP.get(rule, DEFAULT_EXPRESSION)

    # Add the quantitative evidence (the Φ value)
    expression = f"{base_sentence} (Φ = {phi:+.3f})"

    return expression