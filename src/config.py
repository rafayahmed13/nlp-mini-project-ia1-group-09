"""Configuration and deterministic seed utilities for HDCA."""

import os
import random
import numpy as np

SEED = 42
TEMPERATURE = 0.2
LOAN_MODEL_NAME = "gemini-3.5-flash-lite"
EXAM_MODEL_NAME = "gemini-3.5-flash"

LOAN_AGENT_ROLES = {
    "Agent_A_Cautious": (
        "You are a cautious, risk-averse decision-maker. "
        "You prioritize uncertainty, downside exposure, and missing evidence."
    ),
    "Agent_B_Neutral": (
        "You are a neutral, rule-based decision-maker. "
        "You follow stated criteria and apply thresholds consistently."
    ),
    "Agent_C_Lenient": (
        "You are a lenient, context-sensitive decision-maker. "
        "You weigh mitigating circumstances and qualitative context."
    ),
}

EXAM_AGENT_ROLES = {
    "Agent_A_Cautious": (
        "You are a cautious, strict grader. You prioritize technical accuracy, "
        "penalize missing evidence heavily, and look for gaps in reasoning."
    ),
    "Agent_B_Neutral": (
        "You are a neutral, rule-based grader. You follow the stated rubric explicitly "
        "and apply point thresholds mechanically without personal bias."
    ),
    "Agent_C_Lenient": (
        "You are a lenient, context-sensitive grader. You reward partial understanding, "
        "underlying intuition, and student effort even if minor details are missing."
    ),
}


def set_seed(seed: int = SEED) -> None:
    """Fix random seeds across Python and NumPy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
