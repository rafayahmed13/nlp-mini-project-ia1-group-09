"""Core Heterogeneous Deliberative Committee Aggregation (HDCA) pipeline."""

import os
import re
from typing import Dict, Optional, Tuple

from google import genai
from google.genai import types

from src.config import TEMPERATURE


def get_client() -> genai.Client:
    """Initialize Gemini client from GEMINI_API_KEY environment variable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Export it before running live LLM inference."
        )
    return genai.Client(api_key=api_key)


def call_llm(
    client: genai.Client,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = TEMPERATURE,
) -> str:
    """Execute a single Gemini inference call with system instructions."""
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        ),
    )
    return response.text


def extract_verdict(response_text: str) -> str:
    """Extract APPROVE or REJECT verdict from agent output."""
    upper = response_text.upper()
    if "APPROVE" in upper:
        return "APPROVE"
    if "REJECT" in upper:
        return "REJECT"
    return "UNKNOWN"


def extract_score(response_text: str) -> Optional[float]:
    """Extract numeric SCORE: X from agent output."""
    match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", response_text.upper())
    if match:
        return float(match.group(1))
    return None


def run_committee_verdict(
    client: genai.Client,
    model_name: str,
    agent_roles: Dict[str, str],
    case_input: str,
) -> Tuple[str, float, int, Dict]:
    """Run 3-round HDCA deliberation for binary verdict tasks."""
    round_1 = {}
    for agent_name, role_desc in agent_roles.items():
        sys_p = role_desc + "\nProvide your reasoning, ending with 'VERDICT: APPROVE' or 'VERDICT: REJECT'."
        resp = call_llm(client, model_name, sys_p, case_input)
        round_1[agent_name] = {"reasoning": resp, "verdict": extract_verdict(resp)}

    shared_context = "Here is the reasoning from all committee members in Round 1:\n\n"
    for agent_name, data in round_1.items():
        shared_context += f"--- {agent_name} ---\n{data['reasoning']}\n\n"
    shared_context += f"\nOriginal Case:\n{case_input}\n"
    shared_context += (
        "Based on your peers' arguments, reconsider the case. State if you are changing "
        "your stance, provide your revised reasoning, and end with 'VERDICT: APPROVE' or 'VERDICT: REJECT'."
    )

    round_2 = {}
    shifts = 0
    for agent_name, role_desc in agent_roles.items():
        resp = call_llm(client, model_name, role_desc, shared_context)
        verdict = extract_verdict(resp)
        round_2[agent_name] = {"reasoning": resp, "verdict": verdict}
        if verdict != round_1[agent_name]["verdict"]:
            shifts += 1

    verdicts = [v["verdict"] for v in round_2.values()]
    approve_votes = verdicts.count("APPROVE")
    reject_votes = verdicts.count("REJECT")
    decision = "APPROVE" if approve_votes > reject_votes else "REJECT"
    consensus_rate = (max(approve_votes, reject_votes) / len(agent_roles)) * 100.0
    return decision, consensus_rate, shifts, {"round_1": round_1, "round_2": round_2}


def run_committee_score(
    client: genai.Client,
    model_name: str,
    agent_roles: Dict[str, str],
    case_input: str,
) -> Tuple[Optional[float], float, int, Dict]:
    """Run 3-round HDCA deliberation for numeric rubric-scoring tasks."""
    round_1 = {}
    for agent_name, role_desc in agent_roles.items():
        resp = call_llm(client, model_name, role_desc, case_input)
        round_1[agent_name] = {"reasoning": resp, "score": extract_score(resp)}

    shared_context = "Here is the reasoning from all committee members in Round 1:\n\n"
    for agent_name, data in round_1.items():
        shared_context += f"--- {agent_name} ---\n{data['reasoning']}\n\n"
    shared_context += f"\nOriginal Case:\n{case_input}\n"
    shared_context += (
        "Based on your peers' arguments, reconsider the case. State if you are changing "
        "your score, provide your revised reasoning, and end with 'SCORE: X'."
    )

    round_2 = {}
    shifts = 0
    for agent_name, role_desc in agent_roles.items():
        resp = call_llm(client, model_name, role_desc, shared_context)
        score = extract_score(resp)
        round_2[agent_name] = {"reasoning": resp, "score": score}
        if score != round_1[agent_name]["score"]:
            shifts += 1

    valid_scores = [v["score"] for v in round_2.values() if v["score"] is not None]
    if not valid_scores:
        return None, 0.0, shifts, {"round_1": round_1, "round_2": round_2}

    avg_score = round(sum(valid_scores) / len(valid_scores), 2)
    counts = {s: valid_scores.count(s) for s in set(valid_scores)}
    consensus_rate = (max(counts.values()) / len(valid_scores)) * 100.0
    return avg_score, consensus_rate, shifts, {"round_1": round_1, "round_2": round_2}
