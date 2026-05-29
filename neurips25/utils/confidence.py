import json
import math
import re
from typing import Any


def extract_choice_letters(question: str) -> list[str]:
    """Extract available MCQ choice letters (A-F) from a question string."""
    letters = re.findall(r"(?m)^([A-F])\)", question)
    if letters:
        return sorted(set(letters))
    letters = re.findall(r"\b([A-F])\)", question)
    return sorted(set(letters)) if letters else ["A", "B"]


def parse_self_reported_confidence(response: str) -> float | None:
    """
    Parse model self-reported confidence from [CONFIDENCE: X] tag.
    Accepts 0-1 float, 0-100 percentage, or qualitative labels.
    """
    matches = re.findall(r"\[CONFIDENCE:\s*([^\]]+)\]", response, re.IGNORECASE)
    if not matches:
        return None
    raw = matches[-1].strip().lower()
    qualitative = {
        "very low": 0.1, "low": 0.25, "medium": 0.5, "moderate": 0.5,
        "high": 0.75, "very high": 0.9, "certain": 0.95,
    }
    if raw in qualitative:
        return qualitative[raw]
    raw = raw.rstrip("%")
    try:
        value = float(raw)
        if value > 1.0:
            value /= 100.0
        return max(0.0, min(1.0, value))
    except ValueError:
        return None


def normalize_choice_probs(raw_probs: dict[str, float], choices: list[str]) -> dict[str, float]:
    """Normalize probabilities over available choices."""
    filtered = {c: max(0.0, raw_probs.get(c, 0.0)) for c in choices}
    total = sum(filtered.values())
    if total <= 0:
        uniform = 1.0 / len(choices)
        return {c: uniform for c in choices}
    return {c: v / total for c, v in filtered.items()}


def parse_json_choice_probs(text: str, choices: list[str]) -> dict[str, float] | None:
    """Parse JSON object mapping choice letters to probabilities."""
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            return None
        probs = {}
        for choice in choices:
            key = choice
            if key not in data and key.lower() in data:
                key = key.lower()
            if key in data:
                probs[choice] = float(data[key])
        if probs:
            return normalize_choice_probs(probs, choices)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def score_choices_with_llm(llm, conversation: list[dict], choices: list[str]) -> dict[str, float] | None:
    """
    Ask the model to assign probabilities to each choice (fallback when logprobs unavailable).
    """
    if getattr(llm, "score_choice_probs", True) is False:
        return None

    if hasattr(llm, "score_choice_probabilities"):
        try:
            return llm.score_choice_probabilities(conversation, choices)
        except Exception:
            pass

    choice_list = ", ".join(choices)
    scoring_messages = list(conversation) + [{
        "role": "user",
        "content": (
            f"Estimate the probability that each answer choice is correct. "
            f"Choices: {choice_list}. "
            f"Reply with ONLY a JSON object mapping each letter to a probability between 0 and 1, "
            f'e.g. {{"A": 0.1, "B": 0.9}}. Probabilities must sum to 1.'
        ),
    }]
    try:
        response = llm.evaluate(messages=scoring_messages)
        probs = parse_json_choice_probs(response, choices)
        return probs
    except Exception:
        return None


def build_confidence_record(
    response: str,
    selected_letter: str,
    choices: list[str],
    llm,
    conversation: list[dict],
) -> dict[str, Any]:
    """Build confidence log fields for a question answer."""
    record: dict[str, Any] = {
        "choices": choices,
        "confidence_self_reported": parse_self_reported_confidence(response),
    }

    choice_probs = score_choices_with_llm(llm, conversation, choices)
    if choice_probs:
        record["confidence_choice_probs"] = choice_probs
        record["confidence_selected_prob"] = choice_probs.get(selected_letter.upper())
        if record["confidence_selected_prob"] is not None:
            record["confidence_selected_prob"] = round(record["confidence_selected_prob"], 4)
            record["confidence_choice_probs"] = {
                k: round(v, 4) for k, v in choice_probs.items()
            }

    return record


def logprob_to_prob(logprob: float) -> float:
    return math.exp(logprob)
