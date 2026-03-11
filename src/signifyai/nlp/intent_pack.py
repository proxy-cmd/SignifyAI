from __future__ import annotations

INTENT_PACK: dict[str, str] = {
    "hospital_help": "I need medical help.",
    "need_water": "I need water.",
    "need_food": "I need food.",
    "need_toilet": "I need to use the toilet.",
    "call_family": "Please call my family.",
    "emergency": "This is an emergency.",
    "thank_you": "Thank you.",
    "yes": "Yes.",
    "no": "No.",
    "hello": "Hello.",
}


def intent_text(intent_id: str) -> str:
    return INTENT_PACK.get(intent_id, intent_id.replace("_", " ").title())
