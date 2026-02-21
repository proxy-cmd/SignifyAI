from __future__ import annotations

from .phrase_map import get_phrase


TOKEN_MAP = {
    "i love you": "I love you",
    "thank you": "thank you",
    "call me": "call me",
    "good morning": "good morning",
    "good afternoon": "good afternoon",
    "good evening": "good evening",
    "good night": "good night",
    "yes": "yes",
    "no": "no",
    "help": "help",
    "hello": "hello",
    "stop": "stop",
    "okay": "okay",
    "one": "one",
    "two": "two",
    "peace": "peace",
    "rock": "rock",
}


def normalize_token(token: str) -> str:
    t = token.strip().replace("_", " ").lower()
    if not t:
        return ""
    custom = get_phrase(t)
    if custom:
        return custom
    if t in {"i"}:
        return "I"
    if t in TOKEN_MAP:
        return TOKEN_MAP[t]
    return t


def smooth_sentence(tokens: list[str]) -> list[str]:
    """Simple grammatical smoothing for live token streams.

    - removes immediate duplicates
    - normalizes common pronouns
    """
    out: list[str] = []
    for tok in tokens:
        n = normalize_token(tok)
        if not n:
            continue
        if out and out[-1].lower() == n.lower():
            continue
        out.append(n)
    return out


def sentence_to_text(tokens: list[str]) -> str:
    cleaned = smooth_sentence(tokens)
    if not cleaned:
        return ""

    # Tiny grammar pass for common short utterances in demos.
    if cleaned == ["yes"] or cleaned == ["no"] or cleaned == ["stop"]:
        return cleaned[0].capitalize() + "."
    if cleaned == ["help"]:
        return "Please help."
    if cleaned == ["thank you"]:
        return "Thank you."
    if cleaned == ["hello"]:
        return "Hello."
    if cleaned == ["I love you"]:
        return "I love you."

    text = " ".join(cleaned)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def speech_text_for_label(label: str) -> str:
    if label in {"NO_HAND", "UNKNOWN"}:
        return ""
    tok = normalize_token(label)
    if not tok:
        return ""
    if tok[0].islower():
        tok = tok[0].upper() + tok[1:]
    if tok[-1] not in ".!?":
        tok += "."
    return tok
