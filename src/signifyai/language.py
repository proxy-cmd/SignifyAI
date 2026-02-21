from __future__ import annotations


def normalize_token(token: str) -> str:
    t = token.strip().lower()
    if not t:
        return ""
    if t in {"i"}:
        return "I"
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

    text = " ".join(cleaned)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text
