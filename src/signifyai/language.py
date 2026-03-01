from __future__ import annotations

import re

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
    "what": "what",
    "is": "is",
    "your": "your",
    "name": "name",
    "my": "my",
    "i am": "I am",
    "im": "I'm",
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
        parts = [p for p in n.split() if p.strip()]
        for part in parts:
            if out and out[-1].lower() == part.lower():
                continue
            out.append(part)
    return out


def _normalize_words_for_output(words: list[str]) -> list[str]:
    out: list[str] = []
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in {"i", "im", "i'm"}:
            out.append("I" if lw == "i" else "I'm")
            continue
        if i == 0:
            out.append(w.capitalize() if w and w[0].islower() else w)
        else:
            out.append(w)
    return out


def _apply_small_grammar(words: list[str]) -> list[str]:
    out = words[:]
    if len(out) >= 5:
        # "hello i am proxy" -> "Hello, I am proxy"
        if out[0].lower() == "hello" and out[1].lower() == "i" and out[2].lower() == "am":
            out[0] = "Hello,"
    # normalize accidental "i am" split style to proper case
    for i in range(len(out) - 1):
        if out[i].lower() == "i" and out[i + 1].lower() == "am":
            out[i] = "I"
    return out


def _is_question(words: list[str]) -> bool:
    lower = [w.lower().strip(",.!?") for w in words]
    if not lower:
        return False
    q_words = {"what", "who", "where", "when", "why", "how"}
    if any(w in q_words for w in lower):
        return True
    if "name" in lower and ("your" in lower or "you" in lower):
        return True
    return False


def sentence_to_text(tokens: list[str]) -> str:
    words = smooth_sentence(tokens)
    if not words:
        return ""

    # Tiny grammar pass for common short utterances in demos.
    if words == ["yes"] or words == ["no"] or words == ["stop"]:
        return words[0].capitalize() + "."
    if words == ["help"]:
        return "Please help."
    if words == ["thank", "you"] or words == ["thank you"]:
        return "Thank you."
    if words == ["hello"]:
        return "Hello."
    if words == ["I", "love", "you"] or words == ["i", "love", "you"]:
        return "I love you."

    words = _normalize_words_for_output(words)
    words = _apply_small_grammar(words)
    text = " ".join(words)
    text = re.sub(r"\s+([,!?\.])", r"\1", text)
    if text:
        if _is_question(words):
            if text[-1] not in "?!":
                text += "?"
        elif text[-1] not in ".!?":
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
