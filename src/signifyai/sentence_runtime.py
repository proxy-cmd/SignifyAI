from __future__ import annotations


def can_auto_append_token(
    *,
    label: str,
    stable_hits: int,
    min_stable_frames: int,
    now_ts: float,
    last_append_ts: float,
    append_cooldown_sec: float,
    last_token: str | None,
) -> bool:
    if label in {"NO_HAND", "UNKNOWN"}:
        return False
    if stable_hits < max(1, int(min_stable_frames)):
        return False
    if (now_ts - last_append_ts) < max(0.05, float(append_cooldown_sec)):
        return False
    if last_token and last_token.strip().lower() == label.strip().lower():
        return False
    return True


def should_auto_speak_sentence(
    *,
    tokens_count: int,
    now_ts: float,
    last_token_ts: float,
    last_sentence_speak_ts: float,
    pause_sec: float,
) -> bool:
    if tokens_count <= 0:
        return False
    if last_token_ts <= 0.0:
        return False
    wait = max(0.25, float(pause_sec))
    if (now_ts - last_token_ts) < wait:
        return False
    if (now_ts - last_sentence_speak_ts) < wait:
        return False
    return True

