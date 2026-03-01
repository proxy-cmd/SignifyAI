from __future__ import annotations

from dataclasses import dataclass
import time

from .language import sentence_to_text


@dataclass
class SentenceDecoderConfig:
    min_stable_frames: int = 1
    append_cooldown_sec: float = 0.35
    pause_speak_sec: float = 1.0
    max_tokens: int = 14
    no_hand_flush_frames: int = 3


@dataclass
class SentenceDecoderUpdate:
    appended: bool = False
    auto_spoken_text: str = ""


def _canon(token: str) -> str:
    return token.strip().replace("_", " ").lower()


class SentenceDecoder:
    """
    Stateful decoder for rapid sign streams.

    Responsibilities:
    - append stable labels with cooldown and duplicate suppression
    - collapse common short phrase patterns
    - auto-speak sentence after a pause/no-hand boundary
    """

    _PHRASE_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
        (("i", "am"), "i am"),
        (("thank", "you"), "thank you"),
        (("good", "morning"), "good morning"),
        (("good", "afternoon"), "good afternoon"),
        (("good", "evening"), "good evening"),
        (("good", "night"), "good night"),
        (("what", "is", "your", "name"), "what is your name"),
        (("how", "are", "you"), "how are you"),
    )

    def __init__(self, cfg: SentenceDecoderConfig | None = None) -> None:
        self.cfg = cfg or SentenceDecoderConfig()
        self.tokens: list[str] = []
        self.last_append_ts = 0.0
        self.last_token_ts = 0.0
        self.last_sentence_speak_ts = 0.0
        self.no_hand_streak = 0

    def clear(self) -> None:
        self.tokens.clear()
        self.no_hand_streak = 0

    def text(self) -> str:
        return sentence_to_text(self.tokens)

    def append_manual(self, label: str, now_ts: float | None = None) -> bool:
        ts = time.time() if now_ts is None else now_ts
        return self._append_token(label=label, now_ts=ts, force=True)

    def speak_now(self, now_ts: float | None = None) -> str:
        ts = time.time() if now_ts is None else now_ts
        spoken = sentence_to_text(self.tokens)
        if spoken:
            self.tokens.clear()
            self.no_hand_streak = 0
            self.last_sentence_speak_ts = ts
        return spoken

    def update(
        self,
        *,
        label: str,
        stable_hits: int,
        hand_count: int,
        now_ts: float,
        auto_speak_enabled: bool,
    ) -> SentenceDecoderUpdate:
        if hand_count <= 0 or label == "NO_HAND":
            self.no_hand_streak += 1
        else:
            self.no_hand_streak = 0

        appended = False
        if (
            label not in {"NO_HAND", "UNKNOWN"}
            and stable_hits >= max(1, int(self.cfg.min_stable_frames))
        ):
            appended = self._append_token(label=label, now_ts=now_ts, force=False)

        auto_spoken_text = ""
        if auto_speak_enabled and self.tokens:
            wait = max(0.25, float(self.cfg.pause_speak_sec))
            pause_ready = (now_ts - self.last_token_ts) >= wait
            speak_cooldown_ok = (now_ts - self.last_sentence_speak_ts) >= wait
            boundary_ready = self.no_hand_streak >= max(1, int(self.cfg.no_hand_flush_frames))
            if speak_cooldown_ok and (pause_ready or boundary_ready):
                auto_spoken_text = self.speak_now(now_ts=now_ts)

        return SentenceDecoderUpdate(appended=appended, auto_spoken_text=auto_spoken_text)

    def _append_token(self, *, label: str, now_ts: float, force: bool) -> bool:
        token = label.strip()
        if not token:
            return False
        if token in {"NO_HAND", "UNKNOWN"}:
            return False

        if not force:
            cooldown = max(0.05, float(self.cfg.append_cooldown_sec))
            if (now_ts - self.last_append_ts) < cooldown:
                return False
            if self.tokens and _canon(self.tokens[-1]) == _canon(token):
                return False

        self.tokens.append(token)
        if len(self.tokens) > max(1, int(self.cfg.max_tokens)):
            self.tokens = self.tokens[-int(self.cfg.max_tokens) :]

        self.last_append_ts = now_ts
        self.last_token_ts = now_ts
        self._collapse_phrases()
        return True

    def _collapse_phrases(self) -> None:
        # Repeatedly collapse known phrase tails (rapid-sign friendly).
        for _ in range(3):
            changed = False
            canon_tokens = [_canon(t) for t in self.tokens]
            for pattern, merged in self._PHRASE_PATTERNS:
                size = len(pattern)
                if len(canon_tokens) < size:
                    continue
                if tuple(canon_tokens[-size:]) == pattern:
                    self.tokens = self.tokens[:-size] + [merged.upper().replace(" ", "_")]
                    changed = True
                    break
            if not changed:
                break
