from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import cv2
import numpy as np

from ..capture.camera import CameraConfig, CameraStream
from ..contracts import PredictionOutput, SequenceWindow
from ..decoder.rules_intents import IntentHit, RuleIntentDecoder
from ..decoder.stability import StabilityConfig, StabilityFilter
from ..metrics import RollingStageMetrics, StageTimer
from ..model.registry import ModelRegistry
from ..model.sequence_model import load_runtime_model, predict_sequence_model
from ..nlp.intent_pack import intent_text
from ..perception.landmarks import MultiModalPerceptor, PerceptionConfig
from ..speech.engine import SpeechEngine


@dataclass
class RuntimeConfig:
    camera_index: int = 0
    width: int = 960
    height: int = 540
    fps: int = 30
    voice_enabled: bool = True
    seq_len: int = 24
    model_name: str | None = None


class StreamingRuntime:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.cam = CameraStream(CameraConfig(index=cfg.camera_index, width=cfg.width, height=cfg.height, fps=cfg.fps))
        self.perceptor = MultiModalPerceptor(PerceptionConfig(inference_scale=0.65))
        self.rules = RuleIntentDecoder()
        self.stability = StabilityFilter(StabilityConfig(window=7, min_confidence=0.55, hold_sec=0.10))
        self.speech = SpeechEngine(rate=185, volume=1.0)
        self.metrics = RollingStageMetrics()
        self.seq_buf: deque[np.ndarray] = deque(maxlen=max(8, cfg.seq_len))
        self.registry = ModelRegistry()
        model_name = cfg.model_name or self.registry.active()
        self.model_name = model_name or "rules_only"
        self.model = load_runtime_model(model_name) if model_name else None
        self.last_spoken_label = ""
        self.last_spoken_ts = 0.0

    def close(self) -> None:
        self.perceptor.close()
        self.speech.close()
        self.cam.close()

    def _speak_if_needed(self, label: str, conf: float, voice_enabled: bool) -> None:
        now = time.time()
        if not voice_enabled:
            return
        if label in {"unknown", "silence"}:
            return
        if conf < 0.55:
            return
        if label == self.last_spoken_label and (now - self.last_spoken_ts) < 1.8:
            return
        if (now - self.last_spoken_ts) < 0.35:
            return
        self.speech.say_latest(intent_text(label))
        self.last_spoken_label = label
        self.last_spoken_ts = now

    def step(self, voice_enabled: bool) -> tuple[np.ndarray, PredictionOutput]:
        e2e = StageTimer()

        t = StageTimer()
        ok, frame = self.cam.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        frame = cv2.flip(frame, 1)
        self.metrics.add_stage("capture", t.elapsed_ms())

        t = StageTimer()
        lm = self.perceptor.process(frame)
        self.metrics.add_stage("perception", t.elapsed_ms())

        self.seq_buf.append(SequenceWindow(frames=[lm]).to_feature_matrix().reshape(-1))

        t = StageTimer()
        rule_hit = self.rules.decode(lm)
        label = "unknown"
        conf = 0.0
        source = "rules"
        if self.model is not None and len(self.seq_buf) >= self.cfg.seq_len:
            seq = np.stack(list(self.seq_buf)[-self.cfg.seq_len :], axis=0)
            m_lbl, m_conf = predict_sequence_model(self.model, seq)
            if m_conf >= 0.60:
                label, conf, source = m_lbl, m_conf, f"model:{self.model_name}"
            elif rule_hit is not None:
                label, conf, source = rule_hit.intent_id, rule_hit.confidence, source
        elif rule_hit is not None:
            label, conf = rule_hit.intent_id, rule_hit.confidence

        stable_input: IntentHit | None = None
        if label != "unknown":
            if source == "rules" and rule_hit is not None:
                stable_input = rule_hit
            else:
                stable_input = IntentHit(intent_id=label, confidence=conf, source=source)
        stable_label, stable_conf, state = self.stability.update(stable_input)
        if stable_label not in {"unknown", "silence"}:
            label = stable_label
            conf = max(conf, stable_conf)

        self.metrics.add_stage("decode", t.elapsed_ms())

        t = StageTimer()
        self._speak_if_needed(label, conf, voice_enabled)
        self.metrics.add_stage("speech", t.elapsed_ms())

        t = StageTimer()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 88), (0, 0, 0), -1)
        cv2.putText(frame, f"Intent: {label}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
        cv2.putText(
            frame,
            f"Conf {conf:.2f} | Voice {'ON' if voice_enabled else 'OFF'} | Source {source}",
            (18, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (200, 220, 255),
            2,
        )
        self.metrics.add_stage("render", t.elapsed_ms())

        e2e_ms = e2e.elapsed_ms()
        self.metrics.add_e2e(e2e_ms)
        snapshot = self.metrics.snapshot()

        out = PredictionOutput(
            intent_id=label,
            intent_text=intent_text(label),
            confidence=float(conf),
            stability_state=state,
            timestamp_ms=lm.timestamp_ms,
            latency_ms=e2e_ms,
            source_model_version=source,
            stage_latencies_ms={
                "capture": snapshot.get("capture_median_ms", 0.0),
                "perception": snapshot.get("perception_median_ms", 0.0),
                "decode": snapshot.get("decode_median_ms", 0.0),
                "speech": snapshot.get("speech_median_ms", 0.0),
                "render": snapshot.get("render_median_ms", 0.0),
            },
            debug={"metrics": snapshot},
        )
        return frame, out

    def run(self) -> None:
        window = "SignifyAI"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, self.cfg.width, self.cfg.height)
        voice = bool(self.cfg.voice_enabled)

        try:
            while True:
                frame, pred = self.step(voice_enabled=voice)
                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("v"):
                    voice = not voice
                if key == ord("r"):
                    self.stability.reset()
                    self.last_spoken_label = ""
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if pred.debug.get("metrics"):
                    e2e = pred.debug["metrics"].get("e2e_median_ms", 0.0)
                    print(f"\r[e2e median] {e2e:6.1f} ms", end="")
        finally:
            print()
            self.close()
            cv2.destroyAllWindows()
