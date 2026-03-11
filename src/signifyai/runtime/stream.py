from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import cv2
import numpy as np

from ..capture.camera import CameraConfig, CameraStream
from ..contracts import LandmarkFrame, PredictionOutput, SequenceWindow
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
    """Main realtime loop for continuous sign-to-intent translation."""

    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg

        cam_cfg = CameraConfig(index=cfg.camera_index, width=cfg.width, height=cfg.height, fps=cfg.fps)
        self.camera = CameraStream(cam_cfg)
        self.perceptor = MultiModalPerceptor(PerceptionConfig(inference_scale=0.65))
        self.rule_decoder = RuleIntentDecoder()
        self.stability = StabilityFilter(StabilityConfig(window=7, min_confidence=0.55, hold_sec=0.10))
        self.speech = SpeechEngine(rate=185, volume=1.0)
        self.metrics = RollingStageMetrics()

        self.sequence_buffer: deque[np.ndarray] = deque(maxlen=max(8, cfg.seq_len))
        self.registry = ModelRegistry()
        active_model_name = cfg.model_name or self.registry.active()
        self.model_name = active_model_name or "rules_only"
        self.model = load_runtime_model(active_model_name) if active_model_name else None

        self.last_spoken_label = ""
        self.last_spoken_ts = 0.0

    def close(self) -> None:
        self.perceptor.close()
        self.speech.close()
        self.camera.close()

    def step(self, voice_enabled: bool) -> tuple[np.ndarray, PredictionOutput]:
        end_to_end_timer = StageTimer()

        frame = self.capture_frame()
        landmark_frame = self.run_perception(frame)
        self.append_sequence_feature(landmark_frame)

        raw_label, raw_confidence, source, rule_hit = self.run_inference(landmark_frame)
        stable_label, stable_confidence, stability_state = self.apply_stability(
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            source=source,
            rule_hit=rule_hit,
        )

        self.run_speech(stable_label, stable_confidence, voice_enabled)
        self.render_overlay(frame, stable_label, stable_confidence, source, voice_enabled)

        output = self.build_prediction_output(
            label=stable_label,
            confidence=stable_confidence,
            source=source,
            timestamp_ms=landmark_frame.timestamp_ms,
            stability_state=stability_state,
            end_to_end_ms=end_to_end_timer.elapsed_ms(),
        )
        return frame, output

    def run(self) -> None:
        window = "SignifyAI"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, self.cfg.width, self.cfg.height)

        voice_enabled = bool(self.cfg.voice_enabled)
        try:
            while True:
                frame, prediction = self.step(voice_enabled=voice_enabled)
                cv2.imshow(window, frame)
                self.print_latency(prediction)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("v"):
                    voice_enabled = not voice_enabled
                if key == ord("r"):
                    self.reset_runtime_state()

                is_hidden = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
                if is_hidden:
                    break
        finally:
            print()
            self.close()
            cv2.destroyAllWindows()

    def capture_frame(self) -> np.ndarray:
        timer = StageTimer()
        ok, frame = self.camera.read()
        if not ok or frame is None:
            raise RuntimeError("Camera read failed")
        frame = cv2.flip(frame, 1)
        self.metrics.add_stage("capture", timer.elapsed_ms())
        return frame

    def run_perception(self, frame: np.ndarray) -> LandmarkFrame:
        timer = StageTimer()
        landmark_frame = self.perceptor.process(frame)
        self.metrics.add_stage("perception", timer.elapsed_ms())
        return landmark_frame

    def append_sequence_feature(self, landmark_frame: LandmarkFrame) -> None:
        vector = SequenceWindow(frames=[landmark_frame]).to_feature_matrix().reshape(-1)
        self.sequence_buffer.append(vector)

    def run_inference(self, landmark_frame: LandmarkFrame) -> tuple[str, float, str, IntentHit | None]:
        timer = StageTimer()

        rule_hit = self.rule_decoder.decode(landmark_frame)
        label = "unknown"
        confidence = 0.0
        source = "rules"

        if self.model is not None and len(self.sequence_buffer) >= self.cfg.seq_len:
            sequence = np.stack(list(self.sequence_buffer)[-self.cfg.seq_len :], axis=0)
            model_label, model_confidence = predict_sequence_model(self.model, sequence)
            if model_confidence >= 0.60:
                label = model_label
                confidence = model_confidence
                source = f"model:{self.model_name}"
            elif rule_hit is not None:
                label = rule_hit.intent_id
                confidence = rule_hit.confidence
        elif rule_hit is not None:
            label = rule_hit.intent_id
            confidence = rule_hit.confidence

        self.metrics.add_stage("decode", timer.elapsed_ms())
        return label, confidence, source, rule_hit

    def apply_stability(
        self,
        raw_label: str,
        raw_confidence: float,
        source: str,
        rule_hit: IntentHit | None,
    ) -> tuple[str, float, str]:
        stable_input: IntentHit | None = None
        if raw_label != "unknown":
            if source == "rules" and rule_hit is not None:
                stable_input = rule_hit
            else:
                stable_input = IntentHit(intent_id=raw_label, confidence=raw_confidence, source=source)

        stable_label, stable_conf, state = self.stability.update(stable_input)
        if stable_label in {"unknown", "silence"}:
            return raw_label, raw_confidence, state

        return stable_label, max(raw_confidence, stable_conf), state

    def run_speech(self, label: str, confidence: float, voice_enabled: bool) -> None:
        timer = StageTimer()
        self.speak_if_needed(label=label, confidence=confidence, voice_enabled=voice_enabled)
        self.metrics.add_stage("speech", timer.elapsed_ms())

    def speak_if_needed(self, label: str, confidence: float, voice_enabled: bool) -> None:
        if not voice_enabled:
            return
        if label in {"unknown", "silence"}:
            return
        if confidence < 0.55:
            return

        now = time.time()
        if label == self.last_spoken_label and (now - self.last_spoken_ts) < 1.8:
            return
        if (now - self.last_spoken_ts) < 0.35:
            return

        self.speech.say_latest(intent_text(label))
        self.last_spoken_label = label
        self.last_spoken_ts = now

    def render_overlay(self, frame: np.ndarray, label: str, confidence: float, source: str, voice_enabled: bool) -> None:
        timer = StageTimer()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 88), (0, 0, 0), -1)
        line1 = f"Intent: {label}"
        line2 = f"Conf {confidence:.2f} | Voice {'ON' if voice_enabled else 'OFF'} | Source {source}"
        cv2.putText(frame, line1, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
        cv2.putText(
            frame,
            line2,
            (18, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (200, 220, 255),
            2,
        )
        self.metrics.add_stage("render", timer.elapsed_ms())

    def build_prediction_output(
        self,
        label: str,
        confidence: float,
        source: str,
        timestamp_ms: int,
        stability_state: str,
        end_to_end_ms: float,
    ) -> PredictionOutput:
        self.metrics.add_e2e(end_to_end_ms)
        snapshot = self.metrics.snapshot()

        return PredictionOutput(
            intent_id=label,
            intent_text=intent_text(label),
            confidence=float(confidence),
            stability_state=stability_state,
            timestamp_ms=timestamp_ms,
            latency_ms=end_to_end_ms,
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

    @staticmethod
    def print_latency(prediction: PredictionOutput) -> None:
        metrics = prediction.debug.get("metrics", {})
        if not metrics:
            return
        e2e = metrics.get("e2e_median_ms", 0.0)
        print(f"\r[e2e median] {e2e:6.1f} ms", end="")

    def reset_runtime_state(self) -> None:
        self.stability.reset()
        self.last_spoken_label = ""
