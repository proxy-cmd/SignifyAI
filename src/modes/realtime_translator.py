from collections import deque
from datetime import datetime
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from core.eye_detection import EyeCfg, EyeDetector, draw_eye_debug
from core.hand_detection import CamCfg, CamStream, HandCfg, HandDetector, draw_hands
from core.speech_engine import Speaker
from core.stability import Hit, StableCfg, StableFilter
from dataset.recording import frame_to_vec
from modes.demo_mode import DEMO_SIGNS, DemoDecoder
from modes.eye_assist_mode import EyeAssistDecoder
from modes.emergency_mode import AID_SIGNS, AidDecoder
from model.model_manager import ModelHub
from model.sequence_model import load_model_for_runtime, predict_proba_bundle

# hand landmark indexes
TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


INTENTS = {
    "hospital_help": "I need medical help.",
    "need_water": "I need water.",
    "need_food": "I need food.",
    "need_toilet": "I need to use the toilet.",
    "call_family": "Please call my family.",
    "emergency": "This is an emergency.",
    "severe_pain": "I am in severe pain.",
    "cannot_breathe": "I cannot breathe.",
    "bleeding": "I am bleeding.",
    "head_injury": "I have a head injury.",
    "chest_pain": "I have chest pain.",
    "thank_you": "Thank you.",
    "yes": "Yes.",
    "no": "No.",
    "hello": "Hello.",
}


def intent_text(intent_id):
    if intent_id in INTENTS:
        return INTENTS[intent_id]
    return intent_id.replace("_", " ").title()


class StageTimer:
    def __init__(self):
        self.start = time.perf_counter()

    def ms(self):
        return (time.perf_counter() - self.start) * 1000.0


class RollMetrics:
    def __init__(self, size=240):
        self.size = int(size)
        self.stage = {}
        self.e2e = deque(maxlen=self.size)

    def add(self, name, value_ms):
        if name not in self.stage:
            self.stage[name] = deque(maxlen=self.size)
        self.stage[name].append(float(value_ms))

    def add_e2e(self, value_ms):
        self.e2e.append(float(value_ms))

    def snap(self):
        out = {}
        for name, vals in self.stage.items():
            if vals:
                out[name + "_median_ms"] = float(statistics.median(vals))
        if self.e2e:
            out["e2e_median_ms"] = float(statistics.median(self.e2e))
        return out


class LiveCfg:
    def __init__(
        self,
        cam_idx=0,
        w=960,
        h=540,
        fps=30,
        voice=True,
        seq_len=24,
        model_name=None,
        global_model_name="global",
        mode="hybrid",
    ):
        self.cam_idx = cam_idx
        self.w = w
        self.h = h
        self.fps = fps
        self.voice = voice
        self.seq_len = seq_len
        self.model_name = model_name
        self.global_model_name = global_model_name
        self.mode = mode


class RuleDecoder:
    def _dist(self, a, b):
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _finger_state(self, hand):
        state = {}
        for name in ("index", "middle", "ring", "pinky"):
            tip_y = hand[TIP[name], 1]
            pip_y = hand[PIP[name], 1]
            mcp_y = hand[MCP[name], 1]
            state[name] = bool(tip_y < pip_y < (mcp_y + 0.02))

        thumb_tip = hand[TIP["thumb"]]
        thumb_ip = hand[PIP["thumb"]]
        wrist = hand[0]
        palm = (hand[MCP["index"]] + hand[MCP["middle"]] + hand[MCP["ring"]] + hand[MCP["pinky"]]) / 4.0
        state["thumb"] = bool(
            self._dist(thumb_tip, palm) > self._dist(thumb_ip, palm) * 1.04
            and self._dist(thumb_tip, wrist) > self._dist(thumb_ip, wrist) * 1.02
        )
        return state

    def _day_time_intent(self):
        hour = datetime.now().hour
        if hour < 12:
            return "hello"
        return "thank_you"

    def decode(self, frame):
        hands = []
        for hand in (frame.left, frame.right):
            if hand is not None:
                hands.append(hand)
        if not hands:
            return None

        if len(hands) >= 2:
            a = self._finger_state(hands[0])
            b = self._finger_state(hands[1])
            if (not any(a.values())) and (not any(b.values())):
                return Hit("emergency", 0.82, "rules")
            if a["index"] and b["index"]:
                return Hit("hospital_help", 0.84, "rules")
            return Hit("call_family", 0.70, "rules")

        hand = hands[0]
        state = self._finger_state(hand)

        if state["index"] and state["middle"] and not state["ring"] and not state["pinky"]:
            spread = self._dist(hand[TIP["index"]], hand[TIP["middle"]])
            if spread > 0.09:
                return Hit("thank_you", 0.80, "rules")
            return Hit("need_water", 0.78, "rules")

        if state["thumb"] and not state["index"] and not state["middle"] and not state["ring"] and not state["pinky"]:
            tip = hand[TIP["thumb"]]
            wrist = hand[0]
            if tip[1] < wrist[1] - 0.05:
                return Hit("yes", 0.86, "rules")
            if tip[1] > wrist[1] + 0.05:
                return Hit("no", 0.86, "rules")

        if state["thumb"] and state["index"] and state["middle"] and state["ring"] and state["pinky"]:
            return Hit(self._day_time_intent(), 0.83, "rules")

        if not any(state.values()):
            return Hit("need_toilet", 0.72, "rules")
        return Hit("need_food", 0.66, "rules")


class LiveRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cam = CamStream(CamCfg(idx=cfg.cam_idx, w=cfg.w, h=cfg.h, fps=cfg.fps))
        self.det = None
        self.eye_det = None
        if cfg.mode == "eye":
            self.eye_det = EyeDetector(EyeCfg(min_det=0.5, min_track=0.5))
        else:
            self.det = HandDetector(HandCfg(scale=0.85))
        self.rule_dec = RuleDecoder()
        self.demo_dec = DemoDecoder()
        self.aid_dec = AidDecoder()
        self.eye_dec = EyeAssistDecoder()
        self.stable = StableFilter(StableCfg(win=7, min_conf=0.55, hold_sec=0.18))
        self.speaker = Speaker(rate=185, volume=1.0)
        self.metrics = RollMetrics()
        self.seq_buf = deque(maxlen=max(8, cfg.seq_len))

        hub = ModelHub()
        active_name = cfg.model_name or hub.active() or "custom"
        self.main_model_name = active_name
        self.global_model_name = cfg.global_model_name
        self.model_shape_warned = set()
        self.meta_dir = Path("data/models")

        self.main_seq_len = self._read_model_seq_len(self.main_model_name, fallback=max(1, cfg.seq_len))
        self.global_seq_len = self._read_model_seq_len(self.global_model_name, fallback=1)

        self.main_model = load_model_for_runtime(self.main_model_name)
        self.global_model = load_model_for_runtime(self.global_model_name)

        if self.main_model is None:
            print(f"[warn] main model not found: {self.main_model_name}")
        if self.global_model is None:
            print(f"[warn] global model not found: {self.global_model_name}")

        self.main_classes = self._model_classes(self.main_model)
        self.global_classes = self._model_classes(self.global_model)
        self._print_model_summary()

        self.last_spoken_label = ""
        self.last_spoken_ts = 0.0
        self.show_help = False
        self.last_candidate = {}
        self.candidate_count = {}
        self.no_hand_frames = 0
        self.last_eye_label = "unknown"
        self.last_eye_conf = 0.0
        self.last_eye_ts = 0.0
        self.eye_display_hold_sec = 1.1
        self.show_eye_landmarks = True

    @staticmethod
    def _model_classes(model_obj):
        if model_obj is None:
            return []
        inner = model_obj["model"] if isinstance(model_obj, dict) else model_obj
        return [str(c) for c in getattr(inner, "classes_", [])]

    @staticmethod
    def _fmt_classes(classes, max_items=12):
        if not classes:
            return "none"
        if len(classes) <= max_items:
            return ", ".join(classes)
        head = ", ".join(classes[:max_items])
        return f"{head}, ... (+{len(classes) - max_items} more)"

    def _print_model_summary(self):
        print("\n=== Realtime Models ===")
        print(
            f"Main: {self.main_model_name} | seq_len={self.main_seq_len} | classes={self._fmt_classes(self.main_classes)}"
        )
        print(
            f"Global: {self.global_model_name} | seq_len={self.global_seq_len} | classes={self._fmt_classes(self.global_classes)}"
        )
        print("=======================")

    def _read_model_seq_len(self, model_name, fallback):
        meta = self.meta_dir / f"{model_name}.json"
        if not meta.exists():
            return int(fallback)
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            return int(data.get("seq_len", fallback))
        except Exception:
            return int(fallback)

    def close(self):
        if self.det is not None:
            self.det.close()
        if self.eye_det is not None:
            self.eye_det.close()
        self.speaker.close()
        self.cam.close()

    def push_seq(self, frame_data):
        self.seq_buf.append(frame_to_vec(frame_data))

    @staticmethod
    def _has_hand(frame_data):
        return (frame_data.left is not None) or (frame_data.right is not None)

    @staticmethod
    def _is_single_hand(frame_data):
        return (frame_data.left is None) != (frame_data.right is None)

    @staticmethod
    def _swap_lr_seq(seq):
        # per-frame feature layout: left(63), right(63), quality(3)
        if seq.ndim != 2 or seq.shape[1] < 126:
            return seq
        out = np.asarray(seq, dtype=np.float32).copy()
        out[:, :63] = seq[:, 63:126]
        out[:, 63:126] = seq[:, :63]
        return out

    @staticmethod
    def _primary_hand(frame_data):
        if frame_data.left is not None:
            return frame_data.left
        return frame_data.right

    def _global_shape_hint(self, frame_data):
        # lightweight hint for c/o/v/r when using single-hand letter model
        if not self._is_single_hand(frame_data):
            return set()
        hand = self._primary_hand(frame_data)
        if hand is None or len(hand) < 21:
            return set()

        state = self.rule_dec._finger_state(hand)
        spread = self.rule_dec._dist(hand[TIP["index"]], hand[TIP["middle"]])
        thumb_index = self.rule_dec._dist(hand[TIP["thumb"]], hand[TIP["index"]])
        thumb_pinky = self.rule_dec._dist(hand[TIP["thumb"]], hand[TIP["pinky"]])

        if state["index"] and state["middle"] and (not state["ring"]) and (not state["pinky"]):
            if spread > 0.085:
                return {"v"}
            return {"r"}

        if (not state["index"]) and (not state["middle"]) and (not state["ring"]) and (not state["pinky"]):
            if thumb_index < 0.040 and thumb_pinky < 0.090:
                return {"o"}
            return {"c"}

        return set()

    def _predict_with_model(self, model_obj, model_name, seq_len, single_hand=False, relaxed=False, shape_hint=None):
        if model_obj is None:
            return None
        need = max(1, int(seq_len))
        if len(self.seq_buf) < need:
            return None
        items = list(self.seq_buf)
        seq = np.stack(items[-need:], axis=0)

        # skip model safely if feature shape does not match runtime features
        flat_len = int(seq.reshape(1, -1).shape[1])
        inner = model_obj["model"] if isinstance(model_obj, dict) else model_obj
        expected = getattr(inner, "n_features_in_", None)
        if expected is not None and int(expected) != flat_len:
            key = f"{model_name}:{expected}:{flat_len}"
            if key not in self.model_shape_warned:
                print(f"[warn] skipping model {model_name}: expects {expected} features, runtime has {flat_len}.")
                self.model_shape_warned.add(key)
            return None

        flat = seq.reshape(1, -1)
        try:
            probs = predict_proba_bundle(model_obj, flat)[0]
        except Exception as ex:
            key = f"{model_name}:predict_error"
            if key not in self.model_shape_warned:
                print(f"[warn] skipping model {model_name}: {ex}")
                self.model_shape_warned.add(key)
            return None

        # right/left invariant scoring for one-hand gestures
        if single_hand:
            try:
                swapped = self._swap_lr_seq(seq)
                probs_swapped = predict_proba_bundle(model_obj, swapped.reshape(1, -1))[0]
                if probs_swapped is not None and len(probs_swapped) == len(probs):
                    probs = (np.asarray(probs, dtype=np.float32) + np.asarray(probs_swapped, dtype=np.float32)) / 2.0
            except Exception:
                pass

        if probs is None or len(probs) == 0:
            return None

        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        if isinstance(model_obj, dict):
            cls = list(getattr(model_obj["model"], "classes_", []))
        else:
            cls = list(getattr(model_obj, "classes_", []))
        if not cls or idx >= len(cls):
            return None

        # apply weak shape prior for global c/o/v/r letters to reduce single-class collapse
        if model_name == self.global_model_name and shape_hint:
            probs_adj = np.asarray(probs, dtype=np.float32).copy()
            for i, c in enumerate(cls):
                label_i = str(c)
                if label_i in shape_hint:
                    probs_adj[i] *= 1.55
                elif label_i == "r" and ("r" not in shape_hint):
                    probs_adj[i] *= 0.70
            s = float(np.sum(probs_adj))
            if s > 0:
                probs = probs_adj / s
                idx = int(np.argmax(probs))
                conf = float(probs[idx])

        label = str(cls[idx])
        sorted_probs = np.sort(np.asarray(probs, dtype=np.float32))
        second = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
        margin = conf - second

        # default mode can run relaxed for responsiveness; non-default stays stricter
        if relaxed:
            if model_name == self.global_model_name:
                min_conf = 0.40
                min_margin = -1.0
                min_repeat = 1
            else:
                min_conf = 0.55
                min_margin = 0.00
                min_repeat = 1
        else:
            if model_name == self.main_model_name:
                min_conf = 0.70
                min_margin = 0.08
                min_repeat = 2
            else:
                min_conf = 0.55
                min_margin = 0.02
                min_repeat = 1
        if conf < min_conf or margin < min_margin:
            self.last_candidate[model_name] = None
            self.candidate_count[model_name] = 0
            return None

        # require same label across a few frames before accepting
        prev = self.last_candidate.get(model_name)
        if prev == label:
            self.candidate_count[model_name] = int(self.candidate_count.get(model_name, 0)) + 1
        else:
            self.last_candidate[model_name] = label
            self.candidate_count[model_name] = 1

        if self.candidate_count.get(model_name, 0) < min_repeat:
            return None

        return Hit(label, conf, "model:" + model_name)

    def model_predict(self, frame_data, prefer_global=False, relaxed=False):
        if not self._has_hand(frame_data):
            return None

        single_hand = self._is_single_hand(frame_data)
        shape_hint = self._global_shape_hint(frame_data)

        if prefer_global:
            hit = self._predict_with_model(
                self.global_model,
                self.global_model_name,
                self.global_seq_len,
                single_hand=single_hand,
                relaxed=relaxed,
                shape_hint=shape_hint,
            )
            if hit is not None:
                return hit
            return None

        # try recorded/custom model first
        hit = self._predict_with_model(
            self.main_model,
            self.main_model_name,
            self.main_seq_len,
            single_hand=single_hand,
            relaxed=relaxed,
            shape_hint=shape_hint,
        )
        if hit is not None:
            return hit

        # then try global model
        hit = self._predict_with_model(
            self.global_model,
            self.global_model_name,
            self.global_seq_len,
            single_hand=single_hand,
            relaxed=relaxed,
            shape_hint=shape_hint,
        )
        if hit is not None:
            return hit
        return None

    def decode(self, frame_data=None, eye_state=None):
        if self.cfg.mode == "eye":
            return self.eye_dec.decode(eye_state)

        if self.cfg.mode == "aid":
            return self.aid_dec.decode(frame_data)

        if self.cfg.mode == "demo":
            return self.demo_dec.decode(frame_data)

        # default mode is model-only: custom first, then global fallback
        if self.cfg.mode == "default":
            return self.model_predict(frame_data, prefer_global=True, relaxed=True)

        if self.cfg.mode == "hybrid":
            hit = self.model_predict(frame_data)
            if hit is not None:
                return hit
            demo_hit = self.demo_dec.decode(frame_data)
            if demo_hit is not None:
                return demo_hit
            return None

        hit = self.model_predict(frame_data)
        if hit is not None:
            return hit
        demo_hit = self.demo_dec.decode(frame_data)
        if demo_hit is not None:
            return demo_hit
        return None

    def maybe_speak(self, label, conf, voice_on, has_hand, raw_label):
        if not voice_on:
            return
        if not has_hand:
            return
        if label in {"unknown", "silence"}:
            return
        if raw_label is None or raw_label != label:
            return

        now = time.time()
        if label == self.last_spoken_label and (now - self.last_spoken_ts) < 1.8:
            return
        if (now - self.last_spoken_ts) < 0.35:
            return

        self.speaker.say_latest(intent_text(label))
        self.last_spoken_label = label
        self.last_spoken_ts = now

    def draw_overlay(self, frame, label, conf, source, voice_on):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
        cv2.putText(frame, "Intent: " + label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
        info = f"Conf {conf:.2f} | Voice {'ON' if voice_on else 'OFF'} | Source {source} | Mode {self.cfg.mode}"
        cv2.putText(frame, info, (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 220, 255), 2)
        keys = "Keys: q/esc quit | v voice | r reset"
        if self.cfg.mode == "eye":
            keys += " | l landmarks"
        cv2.putText(frame, keys, (18, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

    @staticmethod
    def build_eye_panel():
        panel_h = 460
        panel_w = 600
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (18, 18, 18)
        cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (80, 80, 80), 1)
        cv2.putText(panel, "Eye Assist (Test Mode)", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(panel, "Keep face centered and eyes visible", (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

        lines = [
            "1. LONG BLINK (>=0.75s) -> EMERGENCY",
            "2. SINGLE BLINK -> YES",
            "3. TRIPLE BLINK -> NEED_WATER",
            "4. LOOK LEFT HOLD -> NO",
            "5. LOOK RIGHT HOLD -> CALL_FAMILY",
            "6. LOOK UP HOLD -> NEED_FOOD",
            "7. LOOK DOWN HOLD -> NEED_TOILET",
        ]
        y = 92
        for line in lines:
            cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (235, 235, 235), 1)
            y += 48

        cv2.putText(panel, "Keys: q quit | v voice | r reset | l landmarks", (14, panel_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        return panel

    @staticmethod
    def build_aid_panel():
        panel_h = max(420, 104 + (len(AID_SIGNS) * 34) + 40)
        panel_w = 760
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (18, 18, 18)
        cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (80, 80, 80), 1)
        cv2.putText(panel, "Quick Aid Signs", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(panel, "Use one clear hand", (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        y = 82
        for i, sign in enumerate(AID_SIGNS, start=1):
            line = f"{i}. {sign.label.upper()} -> {sign.hint}"
            cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1)
            y += 34
        cv2.putText(panel, "Keys: q quit | v voice | r reset", (14, panel_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        return panel

    @staticmethod
    def build_demo_panel():
        panel = np.zeros((540, 460, 3), dtype=np.uint8)
        panel[:] = (18, 18, 18)
        cv2.rectangle(panel, (0, 0), (459, 539), (80, 80, 80), 1)
        cv2.putText(panel, "Demo signs", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(panel, "Show one clear hand", (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        y = 84
        for i, sign in enumerate(DEMO_SIGNS, start=1):
            line = f"{i}. {sign.label.upper()} -> {sign.hint}"
            cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (235, 235, 235), 1)
            y += 31
            if y > 508:
                break
        cv2.putText(panel, "Keys: q quit | v voice | r reset", (14, 528), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        return panel

    def run(self):
        win = "SignifyAI"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.cfg.w, self.cfg.h)

        guide_win = "SignifyAI Guide"
        use_side_guide = self.cfg.mode in {"aid", "demo", "eye"}
        if use_side_guide:
            cv2.namedWindow(guide_win, cv2.WINDOW_NORMAL)
            if self.cfg.mode == "demo":
                cv2.resizeWindow(guide_win, 460, 540)
            elif self.cfg.mode == "eye":
                cv2.resizeWindow(guide_win, 600, 460)
            else:
                panel_h = max(420, 104 + (len(AID_SIGNS) * 34) + 40)
                cv2.resizeWindow(guide_win, 760, panel_h)

        voice_on = bool(self.cfg.voice)

        try:
            while True:
                e2e_timer = StageTimer()

                timer = StageTimer()
                ok, frame = self.cam.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera read failed")
                frame = cv2.flip(frame, 1)
                self.metrics.add("capture", timer.ms())

                frame_data = None
                eye_state = None
                has_signal = False

                timer = StageTimer()
                if self.cfg.mode == "eye":
                    if self.eye_det is None:
                        raise RuntimeError("Eye detector is not initialized")
                    eye_state = self.eye_det.process(frame)
                    if eye_state is None:
                        has_signal = False
                    else:
                        has_signal = bool(eye_state.face_found)
                        draw_eye_debug(frame, eye_state, show_landmarks=self.show_eye_landmarks)
                else:
                    if self.det is None:
                        raise RuntimeError("Hand detector is not initialized")
                    frame_data = self.det.process(frame)
                    has_hand = self._has_hand(frame_data)
                    has_signal = has_hand
                    if has_hand:
                        self.no_hand_frames = 0
                    else:
                        self.no_hand_frames += 1
                        if self.no_hand_frames >= 6:
                            self.seq_buf.clear()
                            self.last_candidate.clear()
                            self.candidate_count.clear()
                            self.stable.reset()

                    draw_hands(frame, frame_data)
                    if has_hand:
                        self.push_seq(frame_data)
                self.metrics.add("perception", timer.ms())

                timer = StageTimer()
                raw_hit = self.decode(frame_data=frame_data, eye_state=eye_state)
                self.metrics.add("decode", timer.ms())

                stable_label, stable_conf, _ = self.stable.update(raw_hit)
                # In default mode, show/speak direct model hit immediately to avoid over-smoothing silence.
                if self.cfg.mode == "default" and raw_hit is not None:
                    stable_label = raw_hit.label
                    stable_conf = raw_hit.conf
                if self.cfg.mode == "eye":
                    now_ts = time.time()
                    if raw_hit is not None:
                        self.last_eye_label = raw_hit.label
                        self.last_eye_conf = raw_hit.conf
                        self.last_eye_ts = now_ts
                        stable_label = raw_hit.label
                        stable_conf = raw_hit.conf
                    elif (now_ts - self.last_eye_ts) <= self.eye_display_hold_sec:
                        stable_label = self.last_eye_label
                        stable_conf = self.last_eye_conf

                timer = StageTimer()
                raw_label = None if raw_hit is None else raw_hit.label
                self.maybe_speak(stable_label, stable_conf, voice_on, has_signal, raw_label)
                self.metrics.add("speech", timer.ms())

                timer = StageTimer()
                source = "none" if raw_hit is None else raw_hit.src
                self.draw_overlay(frame, stable_label, stable_conf, source, voice_on)
                self.metrics.add("render", timer.ms())

                self.metrics.add_e2e(e2e_timer.ms())
                snap = self.metrics.snap()
                med = snap.get("e2e_median_ms", 0.0)
                print(f"\r[e2e median] {med:6.1f} ms", end="")

                cv2.imshow(win, frame)
                if use_side_guide:
                    if self.cfg.mode == "demo":
                        cv2.imshow(guide_win, self.build_demo_panel())
                    elif self.cfg.mode == "eye":
                        cv2.imshow(guide_win, self.build_eye_panel())
                    else:
                        cv2.imshow(guide_win, self.build_aid_panel())

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("v"):
                    voice_on = not voice_on
                if key == ord("r"):
                    self.stable.reset()
                    self.last_spoken_label = ""
                if key == ord("l") and self.cfg.mode == "eye":
                    self.show_eye_landmarks = not self.show_eye_landmarks
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if use_side_guide and cv2.getWindowProperty(guide_win, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            print()
            self.close()
            cv2.destroyAllWindows()
