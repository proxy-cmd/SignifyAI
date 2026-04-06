from collections import deque
from datetime import datetime
import json
from pathlib import Path
import statistics
import time
import uuid

import cv2
import numpy as np

from core.eye_detection import EyeCfg, EyeDetector, draw_eye_debug
from core.hand_detection import CamCfg, CamStream, FrameData, HandCfg, HandDetector, draw_hands
from core.output_policy import apply_uncertain, should_speak
from core.speech_engine import Speaker
from core.stability import Hit, StableCfg, StableFilter
from dataset.recording import frame_to_vec
from modes.adaptive_sign_mode import AdaptiveSignDecoder
from modes.eye_assist_mode import EyeAssistDecoder
from modes.emergency_mode import AID_SIGNS, AidDecoder
from model.model_manager import ModelHub
from model.sequence_model import load_runtime_model, predict_probs

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

EMERGENCY_ONLY_INTENTS = {
    "hospital_help",
    "need_water",
    "need_food",
    "need_toilet",
    "call_family",
    "emergency",
    "severe_pain",
    "cannot_breathe",
    "bleeding",
    "head_injury",
    "chest_pain",
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
        default_infer_interval=3,
        uncertainty_min_conf=0.48,
        speech_repeat_cooldown_sec=1.8,
        speech_global_cooldown_sec=0.35,
        watchdog_reset_sec=5.0,
        save_teach_data=False,
        camera_enabled=True,
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
        self.default_infer_interval = max(1, int(default_infer_interval))
        self.uncertainty_min_conf = float(uncertainty_min_conf)
        self.speech_repeat_cooldown_sec = float(speech_repeat_cooldown_sec)
        self.speech_global_cooldown_sec = float(speech_global_cooldown_sec)
        self.watchdog_reset_sec = float(watchdog_reset_sec)
        self.save_teach_data = bool(save_teach_data)
        self.camera_enabled = bool(camera_enabled)


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
        self.cam = CamStream(CamCfg(idx=cfg.cam_idx, w=cfg.w, h=cfg.h, fps=cfg.fps)) if cfg.camera_enabled else None
        self.det = None
        self.eye_det = None
        if cfg.mode == "eye":
            self.eye_det = EyeDetector(EyeCfg(min_det=0.5, min_track=0.5))
        else:
            if cfg.mode in {"default", "teach"}:
                self.eye_det = EyeDetector(EyeCfg(min_det=0.45, min_track=0.45))
            fast_mode = cfg.mode in {"default", "teach", "aid"}
            det_scale = 0.80 if fast_mode else 0.85
            det_min_det = 0.45 if fast_mode else 0.50
            det_fallback = False if fast_mode else True
            det_enhance_fallback = False if fast_mode else True
            det_max_hands = 2
            det_quality = False if fast_mode else True
            self.det = HandDetector(
                HandCfg(
                    scale=det_scale,
                    min_det=det_min_det,
                    max_hands=det_max_hands,
                    full_res_fallback=det_fallback,
                    enhance_fallback=det_enhance_fallback,
                    compute_quality=det_quality,
                )
            )
        self.rule_dec = RuleDecoder()
        self.aid_dec = AidDecoder()
        self.eye_dec = EyeAssistDecoder()
        self.adaptive_dec = AdaptiveSignDecoder()
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

        if self.cfg.mode in {"default", "teach", "aid", "eye"}:
            # Patient runtime modes avoid heavyweight model inference for smoother performance.
            self.main_seq_len = 1
            self.global_seq_len = 1
            self.main_model = None
            self.global_model = None
            self.main_classes = []
            self.global_classes = []
        else:
            self.main_seq_len = self._read_seq_len(self.main_model_name, fallback=max(1, cfg.seq_len))
            self.global_seq_len = self._read_seq_len(self.global_model_name, fallback=1)

            self.main_model = load_runtime_model(self.main_model_name)
            self.global_model = load_runtime_model(self.global_model_name)

            if self.main_model is None:
                print(f"[warn] main model not found: {self.main_model_name}")
            if self.global_model is None:
                print(f"[warn] global model not found: {self.global_model_name}")

            self.main_classes = self._model_classes(self.main_model)
            self.global_classes = self._model_classes(self.global_model)
        self._print_models()

        self.last_spoken_label = ""
        self.last_spoken_ts = 0.0
        self.show_help = False
        self.last_candidate = {}
        self.candidate_count = {}
        self.no_hand_frames = 0
        self.no_signal_start_ts = None
        self.last_watchdog_reset_ts = 0.0
        self.last_eye_label = "unknown"
        self.last_eye_conf = 0.0
        self.last_eye_ts = 0.0
        self.eye_display_hold_sec = 1.1
        self.last_live_label = "unknown"
        self.last_live_conf = 0.0
        self.last_live_ts = 0.0
        self.live_hold_sec = 0.45
        self.show_eye_landmarks = True
        self.show_hand_landmarks = True
        self.face_sample_stride = 1 if self.cfg.mode in {"default", "teach"} else 3
        self.face_sample_counter = 0
        self.last_face_state = None
        self.blink_closed = False
        self.blink_close_start_ms = None
        self.blink_min_ear = 1.0
        self.blink_close_ear = 0.19
        self.blink_open_ear = 0.235
        self.blink_hard_close_ear = 0.165
        self.blink_min_ms = 45
        self.blink_max_ms = 520
        self.triple_blink_window_ms = 1600
        self.blink_teach_cooldown_ms = 3200
        self.last_blink_teach_ts = -1_000_000_000
        self.recent_blinks = deque(maxlen=12)
        self.proto_candidate = None
        self.proto_candidate_count = 0
        self.last_taught_label = ""
        self.last_taught_ts = 0.0
        self.yn_candidate = None
        self.yn_candidate_count = 0
        self.num_candidate = None
        self.num_candidate_count = 0
        self.letter_candidate = None
        self.letter_candidate_count = 0
        self.idle_detect_stride = 3
        self.idle_detect_counter = 0

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

    def _print_models(self):
        print("\n=== Realtime Models ===")
        if self.cfg.mode in {"default", "teach", "aid", "eye"}:
            mode_name = {
                "default": "Default",
                "teach": "Teach",
                "aid": "Emergency Aid",
                "eye": "Eye Assist",
            }.get(self.cfg.mode, self.cfg.mode.title())
            print(f"{mode_name} mode: low-latency runtime (no model inference)")
            print("Emergency rules + eye assist + teachable prototypes")
            print("=======================")
            return
        print(
            f"Main: {self.main_model_name} | seq_len={self.main_seq_len} | classes={self._fmt_classes(self.main_classes)}"
        )
        print(
            f"Global: {self.global_model_name} | seq_len={self.global_seq_len} | classes={self._fmt_classes(self.global_classes)}"
        )
        global_set = {str(c).lower() for c in self.global_classes}
        if global_set and global_set.issubset({"c", "o", "r", "v"}):
            print("[note] global model currently has only classes: c/o/r/v")
            print("[note] non-c/o/r/v signs will be treated as unknown in default mode")
        print("=======================")

    def _read_seq_len(self, model_name, fallback):
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
        if self.cam is not None:
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

    def _pred_model(self, model_obj, model_name, seq_len, single_hand=False, relaxed=False, shape_hint=None):
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
            probs = predict_probs(model_obj, flat)[0]
        except Exception as ex:
            key = f"{model_name}:predict_error"
            if key not in self.model_shape_warned:
                print(f"[warn] skipping model {model_name}: {ex}")
                self.model_shape_warned.add(key)
            return None

        # right/left invariant scoring for one-hand gestures
        if single_hand:
            # In fast default letter mode (global seq_len=1), avoid the extra swapped pass.
            do_swapped = not (
                relaxed
                and model_name == self.global_model_name
                and int(seq_len) == 1
            )
            if not do_swapped:
                probs_swapped = None
            else:
                probs_swapped = None
            try:
                if do_swapped:
                    swapped = self._swap_lr_seq(seq)
                    probs_swapped = predict_probs(model_obj, swapped.reshape(1, -1))[0]
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

        # If global model is a tiny c/o/r/v letter model, avoid forcing wrong labels on unrelated signs.
        if model_name == self.global_model_name:
            cls_set = {str(c).lower() for c in cls}
            if cls_set and cls_set.issubset({"c", "o", "r", "v"}):
                if not shape_hint:
                    self.last_candidate[model_name] = None
                    self.candidate_count[model_name] = 0
                    return None
                if str(label).lower() not in {str(s).lower() for s in shape_hint}:
                    self.last_candidate[model_name] = None
                    self.candidate_count[model_name] = 0
                    return None

        # default mode can run relaxed for responsiveness; non-default stays stricter
        if relaxed:
            if model_name == self.global_model_name:
                min_conf = 0.36
                min_margin = 0.00
                min_repeat = 2
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
            hit = self._pred_model(
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
        hit = self._pred_model(
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
        hit = self._pred_model(
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
            hit = self.aid_dec.decode(frame_data)
            if hit is not None:
                return hit
            # Aid mode also accepts user-taught prototype signs for hospitals/institutions.
            proto_hit = self.adaptive_dec.decode(frame_data, eye_state=eye_state)
            if proto_hit is not None and str(getattr(proto_hit, "src", "")) == "prototype":
                return proto_hit
            return None

        # default/teach modes are adaptive: hardcoded letters/signs + learned prototypes
        if self.cfg.mode in {"default", "teach"}:
            hit = self.adaptive_dec.decode(frame_data, eye_state=eye_state)
            if hit is not None and str(hit.label) in EMERGENCY_ONLY_INTENTS:
                return None

            # Debounce yes/no to avoid dominant repetitive triggers.
            if hit is not None and str(hit.label) in {"yes", "no"}:
                lbl = str(hit.label)
                if self.yn_candidate == lbl:
                    self.yn_candidate_count += 1
                else:
                    self.yn_candidate = lbl
                    self.yn_candidate_count = 1
                if self.yn_candidate_count < 3:
                    return None
            else:
                self.yn_candidate = None
                self.yn_candidate_count = 0

            # Debounce one/two slightly to avoid false jumps on unknown poses.
            if hit is not None and str(hit.label) in {"one", "two"}:
                lbl = str(hit.label)
                if self.num_candidate == lbl:
                    self.num_candidate_count += 1
                else:
                    self.num_candidate = lbl
                    self.num_candidate_count = 1
                if self.num_candidate_count < 2:
                    return None
            else:
                self.num_candidate = None
                self.num_candidate_count = 0

            # Debounce letter rules to reduce random handshape confusion.
            if hit is not None and str(getattr(hit, "src", "")) == "letters":
                lbl = str(hit.label)
                if self.letter_candidate == lbl:
                    self.letter_candidate_count += 1
                else:
                    self.letter_candidate = lbl
                    self.letter_candidate_count = 1
                if self.letter_candidate_count < 2:
                    return None
            else:
                self.letter_candidate = None
                self.letter_candidate_count = 0

            # Prototype labels must remain stable across a few frames.
            if hit is not None and str(getattr(hit, "src", "")) == "prototype":
                lbl = str(hit.label)
                now_ts = time.time()
                # Freshly taught sign should work immediately.
                if lbl == self.last_taught_label and (now_ts - float(self.last_taught_ts)) <= 30.0:
                    return hit
                if float(getattr(hit, "conf", 0.0)) < 0.62:
                    return None
                if self.proto_candidate == lbl:
                    self.proto_candidate_count += 1
                else:
                    self.proto_candidate = lbl
                    self.proto_candidate_count = 1
                if self.proto_candidate_count < 2:
                    return None
            else:
                self.proto_candidate = None
                self.proto_candidate_count = 0
            return hit

        if self.cfg.mode == "hybrid":
            hit = self.model_predict(frame_data)
            if hit is not None:
                return hit
            return None

        hit = self.model_predict(frame_data)
        if hit is not None:
            return hit
        return None

    def maybe_speak(self, label, conf, voice_on, has_hand, raw_label):
        now = time.time()
        if not should_speak(
            label=label,
            raw_label=raw_label,
            has_signal=has_hand,
            voice_on=voice_on,
            now_ts=now,
            last_spoken_label=self.last_spoken_label,
            last_spoken_ts=self.last_spoken_ts,
            repeat_cooldown_sec=self.cfg.speech_repeat_cooldown_sec,
            global_cooldown_sec=self.cfg.speech_global_cooldown_sec,
        ):
            return

        self.speaker.say_latest(intent_text(label))
        self.last_spoken_label = label
        self.last_spoken_ts = now

    def _save_taught_clip(self, label, frame_data):
        base = Path("data/landmarks/raw/live_teach")
        base.mkdir(parents=True, exist_ok=True)
        session_file = base / "session.json"
        if not session_file.exists():
            session = {
                "session_id": "live_teach",
                "intent_id": "dynamic",
                "signer_id": "live",
                "consent_raw_video": False,
                "created_at": int(time.time() * 1000),
                "session_dir": str(base),
            }
            session_file.write_text(json.dumps(session, indent=2), encoding="utf-8")

        clip_id = f"clip_{uuid.uuid4().hex[:10]}"
        npz_path = base / f"{clip_id}.npz"
        seq = np.stack([frame_to_vec(frame_data)], axis=0).astype(np.float32)
        ts = np.asarray([int(getattr(frame_data, "ts_ms", int(time.time() * 1000)))], dtype=np.int64)
        np.savez_compressed(npz_path, sequence=seq, timestamps=ts)

        row = {
            "session_id": "live_teach",
            "clip_id": clip_id,
            "intent_id": str(label),
            "sign_id": str(self.adaptive_dec.sign_id_for(label)),
            "signer_id": "live",
            "consent_raw_video": False,
            "npz_path": str(npz_path),
            "frames": 1,
            "quality": getattr(frame_data, "quality", {}),
            "source_mode": self.cfg.mode,
        }
        with (base / "clips.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _teach_now(self, frame_data, eye_state=None):
        if self.cfg.mode not in {"default", "teach", "aid"} or frame_data is None or (not self._has_hand(frame_data)):
            return False
        try:
            name = input("\n[teach] Name this sign (blank to skip): ").strip()
        except EOFError:
            return False
        if not name:
            return False
        ok = self.adaptive_dec.teach(frame_data, name, eye_state=eye_state)
        if ok:
            if self.cfg.save_teach_data:
                self._save_taught_clip(name, frame_data)
            self.last_taught_label = str(name).strip().lower().replace(" ", "_")
            self.last_taught_ts = time.time()
            if self.cfg.save_teach_data:
                print(f"[teach] saved prototype + trace for '{name}'")
            else:
                print(f"[teach] saved prototype for '{name}' (trace off)")
        else:
            print("[teach] could not save prototype (no clear hand)")
        return ok

    def _triple_teach(self, eye_state):
        if self.cfg.mode not in {"default", "teach", "aid"}:
            return False
        if eye_state is None or (not bool(getattr(eye_state, "face_found", False))):
            self.blink_closed = False
            self.blink_close_start_ms = None
            self.blink_min_ear = 1.0
            self.recent_blinks.clear()
            return False

        now_ms = int(getattr(eye_state, "ts_ms", 0) or 0)
        if now_ms <= 0:
            return False

        ear = (float(getattr(eye_state, "left_ear", 0.0)) + float(getattr(eye_state, "right_ear", 0.0))) * 0.5
        if (not self.blink_closed) and ear <= self.blink_close_ear:
            self.blink_closed = True
            self.blink_close_start_ms = now_ms
            self.blink_min_ear = ear
            return False

        if self.blink_closed:
            self.blink_min_ear = min(float(self.blink_min_ear), float(ear))
            if ear >= self.blink_open_ear:
                self.blink_closed = False
                start_ms = self.blink_close_start_ms if self.blink_close_start_ms is not None else now_ms
                self.blink_close_start_ms = None
                blink_ms = max(0, now_ms - int(start_ms))
                min_ear = float(self.blink_min_ear)
                self.blink_min_ear = 1.0

                valid = (
                    min_ear <= self.blink_hard_close_ear
                    and blink_ms >= self.blink_min_ms
                    and blink_ms <= self.blink_max_ms
                )
                if not valid:
                    return False

                self.recent_blinks.append(now_ms)
                cut = now_ms - self.triple_blink_window_ms
                self.recent_blinks = deque([t for t in self.recent_blinks if t >= cut], maxlen=12)

                if len(self.recent_blinks) >= 3 and (now_ms - int(self.last_blink_teach_ts)) >= self.blink_teach_cooldown_ms:
                    self.recent_blinks.clear()
                    self.last_blink_teach_ts = now_ms
                    return True
        return False

    def draw_overlay(self, frame, label, conf, source, voice_on):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
        cv2.putText(frame, "Intent: " + label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
        info = f"Conf {conf:.2f} | Voice {'ON' if voice_on else 'OFF'} | Source {source} | Mode {self.cfg.mode}"
        cv2.putText(frame, info, (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 220, 255), 2)
        keys = "Keys: q/esc quit | v voice | r reset"
        if self.cfg.mode == "eye":
            keys += " | l landmarks"
        if self.cfg.mode in {"default", "teach", "aid"}:
            keys += " | t teach sign | 3 blinks teach"
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

    def run(self):
        if self.cam is None:
            raise RuntimeError("Camera is disabled for this runner instance")
        win = "SignifyAI"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.cfg.w, self.cfg.h)

        guide_win = "SignifyAI Guide"
        use_side_guide = self.cfg.mode in {"aid", "eye"}
        if use_side_guide:
            cv2.namedWindow(guide_win, cv2.WINDOW_NORMAL)
            if self.cfg.mode == "eye":
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

                    # In default mode with sustained no-hand scene, decimate hand detection
                    # to avoid cumulative lag while preserving quick reacquisition.
                    if self.cfg.mode in {"default", "teach", "aid"} and self.no_hand_frames >= 10:
                        self.idle_detect_counter = (self.idle_detect_counter + 1) % max(1, int(self.idle_detect_stride))
                        if self.idle_detect_counter != 0:
                            frame_data = FrameData(
                                int(time.time() * 1000),
                                0,
                                None,
                                None,
                                {"brightness": 0.0, "blur": 0.0, "hand_area": 0.0},
                            )
                        else:
                            frame_data = self.det.process(frame)
                    else:
                        frame_data = self.det.process(frame)

                    has_hand = self._has_hand(frame_data)
                    has_signal = has_hand
                    if has_hand:
                        self.no_hand_frames = 0
                        self.idle_detect_counter = 0
                    else:
                        self.no_hand_frames += 1
                        if self.no_hand_frames >= 6:
                            self.seq_buf.clear()
                            self.last_candidate.clear()
                            self.candidate_count.clear()
                            self.stable.reset()

                    if has_hand and self.show_hand_landmarks:
                        draw_hands(frame, frame_data)
                    if has_hand:
                        self.push_seq(frame_data)

                    # Hidden face sampling for adaptive matching (no face overlay in default/teach).
                    if self.cfg.mode in {"default", "teach"} and self.eye_det is not None:
                        self.face_sample_counter = (self.face_sample_counter + 1) % max(1, int(self.face_sample_stride))
                        if self.face_sample_counter == 0 or self.last_face_state is None:
                            try:
                                self.last_face_state = self.eye_det.process(frame)
                            except Exception:
                                self.last_face_state = None
                        eye_state = self.last_face_state
                self.metrics.add("perception", timer.ms())

                now_watchdog = time.time()
                if has_signal:
                    self.no_signal_start_ts = None
                else:
                    if self.no_signal_start_ts is None:
                        self.no_signal_start_ts = now_watchdog
                    silent_for = now_watchdog - float(self.no_signal_start_ts)
                    if (
                        silent_for >= float(self.cfg.watchdog_reset_sec)
                        and (now_watchdog - self.last_watchdog_reset_ts) >= float(self.cfg.watchdog_reset_sec)
                    ):
                        self.seq_buf.clear()
                        self.last_candidate.clear()
                        self.candidate_count.clear()
                        self.stable.reset()
                        self.last_watchdog_reset_ts = now_watchdog

                timer = StageTimer()
                raw_hit = self.decode(frame_data=frame_data, eye_state=eye_state)
                self.metrics.add("decode", timer.ms())

                # Keep last live hit briefly so output does not jump to uncertain too quickly.
                if self.cfg.mode in {"default", "teach", "aid"}:
                    now_live = time.time()
                    if raw_hit is not None:
                        self.last_live_label = str(raw_hit.label)
                        self.last_live_conf = float(raw_hit.conf)
                        self.last_live_ts = now_live
                    elif has_signal and (now_live - float(self.last_live_ts)) <= self.live_hold_sec:
                        raw_hit = Hit(self.last_live_label, max(0.50, self.last_live_conf * 0.96), "adaptive_hold")

                stable_label, stable_conf, _ = self.stable.update(raw_hit)
                # In default/teach modes, show/speak direct hit immediately to avoid over-smoothing silence.
                if self.cfg.mode in {"default", "teach", "aid"} and raw_hit is not None:
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

                if self.cfg.mode in {"default", "teach", "aid"} and self._triple_teach(eye_state):
                    self._teach_now(frame_data, eye_state=eye_state)

                source = "none" if raw_hit is None else raw_hit.src
                stable_label, source, _ = apply_uncertain(
                    stable_label,
                    stable_conf,
                    source,
                    self.cfg.uncertainty_min_conf,
                )

                timer = StageTimer()
                raw_label = None if raw_hit is None else raw_hit.label
                self.maybe_speak(stable_label, stable_conf, voice_on, has_signal, raw_label)
                self.metrics.add("speech", timer.ms())

                timer = StageTimer()
                self.draw_overlay(frame, stable_label, stable_conf, source, voice_on)
                self.metrics.add("render", timer.ms())

                frame_e2e_ms = e2e_timer.ms()
                self.metrics.add_e2e(frame_e2e_ms)
                snap = self.metrics.snap()
                med = snap.get("e2e_median_ms", 0.0)
                print(f"\r[e2e median] {med:6.1f} ms", end="")

                cv2.imshow(win, frame)
                if use_side_guide:
                    if self.cfg.mode == "eye":
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
                if key == ord("l"):
                    if self.cfg.mode == "eye":
                        self.show_eye_landmarks = not self.show_eye_landmarks
                if key == ord("t") and self.cfg.mode in {"default", "teach", "aid"}:
                    self._teach_now(frame_data, eye_state=eye_state)
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if use_side_guide and cv2.getWindowProperty(guide_win, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            print()
            self.close()
            cv2.destroyAllWindows()
