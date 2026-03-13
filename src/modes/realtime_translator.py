from collections import deque
from datetime import datetime
import statistics
import time

import cv2
import numpy as np

from core.hand_detection import CamCfg, CamStream, HandCfg, HandDetector, draw_hands
from core.speech_engine import Speaker
from core.stability import Hit, StableCfg, StableFilter
from dataset.recording import frame_to_vec
from modes.demo_mode import DEMO_SIGNS, DemoDecoder
from modes.emergency_mode import AID_SIGNS, AidDecoder
from model.model_manager import ModelHub
from model.sequence_model import load_model_for_runtime, predict_seq

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
    def __init__(self, cam_idx=0, w=960, h=540, fps=30, voice=True, seq_len=24, model_name=None, mode="hybrid"):
        self.cam_idx = cam_idx
        self.w = w
        self.h = h
        self.fps = fps
        self.voice = voice
        self.seq_len = seq_len
        self.model_name = model_name
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
        self.det = HandDetector(HandCfg(scale=0.65))
        self.rule_dec = RuleDecoder()
        self.demo_dec = DemoDecoder()
        self.aid_dec = AidDecoder()
        self.stable = StableFilter(StableCfg(win=7, min_conf=0.55, hold_sec=0.10))
        self.speaker = Speaker(rate=185, volume=1.0)
        self.metrics = RollMetrics()
        self.seq_buf = deque(maxlen=max(8, cfg.seq_len))

        hub = ModelHub()
        active_name = cfg.model_name or hub.active()
        self.model_name = active_name or "rules_only"
        if active_name:
            self.model = load_model_for_runtime(active_name)
        else:
            self.model = None

        self.last_spoken_label = ""
        self.last_spoken_ts = 0.0
        self.show_help = True

    def close(self):
        self.det.close()
        self.speaker.close()
        self.cam.close()

    def push_seq(self, frame_data):
        self.seq_buf.append(frame_to_vec(frame_data))

    def model_predict(self):
        if self.model is None:
            return None
        if len(self.seq_buf) < self.cfg.seq_len:
            return None
        items = list(self.seq_buf)
        seq = np.stack(items[-self.cfg.seq_len :], axis=0)
        label, conf = predict_seq(self.model, seq)
        if conf >= 0.60:
            return Hit(label, conf, "model:" + self.model_name)
        return None

    def decode(self, frame_data):
        if self.cfg.mode == "aid":
            return self.aid_dec.decode(frame_data)

        if self.cfg.mode == "demo":
            return self.demo_dec.decode(frame_data)

        if self.cfg.mode == "hybrid":
            hit = self.model_predict()
            if hit is not None:
                return hit
            demo_hit = self.demo_dec.decode(frame_data)
            if demo_hit is not None:
                return demo_hit
            return None

        hit = self.model_predict()
        if hit is not None:
            return hit
        return self.rule_dec.decode(frame_data)

    def maybe_speak(self, label, conf, voice_on):
        if not voice_on:
            return
        if label in {"unknown", "silence"}:
            return
        if conf < 0.55:
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
        cv2.putText(frame, "Keys: q/esc quit | v voice | r reset | h guide", (18, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
        if self.show_help:
            self.draw_help(frame)

    def draw_help(self, frame):
        if self.cfg.mode == "demo":
            box_w = 360
            box_h = 26 + (len(DEMO_SIGNS) * 20)
            x1 = max(8, frame.shape[1] - box_w - 10)
            y1 = 120
            x2 = x1 + box_w
            y2 = min(frame.shape[0] - 8, y1 + box_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
            cv2.putText(frame, "Demo signs", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            y = y1 + 40
            for i, sign in enumerate(DEMO_SIGNS, start=1):
                line = f"{i}. {sign.label.upper()} -> {sign.hint}"
                cv2.putText(frame, line, (x1 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
                y += 20
            return

        lines = [
            "Guide (hybrid mode)",
            "This mode uses model + demo signs.",
            "No quick-aid emergency shortcuts here.",
            "For fixed signs use demo mode.",
            "Press h to hide/show this help.",
        ]
        box_w = 360
        box_h = 24 + (len(lines) * 24)
        x1 = max(8, frame.shape[1] - box_w - 10)
        y1 = 120
        x2 = x1 + box_w
        y2 = y1 + box_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
        y = y1 + 24
        for i, text in enumerate(lines):
            color = (0, 255, 255) if i == 0 else (220, 220, 220)
            cv2.putText(frame, text, (x1 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
            y += 24

    @staticmethod
    def build_aid_panel():
        panel = np.zeros((360, 420, 3), dtype=np.uint8)
        panel[:] = (18, 18, 18)
        cv2.rectangle(panel, (0, 0), (419, 359), (80, 80, 80), 1)
        cv2.putText(panel, "Quick Aid Signs", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(panel, "Use one clear hand", (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        y = 82
        for i, sign in enumerate(AID_SIGNS, start=1):
            line = f"{i}. {sign.label.upper()} -> {sign.hint}"
            cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1)
            y += 38
        cv2.putText(panel, "Keys: q quit | v voice | r reset", (14, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        return panel

    def run(self):
        win = "SignifyAI"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.cfg.w, self.cfg.h)

        guide_win = "SignifyAI Guide"
        use_side_guide = self.cfg.mode == "aid"
        if use_side_guide:
            cv2.namedWindow(guide_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(guide_win, 420, 360)

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

                timer = StageTimer()
                frame_data = self.det.process(frame)
                self.metrics.add("perception", timer.ms())

                draw_hands(frame, frame_data)
                self.push_seq(frame_data)

                timer = StageTimer()
                raw_hit = self.decode(frame_data)
                self.metrics.add("decode", timer.ms())

                stable_label, stable_conf, _ = self.stable.update(raw_hit)
                if stable_label in {"unknown", "silence"} and raw_hit is not None:
                    stable_label = raw_hit.label
                    stable_conf = raw_hit.conf

                timer = StageTimer()
                self.maybe_speak(stable_label, stable_conf, voice_on)
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
                    cv2.imshow(guide_win, self.build_aid_panel())

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("v"):
                    voice_on = not voice_on
                if key == ord("r"):
                    self.stable.reset()
                    self.last_spoken_label = ""
                if key == ord("h"):
                    self.show_help = not self.show_help

                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if use_side_guide and cv2.getWindowProperty(guide_win, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            print()
            self.close()
            cv2.destroyAllWindows()
