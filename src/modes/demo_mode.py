from collections import deque

import numpy as np

from core.stability import Hit

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


class DemoSign:
    def __init__(self, label, hint):
        self.label = label
        self.hint = hint


DEMO_SIGNS = [
    DemoSign("hello", "Open palm + wave"),
    DemoSign("thank_you", "Open palm moves down/out (demo approximation)"),
    DemoSign("please", "Flat hand + small circle (demo approximation)"),
    DemoSign("sorry", "Closed fist + small circle (demo approximation)"),
    DemoSign("yes", "Thumb up"),
    DemoSign("no", "Thumb down"),
    DemoSign("smile", "Thumb + index touch, other 3 fingers up"),
    DemoSign("call_me", "Thumb + pinky up"),
    DemoSign("one", "Only index up"),
    DemoSign("two", "Index + middle up"),
    DemoSign("three", "Index + middle + ring up"),
    DemoSign("four", "Four fingers up (no thumb)"),
    DemoSign("five", "All five fingers up"),
    DemoSign("stop", "Closed fist"),
]


class DemoDecoder:
    def __init__(self):
        self.hist = deque(maxlen=14)
        self.path = deque(maxlen=16)

    @staticmethod
    def _dist(a, b):
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _state(self, hand):
        out = {}
        for name in ("index", "middle", "ring", "pinky"):
            tip_y = float(hand[TIP[name], 1])
            pip_y = float(hand[PIP[name], 1])
            mcp_y = float(hand[MCP[name], 1])
            out[name] = bool(tip_y < (pip_y - 0.018) and pip_y < (mcp_y - 0.003))

        thumb_tip = hand[TIP["thumb"]]
        thumb_ip = hand[PIP["thumb"]]
        idx_mcp = hand[MCP["index"]]
        thumb_open = self._dist(thumb_tip, idx_mcp) > self._dist(thumb_ip, idx_mcp) * 1.08
        thumb_move = abs(float(thumb_tip[0] - thumb_ip[0])) > 0.035 or abs(float(thumb_tip[1] - thumb_ip[1])) > 0.035
        out["thumb"] = bool(thumb_open or thumb_move)
        return out

    @staticmethod
    def _count_open(s):
        return int(s["thumb"]) + int(s["index"]) + int(s["middle"]) + int(s["ring"]) + int(s["pinky"])

    def _is_wave(self, hand, open_palm):
        if not open_palm:
            self.hist.clear()
            return False
        self.hist.append(float(hand[0, 0]))
        if len(self.hist) < 8:
            return False
        xs = np.asarray(self.hist, dtype=np.float32)
        amp = float(xs.max() - xs.min())
        dx = np.diff(xs)
        dirs = np.sign(dx)
        dirs[np.abs(dx) < 0.01] = 0
        dirs = dirs[dirs != 0]
        if len(dirs) < 3:
            return False
        turns = int(np.sum(np.abs(np.diff(dirs)) > 0))
        return amp > 0.10 and turns >= 2

    def _is_circle_motion(self):
        if len(self.path) < 10:
            return False
        pts = np.asarray(self.path, dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]

        amp_x = float(xs.max() - xs.min())
        amp_y = float(ys.max() - ys.min())
        if amp_x < 0.03 or amp_y < 0.03:
            return False

        dx = np.diff(xs)
        dy = np.diff(ys)
        dirs_x = np.sign(dx)
        dirs_y = np.sign(dy)
        turns_x = int(np.sum(np.abs(np.diff(dirs_x)) > 0))
        turns_y = int(np.sum(np.abs(np.diff(dirs_y)) > 0))
        return turns_x >= 2 and turns_y >= 2

    def _is_thank_you_motion(self, open_palm):
        if not open_palm:
            return False
        if len(self.path) < 6:
            return False
        start_x, start_y = self.path[0]
        end_x, end_y = self.path[-1]
        move_y = float(end_y - start_y)
        move_x = float(abs(end_x - start_x))
        return start_y < 0.50 and move_y > 0.05 and move_x < 0.12

    def decode(self, frame):
        hands = [h for h in (frame.left, frame.right) if h is not None]
        if not hands:
            self.hist.clear()
            self.path.clear()
            return None

        hand = hands[0]
        self.path.append((float(hand[0, 0]), float(hand[0, 1])))
        s = self._state(hand)
        n = self._count_open(s)

        folded = not s["index"] and not s["middle"] and not s["ring"] and not s["pinky"]
        if folded and s["thumb"]:
            thumb_tip = hand[TIP["thumb"]]
            wrist = hand[0]
            if thumb_tip[1] < wrist[1] - 0.05:
                return Hit("yes", 0.92, "demo")
            if thumb_tip[1] > wrist[1] + 0.05:
                return Hit("no", 0.92, "demo")

        open_palm = s["index"] and s["middle"] and s["ring"] and s["pinky"] and s["thumb"]
        if self._is_thank_you_motion(open_palm=open_palm):
            return Hit("thank_you", 0.92, "demo")
        if open_palm and self._is_wave(hand, open_palm=True):
            return Hit("hello", 0.94, "demo")
        if open_palm:
            return Hit("five", 0.92, "demo")

        # please: flat hand (thumb folded) with circular movement
        flat_hand = (not s["thumb"]) and s["index"] and s["middle"] and s["ring"] and s["pinky"]
        if flat_hand and self._is_circle_motion():
            return Hit("please", 0.90, "demo")

        # sorry: closed fist with small circle
        if folded and (not s["thumb"]) and self._is_circle_motion():
            return Hit("sorry", 0.90, "demo")

        # thumb + index circle, other fingers open
        thumb_tip = hand[TIP["thumb"]]
        index_tip = hand[TIP["index"]]
        tip_dist = self._dist(thumb_tip, index_tip)
        if tip_dist < 0.050 and s["middle"] and s["ring"] and s["pinky"]:
            return Hit("smile", 0.92, "demo")

        # shaka / call-me style
        if s["thumb"] and s["pinky"] and (not s["index"]) and (not s["middle"]) and (not s["ring"]):
            return Hit("call_me", 0.91, "demo")

        if n <= 1 and folded:
            return Hit("stop", 0.90, "demo")
        if s["index"] and not s["middle"] and not s["ring"] and not s["pinky"]:
            return Hit("one", 0.90, "demo")
        if s["index"] and s["middle"] and not s["ring"] and not s["pinky"]:
            return Hit("two", 0.90, "demo")
        if s["index"] and s["middle"] and s["ring"] and not s["pinky"]:
            return Hit("three", 0.90, "demo")
        if s["index"] and s["middle"] and s["ring"] and s["pinky"] and not s["thumb"]:
            return Hit("four", 0.90, "demo")

        return None
