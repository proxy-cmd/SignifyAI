import numpy as np

from core.stability import Hit

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


class AidSign:
    def __init__(self, label, hint):
        self.label = label
        self.hint = hint


AID_SIGNS = [
    AidSign("need_water", "Index finger only"),
    AidSign("need_food", "Index + middle"),
    AidSign("need_toilet", "Pinky finger only"),
    AidSign("call_family", "Three fingers (index+middle+ring)"),
    AidSign("hospital_help", "Open palm"),
    AidSign("emergency", "Closed fist"),
    AidSign("severe_pain", "Index + pinky"),
    AidSign("cannot_breathe", "Thumb + index"),
    AidSign("bleeding", "Index + ring"),
    AidSign("head_injury", "Middle finger only"),
    AidSign("chest_pain", "Ring finger only"),
    AidSign("yes", "Thumb up"),
    AidSign("no", "Thumb down"),
]


class AidDecoder:
    @staticmethod
    def _state(hand):
        out = {}
        for name in ("index", "middle", "ring", "pinky"):
            tip_y = float(hand[TIP[name], 1])
            pip_y = float(hand[PIP[name], 1])
            mcp_y = float(hand[MCP[name], 1])
            out[name] = bool(tip_y < (pip_y - 0.018) and pip_y < (mcp_y - 0.003))

        thumb_tip = hand[TIP["thumb"]]
        thumb_ip = hand[PIP["thumb"]]
        idx_mcp = hand[MCP["index"]]
        thumb_open = (abs(float(thumb_tip[0] - thumb_ip[0])) > 0.035) or (float(np.linalg.norm(thumb_tip[:2] - idx_mcp[:2])) > 0.14)
        out["thumb"] = bool(thumb_open)
        return out

    def decode(self, frame):
        hands = [h for h in (frame.left, frame.right) if h is not None]
        if not hands:
            return None

        hand = hands[0]
        s = self._state(hand)
        folded = (not s["index"]) and (not s["middle"]) and (not s["ring"]) and (not s["pinky"])

        if s["thumb"] and folded:
            thumb_tip = hand[TIP["thumb"]]
            wrist = hand[0]
            if thumb_tip[1] < wrist[1] - 0.05:
                return Hit("yes", 0.92, "aid")
            if thumb_tip[1] > wrist[1] + 0.05:
                return Hit("no", 0.92, "aid")

        if s["thumb"] and s["index"] and not s["middle"] and not s["ring"] and not s["pinky"]:
            return Hit("cannot_breathe", 0.91, "aid")
        if s["index"] and s["pinky"] and not s["middle"] and not s["ring"]:
            return Hit("severe_pain", 0.91, "aid")
        if s["index"] and s["ring"] and not s["middle"] and not s["pinky"]:
            return Hit("bleeding", 0.90, "aid")
        if s["middle"] and not s["index"] and not s["ring"] and not s["pinky"]:
            return Hit("head_injury", 0.90, "aid")
        if s["ring"] and not s["index"] and not s["middle"] and not s["pinky"]:
            return Hit("chest_pain", 0.90, "aid")

        if s["index"] and not s["middle"] and not s["ring"] and not s["pinky"]:
            return Hit("need_water", 0.92, "aid")
        if s["index"] and s["middle"] and not s["ring"] and not s["pinky"]:
            return Hit("need_food", 0.92, "aid")
        if s["pinky"] and not s["index"] and not s["middle"] and not s["ring"]:
            return Hit("need_toilet", 0.92, "aid")
        if s["index"] and s["middle"] and s["ring"] and not s["pinky"]:
            return Hit("call_family", 0.90, "aid")

        open_palm = s["thumb"] and s["index"] and s["middle"] and s["ring"] and s["pinky"]
        if open_palm:
            return Hit("hospital_help", 0.90, "aid")

        if folded:
            return Hit("emergency", 0.90, "aid")

        return None
