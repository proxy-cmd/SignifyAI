import json
from pathlib import Path
import time

import numpy as np

from core.stability import Hit

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


class PrototypeStore:
    def __init__(self, path=Path("data/models/sign_prototypes.json")):
        self.path = Path(path)
        self.data = {}
        self.load()

    def load(self):
        if not self.path.exists():
            self.data = {}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}
            return
        out = {}
        for label, item in raw.items():
            vec = np.asarray(item.get("vec", []), dtype=np.float32)
            if vec.size == 0:
                continue
            out[str(label)] = {
                "vec": vec,
                "count": int(item.get("count", 1)),
                "updated_at": int(item.get("updated_at", time.time() * 1000)),
            }
        self.data = out

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {}
        for label, item in self.data.items():
            raw[label] = {
                "vec": np.asarray(item["vec"], dtype=np.float32).tolist(),
                "count": int(item.get("count", 1)),
                "updated_at": int(item.get("updated_at", time.time() * 1000)),
            }
        self.path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    def add(self, label, vec):
        key = str(label).strip().lower().replace(" ", "_")
        if not key:
            return
        vec = np.asarray(vec, dtype=np.float32)
        if vec.size == 0:
            return

        now_ms = int(time.time() * 1000)
        if key in self.data:
            prev = np.asarray(self.data[key]["vec"], dtype=np.float32)
            count = int(self.data[key].get("count", 1))
            new_vec = ((prev * count) + vec) / float(count + 1)
            self.data[key] = {"vec": new_vec, "count": count + 1, "updated_at": now_ms}
        else:
            self.data[key] = {"vec": vec, "count": 1, "updated_at": now_ms}
        self.save()

    def best_match(self, vec, max_dist=0.18):
        if not self.data:
            return None
        vec = np.asarray(vec, dtype=np.float32)
        if vec.size == 0:
            return None

        best_label = None
        best_dist = 1e9
        for label, item in self.data.items():
            ref = np.asarray(item["vec"], dtype=np.float32)
            # Keep backward compatibility with older hand-only prototypes.
            dim = int(min(ref.size, vec.size))
            if dim <= 0:
                continue
            dist = float(np.linalg.norm(ref[:dim] - vec[:dim])) / float(np.sqrt(dim) + 1e-6)
            if dist < best_dist:
                best_dist = dist
                best_label = label

        if best_label is None or best_dist > max_dist:
            return None

        conf = float(max(0.55, min(0.98, 1.0 - (best_dist / max_dist) * 0.55)))
        return Hit(best_label, conf, "prototype")


class AdaptiveSignDecoder:
    def __init__(self):
        self.store = PrototypeStore()

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
    def _primary_hand(frame):
        if frame is None:
            return None
        if getattr(frame, "left", None) is not None:
            return frame.left
        return getattr(frame, "right", None)

    @staticmethod
    def _face_hint_vec(eye_state):
        # Very low-weight face hint: helps disambiguation, never hard-requires expressions.
        if eye_state is None or (not bool(getattr(eye_state, "face_found", False))):
            return np.asarray([0.0, 0.5, 0.5], dtype=np.float32)
        ear = (float(getattr(eye_state, "left_ear", 0.0)) + float(getattr(eye_state, "right_ear", 0.0))) * 0.5
        gx = float(np.clip(getattr(eye_state, "gaze_x", 0.5), 0.0, 1.0))
        gy = float(np.clip(getattr(eye_state, "gaze_y", 0.5), 0.0, 1.0))
        return np.asarray([ear, gx, gy], dtype=np.float32)

    def encode(self, frame, eye_state=None):
        hand = self._primary_hand(frame)
        if hand is None or len(hand) < 21:
            return None

        pts = np.asarray(hand[:, :2], dtype=np.float32)
        wrist = pts[0].copy()
        pts = pts - wrist
        scale = float(np.max(np.linalg.norm(pts, axis=1)))
        if scale < 1e-6:
            return None
        pts /= scale

        s = self._state(hand)
        state_vec = np.asarray(
            [float(s["thumb"]), float(s["index"]), float(s["middle"]), float(s["ring"]), float(s["pinky"])],
            dtype=np.float32,
        )
        face_vec = self._face_hint_vec(eye_state) * 0.12
        vec = np.concatenate([pts.reshape(-1), state_vec, face_vec], axis=0).astype(np.float32)
        return vec

    def _decode_rules(self, frame):
        hand = self._primary_hand(frame)
        if hand is None or len(hand) < 21:
            return None

        s = self._state(hand)
        thumb = bool(s["thumb"])
        index = bool(s["index"])
        middle = bool(s["middle"])
        ring = bool(s["ring"])
        pinky = bool(s["pinky"])

        folded = (not index) and (not middle) and (not ring) and (not pinky)
        tip_dist = self._dist(hand[TIP["thumb"]], hand[TIP["index"]])
        im_tip_dist = self._dist(hand[TIP["index"]], hand[TIP["middle"]])
        wrist = hand[0]

        # Frequent daily-use signs
        if thumb and folded:
            tip = hand[TIP["thumb"]]
            if tip[1] < wrist[1] - 0.05:
                return Hit("yes", 0.92, "rules")
            if tip[1] > wrist[1] + 0.05:
                return Hit("no", 0.92, "rules")
        if index and (not middle) and (not ring) and (not pinky):
            return Hit("need_water", 0.90, "rules")
        if index and middle and (not ring) and (not pinky):
            return Hit("need_food", 0.90, "rules")

        # Letter rules (high-frequency handshapes)
        if pinky and (not index) and (not middle) and (not ring) and (not thumb):
            return Hit("i", 0.91, "letters")
        if thumb and pinky and (not index) and (not middle) and (not ring):
            return Hit("y", 0.90, "letters")
        if thumb and index and (not middle) and (not ring) and (not pinky):
            idx_tip = hand[TIP["index"]]
            if idx_tip[1] > wrist[1]:
                return Hit("q", 0.84, "letters")
            return Hit("l", 0.90, "letters")
        if index and middle and (not ring) and (not pinky):
            if im_tip_dist < 0.028:
                return Hit("r", 0.84, "letters")
            return Hit("v", 0.88, "letters")
        if index and middle and ring and (not pinky):
            return Hit("w", 0.88, "letters")

        return None

    def decode(self, frame, eye_state=None):
        rule_hit = self._decode_rules(frame)
        if rule_hit is not None:
            return rule_hit

        vec = self.encode(frame, eye_state=eye_state)
        if vec is None:
            return None
        return self.store.best_match(vec)

    def teach(self, frame, label, eye_state=None):
        vec = self.encode(frame, eye_state=eye_state)
        if vec is None:
            return False
        self.store.add(label, vec)
        return True
