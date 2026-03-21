import json
from pathlib import Path
import time
import uuid

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
            vec = self._norm_vec(vec)
            profile = item.get("profile", {})
            if not isinstance(profile, dict):
                profile = {}
            out[str(label)] = {
                "vec": vec,
                "count": int(item.get("count", 1)),
                "updated_at": int(item.get("updated_at", time.time() * 1000)),
                "profile": profile,
                "sign_id": str(item.get("sign_id", uuid.uuid4().hex[:12])),
            }
        self.data = out

    @staticmethod
    def _v2_to_v1(vec):
        # v2 layout (143): Lxy50 + Lz21 + Ls5 + Rxy50 + Rz21 + Rs5 + p3 + i1 + f3
        # v1 layout (101): Lxy50 + Ls5 + Rxy50 + Rs5 + p3 + i1 + f3
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.size != 143:
            return vec
        left_xy = vec[0:50]
        left_state = vec[71:76]
        right_xy = vec[76:126]
        right_state = vec[126:131]
        presence = vec[136:139]
        inter = vec[139:140]
        face = vec[140:143]
        return np.concatenate([left_xy, left_state, right_xy, right_state, presence, inter, face], axis=0).astype(np.float32)

    @classmethod
    def _norm_vec(cls, vec):
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.size == 143:
            return cls._v2_to_v1(vec)
        return vec

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {}
        for label, item in self.data.items():
            raw[label] = {
                "vec": np.asarray(item["vec"], dtype=np.float32).tolist(),
                "count": int(item.get("count", 1)),
                "updated_at": int(item.get("updated_at", time.time() * 1000)),
                "profile": dict(item.get("profile", {})),
                "sign_id": str(item.get("sign_id", "")),
            }
        self.path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    def add(self, label, vec, profile=None):
        key = str(label).strip().lower().replace(" ", "_")
        if not key:
            return None
        vec = self._norm_vec(vec)
        if vec.size == 0:
            return None

        profile = profile if isinstance(profile, dict) else {}

        now_ms = int(time.time() * 1000)
        if key in self.data:
            prev = self._norm_vec(np.asarray(self.data[key]["vec"], dtype=np.float32))
            if prev.size != vec.size:
                dim = int(min(prev.size, vec.size))
                prev = prev[:dim]
                vec = vec[:dim]
            count = int(self.data[key].get("count", 1))
            new_vec = ((prev * count) + vec) / float(count + 1)
            sign_id = str(self.data[key].get("sign_id", uuid.uuid4().hex[:12]))
            prev_profile = dict(self.data[key].get("profile", {}))
            merged_profile = dict(prev_profile)
            for k, v in profile.items():
                merged_profile[k] = v
            self.data[key] = {
                "vec": new_vec,
                "count": count + 1,
                "updated_at": now_ms,
                "profile": merged_profile,
                "sign_id": sign_id,
            }
        else:
            self.data[key] = {
                "vec": vec,
                "count": 1,
                "updated_at": now_ms,
                "profile": dict(profile),
                "sign_id": uuid.uuid4().hex[:12],
            }
        self.save()
        return str(self.data[key].get("sign_id", ""))

    @staticmethod
    def _profile_ok(stored, live):
        if not isinstance(stored, dict) or not isinstance(live, dict):
            return True
        s_count = int(stored.get("hand_count", 0))
        l_count = int(live.get("hand_count", 0))
        if s_count > 0 and l_count > 0 and s_count != l_count:
            return False

        s_face = int(stored.get("face_found", -1))
        l_face = int(live.get("face_found", -1))
        # Mild gate: if a sign was taught while face was hidden, reject when face is clearly visible.
        if s_face == 0 and l_face == 1:
            return False

        s_zone = str(stored.get("pose_zone", "free"))
        l_zone = str(live.get("pose_zone", "free"))
        # Stronger gate only for location-sensitive zones.
        if s_zone in {"face_front", "head_top"} and l_zone in {"face_front", "head_top"} and s_zone != l_zone:
            return False
        return True

    def best_match(self, vec, max_dist=0.23, profile=None):
        if not self.data:
            return None
        vec = self._norm_vec(np.asarray(vec, dtype=np.float32))
        if vec.size == 0:
            return None

        best_label = None
        best_dist = 1e9
        second_dist = 1e9
        for label, item in self.data.items():
            if not self._profile_ok(item.get("profile", {}), profile or {}):
                continue
            ref = self._norm_vec(np.asarray(item["vec"], dtype=np.float32))
            # Keep backward compatibility with older hand-only prototypes.
            dim = int(min(ref.size, vec.size))
            if dim <= 0:
                continue
            dist = float(np.linalg.norm(ref[:dim] - vec[:dim])) / float(np.sqrt(dim) + 1e-6)
            if dist < best_dist:
                second_dist = best_dist
                best_dist = dist
                best_label = label
            elif dist < second_dist:
                second_dist = dist

        if best_label is None or best_dist > max_dist:
            return None
        # If nearest and second-nearest are too close, skip to avoid confusion.
        if second_dist < 1e8:
            gap = float(second_dist - best_dist)
            ratio = float(second_dist / (best_dist + 1e-6))
            if gap < 0.008 and ratio < 1.12:
                return None

        conf = float(max(0.55, min(0.98, 1.0 - (best_dist / max_dist) * 0.55)))
        return Hit(best_label, conf, "prototype")


class AdaptiveSignDecoder:
    def __init__(self):
        self.store = PrototypeStore()

    @staticmethod
    def _dist(a, b):
        return float(np.linalg.norm(a[:2] - b[:2]))

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
        thumb_open = AdaptiveSignDecoder._dist(thumb_tip, idx_mcp) > AdaptiveSignDecoder._dist(thumb_ip, idx_mcp) * 1.08
        thumb_move = abs(float(thumb_tip[0] - thumb_ip[0])) > 0.035 or abs(float(thumb_tip[1] - thumb_ip[1])) > 0.035
        out["thumb"] = bool(thumb_open or thumb_move)
        return out

    @staticmethod
    def _encode_hand(hand):
        if hand is None or len(hand) < 21:
            return np.zeros((47,), dtype=np.float32)
        pts = np.asarray(hand[:, :2], dtype=np.float32)
        wrist = pts[0].copy()
        pts = pts - wrist
        scale = float(np.max(np.linalg.norm(pts, axis=1)))
        if scale > 1e-6:
            pts = pts / scale
        else:
            pts = np.zeros_like(pts)

        s = AdaptiveSignDecoder._state(hand)
        state_vec = np.asarray(
            [float(s["thumb"]), float(s["index"]), float(s["middle"]), float(s["ring"]), float(s["pinky"])],
            dtype=np.float32,
        )
        return np.concatenate([pts.reshape(-1), state_vec], axis=0).astype(np.float32)

    @staticmethod
    def _frame_profile(frame, eye_state=None):
        if frame is None:
            return {"hand_count": 0, "has_left": 0, "has_right": 0, "face_found": -1, "pose_zone": "free"}
        has_left = int(getattr(frame, "left", None) is not None)
        has_right = int(getattr(frame, "right", None) is not None)
        if eye_state is None:
            face_found = -1
        else:
            face_found = int(bool(getattr(eye_state, "face_found", False)))
        pose_zone = AdaptiveSignDecoder._pose_zone(frame, eye_state=eye_state)
        return {
            "hand_count": int(has_left + has_right),
            "has_left": has_left,
            "has_right": has_right,
            "face_found": face_found,
            "pose_zone": pose_zone,
        }

    @staticmethod
    def _primary_hand(frame):
        if frame is None:
            return None
        if getattr(frame, "left", None) is not None:
            return frame.left
        return getattr(frame, "right", None)

    @staticmethod
    def _pose_zone(frame, eye_state=None):
        hand = AdaptiveSignDecoder._primary_hand(frame)
        if hand is None or len(hand) < 21:
            return "free"
        if eye_state is None or (not bool(getattr(eye_state, "face_found", False))):
            return "free"

        pts = np.asarray(hand[:, :2], dtype=np.float32)
        hx = float((float(np.min(pts[:, 0])) + float(np.max(pts[:, 0]))) * 0.5)
        hy = float((float(np.min(pts[:, 1])) + float(np.max(pts[:, 1]))) * 0.5)

        k = list(getattr(eye_state, "keypoints", []) or [])
        if len(k) < 8:
            return "free"
        ex = float(sum(float(p[0]) for p in k[:8]) / 8.0)
        ey = float(sum(float(p[1]) for p in k[:8]) / 8.0)
        eye_w = float(max(float(k[1][0]), float(k[5][0])) - min(float(k[0][0]), float(k[4][0])))
        eye_h = float(max(float(k[3][1]), float(k[7][1])) - min(float(k[2][1]), float(k[6][1])))
        eye_w = max(eye_w, 0.05)
        eye_h = max(eye_h, 0.02)

        if hy < (ey - eye_h * 2.3):
            return "head_top"
        near_x = abs(hx - ex) <= (eye_w * 0.85)
        near_y = abs(hy - ey) <= (eye_h * 2.8)
        if near_x and near_y:
            return "face_front"
        return "free"

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
        if frame is None:
            return None

        left = getattr(frame, "left", None)
        right = getattr(frame, "right", None)
        if left is None and right is None:
            return None

        left_vec = self._encode_hand(left)
        right_vec = self._encode_hand(right)
        profile = self._frame_profile(frame, eye_state=eye_state)
        presence_vec = np.asarray(
            [
                float(profile.get("has_left", 0)),
                float(profile.get("has_right", 0)),
                float(profile.get("hand_count", 0)) / 2.0,
            ],
            dtype=np.float32,
        )

        inter_hand = 0.0
        if left is not None and right is not None:
            inter_hand = float(np.linalg.norm(np.asarray(left[0, :2]) - np.asarray(right[0, :2])))
        inter_vec = np.asarray([inter_hand], dtype=np.float32)

        face_vec = self._face_hint_vec(eye_state) * 0.28
        vec = np.concatenate([left_vec, right_vec, presence_vec, inter_vec, face_vec], axis=0).astype(np.float32)
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
            ip = hand[PIP["thumb"]]
            strong_fold = (
                float(hand[TIP["index"], 1]) > float(hand[MCP["index"], 1]) + 0.01
                and float(hand[TIP["middle"], 1]) > float(hand[MCP["middle"], 1]) + 0.01
                and float(hand[TIP["ring"], 1]) > float(hand[MCP["ring"], 1]) + 0.01
                and float(hand[TIP["pinky"], 1]) > float(hand[MCP["pinky"], 1]) + 0.01
            )
            dx = abs(float(tip[0] - wrist[0]))
            up = float(wrist[1] - tip[1])
            down = float(tip[1] - wrist[1])
            clear_vertical = dx < 0.12
            if strong_fold and clear_vertical and up > 0.10 and float(ip[1] - tip[1]) > 0.02:
                return Hit("yes", 0.92, "rules")
            if strong_fold and clear_vertical and down > 0.10 and float(tip[1] - ip[1]) > 0.02:
                return Hit("no", 0.92, "rules")
        if index and (not middle) and (not ring) and (not pinky):
            # Keep one strict so unknown handshapes don't get forced to "one".
            idx_tip = hand[TIP["index"]]
            idx_mcp = hand[MCP["index"]]
            mid_tip = hand[TIP["middle"]]
            mid_mcp = hand[MCP["middle"]]
            if float(idx_mcp[1] - idx_tip[1]) > 0.12 and float(mid_tip[1] - mid_mcp[1]) > 0.02:
                return Hit("one", 0.90, "rules")

        if index and middle and (not ring) and (not pinky):
            # Keep two strict so V-like random poses don't trigger too often.
            spread = self._dist(hand[TIP["index"]], hand[TIP["middle"]])
            ring_tip = hand[TIP["ring"]]
            ring_mcp = hand[MCP["ring"]]
            if spread > 0.040 and float(ring_tip[1] - ring_mcp[1]) > 0.02:
                return Hit("two", 0.90, "rules")

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
        profile = self._frame_profile(frame, eye_state=eye_state)
        return self.store.best_match(vec, profile=profile)

    def teach(self, frame, label, eye_state=None):
        vec = self.encode(frame, eye_state=eye_state)
        if vec is None:
            return False
        profile = self._frame_profile(frame, eye_state=eye_state)
        self.store.add(label, vec, profile=profile)
        return True

    def sign_id_for(self, label):
        key = str(label).strip().lower().replace(" ", "_")
        item = self.store.data.get(key)
        if not item:
            return ""
        return str(item.get("sign_id", ""))
