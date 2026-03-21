from core.stability import Hit


class EyeAssistDecoder:
    def __init__(self):
        self.closed = False
        self.close_start_ms = None
        self.close_min_ear = 1.0
        self.last_emit_ms = -1_000_000_000
        self.pending_yes_ts = None
        self.recent_blinks = []
        self.left_hold_start_ms = None
        self.right_hold_start_ms = None
        self.up_hold_start_ms = None

        # Standard profile: balanced between comfort and control.
        self.ear_close = 0.215
        self.ear_open = 0.245
        self.ear_hard_close = 0.185
        self.single_blink_min_ms = 45
        self.long_blink_ms = 750
        self.single_yes_delay_ms = 320
        self.triple_blink_window_ms = 1400
        self.left_gaze_max = 0.40
        self.right_gaze_min = 0.60
        self.up_gaze_max = 0.42
        self.gaze_hold_ms = 500
        self.emit_cooldown_ms = 700

    def _can_emit(self, now_ms):
        return (int(now_ms) - int(self.last_emit_ms)) >= self.emit_cooldown_ms

    def _emit(self, label, conf, now_ms):
        self.last_emit_ms = int(now_ms)
        self.pending_yes_ts = None
        self.left_hold_start_ms = None
        self.right_hold_start_ms = None
        self.up_hold_start_ms = None
        return Hit(label, conf, "eye")

    def _add_short_blink(self, now_ms):
        self.recent_blinks.append(int(now_ms))
        cut = int(now_ms) - self.triple_blink_window_ms
        self.recent_blinks = [t for t in self.recent_blinks if t >= cut]

    def _is_triple_blink(self):
        return len(self.recent_blinks) >= 3

    def decode(self, eye_state):
        if eye_state is None:
            return None

        now_ms = int(getattr(eye_state, "ts_ms", 0) or 0)
        if now_ms <= 0:
            return None

        if not bool(getattr(eye_state, "face_found", False)):
            self.closed = False
            self.close_start_ms = None
            self.close_min_ear = 1.0
            self.pending_yes_ts = None
            self.recent_blinks.clear()
            self.left_hold_start_ms = None
            self.right_hold_start_ms = None
            self.up_hold_start_ms = None
            return None

        ear = (float(getattr(eye_state, "left_ear", 0.0)) + float(getattr(eye_state, "right_ear", 0.0))) * 0.5
        gaze_x = float(getattr(eye_state, "gaze_x", 0.5))
        gaze_y = float(getattr(eye_state, "gaze_y", 0.5))
        center_x = abs(gaze_x - 0.5) <= 0.16

        # Commit pending single-blink -> yes if user did not continue into triple-blink.
        if self.pending_yes_ts is not None and (now_ms - int(self.pending_yes_ts)) >= self.single_yes_delay_ms:
            if self._can_emit(now_ms):
                return self._emit("yes", 0.90, now_ms)
            self.pending_yes_ts = None

        # Blink transition logic.
        # Guard against false triggers when user intentionally looks down.
        allow_blink_arm = gaze_y < 0.74
        if ear <= self.ear_close and allow_blink_arm:
            if not self.closed:
                self.closed = True
                self.close_start_ms = now_ms
                self.close_min_ear = ear
            else:
                self.close_min_ear = min(float(self.close_min_ear), float(ear))
            return None

        if self.closed and ear >= self.ear_open:
            self.closed = False
            start_ms = self.close_start_ms if self.close_start_ms is not None else now_ms
            blink_ms = max(0, now_ms - int(start_ms))
            self.close_start_ms = None
            min_ear = float(self.close_min_ear)
            self.close_min_ear = 1.0

            valid_blink = bool(min_ear <= self.ear_hard_close)
            if not valid_blink:
                return None

            if blink_ms >= self.long_blink_ms and self._can_emit(now_ms):
                self.pending_yes_ts = None
                self.recent_blinks.clear()
                return self._emit("emergency", 0.95, now_ms)

            if blink_ms >= self.single_blink_min_ms:
                self._add_short_blink(now_ms)
                if self._is_triple_blink() and self._can_emit(now_ms):
                    self.recent_blinks.clear()
                    self.pending_yes_ts = None
                    return self._emit("need_water", 0.90, now_ms)
                self.pending_yes_ts = now_ms

        # Directional holds for additional intents.
        if gaze_x <= self.left_gaze_max and ear >= self.ear_open:
            if self.left_hold_start_ms is None:
                self.left_hold_start_ms = now_ms
            if (now_ms - self.left_hold_start_ms) >= self.gaze_hold_ms and self._can_emit(now_ms):
                return self._emit("no", 0.86, now_ms)
        else:
            self.left_hold_start_ms = None

        if gaze_x >= self.right_gaze_min and ear >= self.ear_open:
            if self.right_hold_start_ms is None:
                self.right_hold_start_ms = now_ms
            if (now_ms - self.right_hold_start_ms) >= self.gaze_hold_ms and self._can_emit(now_ms):
                return self._emit("call_family", 0.86, now_ms)
        else:
            self.right_hold_start_ms = None

        if gaze_y <= self.up_gaze_max and center_x and ear >= self.ear_open:
            if self.up_hold_start_ms is None:
                self.up_hold_start_ms = now_ms
            if (now_ms - self.up_hold_start_ms) >= self.gaze_hold_ms and self._can_emit(now_ms):
                return self._emit("need_food", 0.86, now_ms)
        else:
            self.up_hold_start_ms = None

        return None
