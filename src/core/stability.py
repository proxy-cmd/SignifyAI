from collections import Counter, deque
import time


class Hit:
    def __init__(self, label, conf, src="rules"):
        self.label = label
        self.conf = conf
        self.src = src


class StableCfg:
    def __init__(self, win=7, min_conf=0.60, hold_sec=0.12):
        self.win = win
        self.min_conf = min_conf
        self.hold_sec = hold_sec


class StableFilter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.labels = deque(maxlen=cfg.win)
        self.scores = deque(maxlen=cfg.win)
        self.pending = "unknown"
        self.pending_ts = time.time()
        self.final = "unknown"

    def _vote(self):
        if not self.labels:
            return "unknown"
        count = Counter(self.labels)
        return count.most_common(1)[0][0]

    def update(self, hit):
        if hit is None:
            self.labels.append("silence")
            self.scores.append(0.0)
        else:
            if hit.conf >= self.cfg.min_conf:
                label = hit.label
            else:
                label = "unknown"
            self.labels.append(label)
            self.scores.append(hit.conf)

        voted = self._vote()
        now = time.time()
        if voted != self.pending:
            self.pending = voted
            self.pending_ts = now

        if (now - self.pending_ts) >= self.cfg.hold_sec:
            self.final = self.pending

        if self.scores:
            avg = float(sum(self.scores) / len(self.scores))
        else:
            avg = 0.0

        if self.final == voted:
            state = "stable"
        else:
            state = "pending"
        return self.final, avg, state

    def reset(self):
        self.labels.clear()
        self.scores.clear()
        self.pending = "unknown"
        self.final = "unknown"
        self.pending_ts = time.time()
