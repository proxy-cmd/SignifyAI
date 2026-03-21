from pathlib import Path
import json
import time
import uuid


class SessionTelemetry:
    def __init__(self, out_dir=Path("data/logs/realtime"), session_id=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = str(session_id or f"rt_{uuid.uuid4().hex[:12]}")
        self.path = self.out_dir / f"{self.session_id}.jsonl"
        self._fp = self.path.open("a", encoding="utf-8")

    def log(self, payload):
        row = dict(payload or {})
        row.setdefault("ts_ms", int(time.time() * 1000))
        row.setdefault("session_id", self.session_id)
        self._fp.write(json.dumps(row, ensure_ascii=True) + "\n")
        self._fp.flush()

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass
