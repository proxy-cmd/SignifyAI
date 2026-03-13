from pathlib import Path
import json
import time


class ModelHub:
    def __init__(self, path=Path("data/models/registry.json")):
        self.path = path

    def _load(self):
        # read registry file; if missing, return default structure
        if not self.path.exists():
            return {"active_model": None, "history": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def active(self):
        return self._load().get("active_model")

    def history_count(self):
        return len(self._load().get("history", []))

    def promote(self, name, notes=""):
        # set active model and append history entry
        data = self._load()
        data["active_model"] = name
        data.setdefault("history", []).append(
            {
                "model": name,
                "timestamp_ms": int(time.time() * 1000),
                "notes": notes,
            }
        )
        self._save(data)
        return data
