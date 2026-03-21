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
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"active_model": None, "history": []}

        if "history" not in data or not isinstance(data.get("history"), list):
            data["history"] = []
        if "active_model" not in data:
            data["active_model"] = None
        return data

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

    def last_promoted(self):
        data = self._load()
        history = data.get("history", [])
        if not history:
            return None
        return history[-1]

    def rollback(self, notes=""):
        data = self._load()
        history = data.get("history", [])
        if len(history) < 2:
            return {"ok": False, "reason": "No previous model in history"}

        prev = history[-2].get("model")
        data["active_model"] = prev
        data.setdefault("history", []).append(
            {
                "model": prev,
                "timestamp_ms": int(time.time() * 1000),
                "notes": notes or "manual rollback",
                "rollback": True,
            }
        )
        self._save(data)
        return {"ok": True, "active_model": prev}
