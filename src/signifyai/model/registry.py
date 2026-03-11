from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time


@dataclass
class ModelRegistry:
    path: Path = Path("data/models/registry.json")

    def load(self) -> dict:
        if not self.path.exists():
            return {"active_model": None, "history": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def active(self) -> str | None:
        return self.load().get("active_model")

    def snapshot(self) -> dict:
        return self.load()

    def history_count(self) -> int:
        data = self.load()
        return len(data.get("history", []))

    def promote_model(self, model_name: str, notes: str = "") -> dict:
        payload = self.load()
        payload["active_model"] = model_name
        payload.setdefault("history", []).append(
            {
                "model": model_name,
                "timestamp_ms": int(time.time() * 1000),
                "notes": notes,
            }
        )
        self.save(payload)
        return payload

    # Backward-compatible names.
    _load = load
    _save = save
