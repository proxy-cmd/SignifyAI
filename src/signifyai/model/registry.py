from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time


@dataclass
class ModelRegistry:
    path: Path = Path("data/models/registry.json")

    def _load(self) -> dict:
        if not self.path.exists():
            return {"active_model": None, "history": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def active(self) -> str | None:
        return self._load().get("active_model")

    def snapshot(self) -> dict:
        return self._load()

    def history_count(self) -> int:
        payload = self._load()
        return len(payload.get("history", []))

    def promote_model(self, model_name: str, notes: str = "") -> dict:
        payload = self._load()
        payload["active_model"] = model_name
        payload.setdefault("history", []).append(
            {
                "model": model_name,
                "timestamp_ms": int(time.time() * 1000),
                "notes": notes,
            }
        )
        self._save(payload)
        return payload
