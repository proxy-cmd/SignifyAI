from __future__ import annotations

import json
from pathlib import Path

from .config import DEFAULT_PHRASE_MAP_PATH


def load_phrase_map(path: Path = DEFAULT_PHRASE_MAP_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            out: dict[str, str] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str):
                    key = k.strip().lower().replace(" ", "_")
                    if key:
                        out[key] = v.strip()
            return out
    except Exception:
        return {}
    return {}


def save_phrase_map(mapping: dict[str, str], path: Path = DEFAULT_PHRASE_MAP_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {k.strip().lower().replace(" ", "_"): v.strip() for k, v in mapping.items() if k and v}
    path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def set_phrase(label: str, text: str, path: Path = DEFAULT_PHRASE_MAP_PATH) -> None:
    mapping = load_phrase_map(path)
    key = label.strip().lower().replace(" ", "_")
    if not key:
        raise ValueError("Label cannot be empty.")
    if not text.strip():
        raise ValueError("Phrase text cannot be empty.")
    mapping[key] = text.strip()
    save_phrase_map(mapping, path)


def get_phrase(label: str, path: Path = DEFAULT_PHRASE_MAP_PATH) -> str | None:
    key = label.strip().lower().replace(" ", "_")
    return load_phrase_map(path).get(key)

