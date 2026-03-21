import argparse
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = DATA_DIR / "models"
LIVE_TEACH_DIR = DATA_DIR / "landmarks" / "raw" / "live_teach"
RAW_DIR = DATA_DIR / "landmarks" / "raw"
VERSIONS_DIR = DATA_DIR / "landmarks" / "versions"


def setup_dirs():
    (DATA_DIR / "exports").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "external_global").mkdir(parents=True, exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_TEACH_DIR.mkdir(parents=True, exist_ok=True)

    proto = MODELS_DIR / "sign_prototypes.json"
    if not proto.exists():
        proto.write_text("{}\n", encoding="utf-8")

    gitkeep = LIVE_TEACH_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")

    landmarks_gitkeep = DATA_DIR / "landmarks" / ".gitkeep"
    if not landmarks_gitkeep.exists():
        landmarks_gitkeep.write_text("", encoding="utf-8")

    models_gitkeep = MODELS_DIR / ".gitkeep"
    if not models_gitkeep.exists():
        models_gitkeep.write_text("", encoding="utf-8")

def show_paths():
    required_paths = [
        DATA_DIR / "models" / "registry.json",
        DATA_DIR / "models" / "sign_prototypes.json",
        DATA_DIR / "landmarks" / "raw" / "live_teach",
        DATA_DIR / "landmarks" / "versions",
        DATA_DIR / "external",
        DATA_DIR / "models" / "hand_landmarker.task",
        DATA_DIR / "models" / "face_landmarker.task",
    ]

    registry = MODELS_DIR / "registry.json"
    if not registry.exists():
        registry.write_text(json.dumps({"active_model": None, "history": []}, indent=2) + "\n", encoding="utf-8")

    print("[data] Runtime path status:")
    for p in required_paths:
        state = "ok" if p.exists() else "missing"
        print(f"[{state}] {p.as_posix()}")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap local SignifyAI runtime folders/files")
    _ = parser.parse_args()

    setup_dirs()
    show_paths()
    print("[done] Local data bootstrap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
