from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import zipfile

from .config import PATHS


@dataclass
class ReleaseBundleConfig:
    out_dir: Path = PATHS.project_root / "dist"
    include_videos: bool = False


def _safe_add(zf: zipfile.ZipFile, path: Path, arcname: str) -> int:
    if path.exists() and path.is_file():
        zf.write(path, arcname=arcname)
        return 1
    return 0


def build_release_bundle(cfg: ReleaseBundleConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = cfg.out_dir / f"signifyai_release_{ts}.zip"

    project_root = PATHS.project_root
    models_dir = project_root / "models"
    data_processed = project_root / "data" / "processed"

    added = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Core model artifacts
        for name in [
            "gesture_model.joblib",
            "labels.json",
            "model_metadata.json",
            "temporal_gesture_model.joblib",
            "temporal_labels.json",
            "temporal_model_metadata.json",
        ]:
            added += _safe_add(zf, models_dir / name, f"models/{name}")

        # Reports and logs
        for name in ["session_report.md", "session_summary.json", "session_log.csv", "confusion_matrix.csv"]:
            added += _safe_add(zf, data_processed / name, f"data/processed/{name}")

        # Screenshots
        shots_dir = data_processed / "screenshots"
        if shots_dir.exists():
            for p in shots_dir.glob("*.png"):
                zf.write(p, arcname=f"data/processed/screenshots/{p.name}")
                added += 1

        # Optional videos
        if cfg.include_videos:
            vids_dir = data_processed / "videos"
            if vids_dir.exists():
                for p in vids_dir.glob("*.mp4"):
                    zf.write(p, arcname=f"data/processed/videos/{p.name}")
                    added += 1

        # Readme for judges
        readme = project_root / "README.md"
        added += _safe_add(zf, readme, "README.md")

    if added == 0:
        raise RuntimeError("No artifacts found to bundle. Run training/realtime/report first.")
    return zip_path

