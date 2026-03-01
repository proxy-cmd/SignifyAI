from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Callable

import cv2
import numpy as np

from .analytics import append_event
from .config import FEATURE_SIZE, PATHS
from .dataset import save_records
from .doctor import run_doctor
from .preflight import run_preflight
from .production_train import ProductionTrainConfig, run_production_training
from .prototype_adapt import adapt_sign_from_images, extract_points_from_image
from .release import ReleaseBundleConfig, build_release_bundle
from .report import ReportConfig, build_session_report
from .sequence_dataset import build_sequence_dataset_from_frames
from .temporal_model import TemporalTrainConfig, run_temporal_training
from .train import TrainConfig, run_training
from .video_infer import VideoInferConfig, run_video_inference


def _is_tensorflow_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


@dataclass
class QACheck:
    name: str
    status: str  # PASS | FAIL | SKIP
    seconds: float
    detail: str


@dataclass
class QAConfig:
    out_json: Path = PATHS.data_processed / "qa_validation_report.json"
    include_pytest: bool = True
    include_cli_help_checks: bool = True
    include_release_bundle_check: bool = True


def _run_check(name: str, fn: Callable[[], str]) -> QACheck:
    t0 = time.time()
    try:
        detail = fn()
        return QACheck(name=name, status="PASS", seconds=round(time.time() - t0, 3), detail=detail)
    except Exception as ex:
        return QACheck(name=name, status="FAIL", seconds=round(time.time() - t0, 3), detail=str(ex))


def _run_subprocess(command: list[str], cwd: Path, timeout_sec: int = 120) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
    )
    return int(proc.returncode), (proc.stdout or "")


def _command_help_checks(project_root: Path) -> list[QACheck]:
    checks: list[QACheck] = []
    main_py = project_root / "src" / "main.py"
    python = sys.executable
    commands = [
        "collect",
        "data-help",
        "collect-seq",
        "calibrate",
        "train",
        "train-deep",
        "train-all",
        "check-dataset",
        "import-kaggle",
        "import-url",
        "import-zip",
        "build-image-dataset",
        "build-video-dataset",
        "build-seq-dataset",
        "train-seq",
        "set-phrase",
        "list-phrases",
        "record-combo",
        "teach-sign",
        "run",
        "report",
        "model-report",
        "validate-all",
        "final-test",
        "preflight",
        "doctor",
        "bootstrap-ml",
        "bootstrap-url",
        "benchmark",
        "release-bundle",
        "infer-video",
        "adapt-sign",
        "adapt-signs-folder",
        "image-points",
        "train-production",
    ]

    def check_root_help() -> str:
        rc, out = _run_subprocess([python, "-u", str(main_py), "-h"], cwd=project_root)
        if rc != 0:
            raise RuntimeError("main.py -h failed")
        if "SignifyAI command runner" not in out:
            raise RuntimeError("Unexpected help output")
        return "root help OK"

    checks.append(_run_check("cli_help_root", check_root_help))

    for cmd in commands:
        def make_cmd_check(c: str) -> Callable[[], str]:
            def _f() -> str:
                rc, out = _run_subprocess([python, "-u", str(main_py), c, "-h"], cwd=project_root)
                if rc != 0:
                    raise RuntimeError(f"{c} -h failed")
                if "usage:" not in out.lower():
                    raise RuntimeError(f"{c} help missing usage")
                return f"{c} -h OK"
            return _f

        checks.append(_run_check(f"cli_help_{cmd}", make_cmd_check(cmd)))
    return checks


def _find_any_hand_image(project_root: Path) -> Path | None:
    images_root = project_root / "data" / "raw" / "images"
    if not images_root.exists():
        return None
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return p
    return None


def _make_smoke_video(image_path: Path, out_video: Path) -> Path:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    h, w = frame.shape[:2]
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to create smoke video writer")
    try:
        for i in range(40):
            shift = int(6 * np.sin(i / 4.0))
            moved = np.roll(frame, shift=shift, axis=1)
            writer.write(moved)
    finally:
        writer.release()
    return out_video


def run_validate_all(cfg: QAConfig) -> Path:
    project_root = PATHS.project_root
    checks: list[QACheck] = []

    if cfg.include_cli_help_checks:
        checks.extend(_command_help_checks(project_root))

    checks.append(
        _run_check(
            "doctor_skip_camera",
            lambda: "doctor OK"
            if all(r.ok for r in run_doctor(camera_index=0, check_camera=False))
            else "doctor had failures",
        )
    )

    for mode in ("rules", "ml", "temporal", "hybrid"):
        checks.append(
            _run_check(
                f"preflight_{mode}_skip_camera",
                lambda m=mode: "preflight OK"
                if run_preflight(mode=m, camera_index=0, skip_camera=True) == 0
                else "preflight failed",
            )
        )

    def training_smoke() -> str:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            csv_path = tmp_root / "dataset.csv"
            model_path = tmp_root / "model.joblib"
            labels_path = tmp_root / "labels.json"
            metadata_path = tmp_root / "meta.json"

            recs = []
            for _ in range(90):
                recs.append((np.full((FEATURE_SIZE,), 0.18, dtype=np.float32), "a"))
            for _ in range(90):
                recs.append((np.full((FEATURE_SIZE,), 0.82, dtype=np.float32), "b"))
            save_records(recs, csv_path)

            run_training(
                TrainConfig(
                    dataset_csv=csv_path,
                    model_path=model_path,
                    labels_path=labels_path,
                    metadata_path=metadata_path,
                    automl=True,
                )
            )
            if not model_path.exists():
                raise RuntimeError("model not produced")
            return "train automl smoke OK"

    checks.append(_run_check("train_automl_smoke", training_smoke))

    if _is_tensorflow_available():
        def deep_training_smoke() -> str:
            from .deep_model import DeepTrainConfig, run_deep_training

            with tempfile.TemporaryDirectory() as tmp:
                tmp_root = Path(tmp)
                csv_path = tmp_root / "dataset.csv"
                recs = []
                for _ in range(100):
                    recs.append((np.full((FEATURE_SIZE,), 0.20, dtype=np.float32), "x"))
                for _ in range(100):
                    recs.append((np.full((FEATURE_SIZE,), 0.80, dtype=np.float32), "y"))
                save_records(recs, csv_path)

                result = run_deep_training(
                    DeepTrainConfig(
                        dataset_csv=csv_path,
                        model_path=tmp_root / "deep_model.keras",
                        labels_path=tmp_root / "deep_labels.json",
                        metadata_path=tmp_root / "deep_meta.json",
                        preprocess_path=tmp_root / "deep_preprocess.joblib",
                        epochs=30,
                        batch_size=32,
                        patience=6,
                        min_samples_per_label=5,
                        seed=42,
                    )
                )
                if result.accuracy < 0.90:
                    raise RuntimeError(f"unexpected low deep accuracy: {result.accuracy:.3f}")
                return f"deep train smoke OK (acc={result.accuracy:.3f})"

        checks.append(_run_check("train_deep_smoke", deep_training_smoke))
    else:
        checks.append(QACheck(name="train_deep_smoke", status="SKIP", seconds=0.0, detail="tensorflow not installed"))

    def temporal_smoke() -> str:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            csv_path = tmp_root / "dataset.csv"
            npz_path = tmp_root / "seq.npz"
            model_path = tmp_root / "temp_model.joblib"
            labels_path = tmp_root / "temp_labels.json"
            metadata_path = tmp_root / "temp_meta.json"

            recs = []
            for _ in range(120):
                recs.append((np.full((FEATURE_SIZE,), 0.15, dtype=np.float32), "a"))
            for _ in range(120):
                recs.append((np.full((FEATURE_SIZE,), 0.85, dtype=np.float32), "b"))
            save_records(recs, csv_path)
            build_sequence_dataset_from_frames(csv_path, npz_path, seq_len=12, stride=3)
            run_temporal_training(
                TemporalTrainConfig(
                    dataset_npz=npz_path,
                    model_path=model_path,
                    labels_path=labels_path,
                    metadata_path=metadata_path,
                )
            )
            if not model_path.exists():
                raise RuntimeError("temporal model not produced")
            return "temporal smoke OK"

    checks.append(_run_check("temporal_train_smoke", temporal_smoke))

    def production_train_smoke() -> str:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            csv_path = tmp_root / "dataset.csv"
            recs = []
            for _ in range(120):
                recs.append((np.full((FEATURE_SIZE,), 0.22, dtype=np.float32), "hello"))
            for _ in range(120):
                recs.append((np.full((FEATURE_SIZE,), 0.78, dtype=np.float32), "thanks"))
            save_records(recs, csv_path)
            summary = run_production_training(
                ProductionTrainConfig(
                    frame_dataset_csv=csv_path,
                    frame_model_path=tmp_root / "gesture_model.joblib",
                    frame_labels_path=tmp_root / "labels.json",
                    frame_metadata_path=tmp_root / "model_metadata.json",
                    sequence_dataset_npz=tmp_root / "sequence_dataset.npz",
                    sequence_len=12,
                    sequence_stride=3,
                    temporal_model_path=tmp_root / "temporal_gesture_model.joblib",
                    temporal_labels_path=tmp_root / "temporal_labels.json",
                    temporal_metadata_path=tmp_root / "temporal_model_metadata.json",
                    summary_path=tmp_root / "summary.json",
                )
            )
            if not summary.exists():
                raise RuntimeError("production summary missing")
            return "train-production smoke OK"

    checks.append(_run_check("train_production_smoke", production_train_smoke))

    def report_smoke() -> str:
        append_event(PATHS.data_processed / "session_log.csv", label="HELLO", confidence=0.91, hand_count=1)
        out = build_session_report(
            ReportConfig(
                log_path=PATHS.data_processed / "session_log.csv",
                out_path=PATHS.data_processed / "session_report.md",
            )
        )
        if not out.exists():
            raise RuntimeError("report not created")
        return f"report OK: {out}"

    checks.append(_run_check("report_smoke", report_smoke))

    if cfg.include_release_bundle_check:
        checks.append(
            _run_check(
                "release_bundle_smoke",
                lambda: f"bundle OK: {build_release_bundle(ReleaseBundleConfig(out_dir=PATHS.project_root / 'dist'))}",
            )
        )

    sample_image = _find_any_hand_image(project_root)
    if sample_image is None:
        checks.append(QACheck(name="image_points_smoke", status="SKIP", seconds=0.0, detail="No sample image found"))
        checks.append(QACheck(name="prototype_adapt_smoke", status="SKIP", seconds=0.0, detail="No sample image found"))
        checks.append(QACheck(name="video_infer_smoke", status="SKIP", seconds=0.0, detail="No sample image found"))
    else:
        checks.append(
            _run_check(
                "image_points_smoke",
                lambda: (
                    lambda info: f"image-points OK: hands={info.hand_count}, variant={info.best_variant}"
                )(
                    extract_points_from_image(
                        sample_image,
                        save_overlay_to=PATHS.data_processed / "qa_points_overlay.png",
                    )[1]
                ),
            )
        )

        checks.append(
            _run_check(
                "prototype_adapt_smoke",
                lambda: (
                    lambda stats: f"adapt-sign OK: saved={stats.saved_vectors}"
                )(
                    adapt_sign_from_images(
                        label="qa_sign",
                        image_paths=[sample_image],
                        out_db=PATHS.models / "prototype_signs.npz",
                        phrase_text="Quality check sign",
                    )
                ),
            )
        )

        def video_infer_smoke() -> str:
            smoke_video = _make_smoke_video(sample_image, PATHS.data_processed / "qa_smoke_video.mp4")
            out = run_video_inference(
                VideoInferConfig(
                    input_video=smoke_video,
                    out_json=PATHS.data_processed / "qa_video_infer.json",
                    mode="hybrid",
                )
            )
            if not out.exists():
                raise RuntimeError("video infer output missing")
            return f"video infer OK: {out}"

        checks.append(_run_check("video_infer_smoke", video_infer_smoke))

    if cfg.include_pytest:
        def pytest_check() -> str:
            rc, out = _run_subprocess(
                [sys.executable, "-m", "pytest", "tests", "-q"],
                cwd=project_root,
                timeout_sec=1800,
            )
            if rc != 0:
                tail = "\n".join(out.splitlines()[-20:])
                raise RuntimeError(f"pytest failed\n{tail}")
            tail = "\n".join(out.splitlines()[-5:])
            return tail or "pytest passed"

        checks.append(_run_check("pytest_suite", pytest_check))

    passed = sum(1 for c in checks if c.status == "PASS")
    failed = sum(1 for c in checks if c.status == "FAIL")
    skipped = sum(1 for c in checks if c.status == "SKIP")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "project_root": str(project_root),
        "summary": {
            "total": len(checks),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": round((passed / max(1, (passed + failed))) * 100.0, 2),
        },
        "checks": [asdict(c) for c in checks],
    }

    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cfg.out_json
