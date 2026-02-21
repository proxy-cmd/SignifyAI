from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import warnings

# Reduce noisy TensorFlow/MediaPipe logs for cleaner console output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
)

from signifyai.collect import CollectConfig, run_collection
from signifyai.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RAW_IMAGES_DIR,
    DEFAULT_REPORT_PATH,
    DEFAULT_SESSION_LOG_PATH,
)
from signifyai.external_data import import_dataset_from_url, import_from_kaggle, import_zip_dataset
from signifyai.image_dataset import BuildImageDatasetConfig, build_dataset_from_images
from signifyai.realtime import RealtimeConfig, run_realtime
from signifyai.report import ReportConfig, build_session_report
from signifyai.train import TrainConfig, run_training


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SignifyAI command runner")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_collect = sub.add_parser("collect", help="Collect landmark samples for one label")
    p_collect.add_argument("--label", required=True, help="Label name (example: hello)")
    p_collect.add_argument("--samples", type=int, default=300)
    p_collect.add_argument("--camera", type=int, default=0)
    p_collect.add_argument("--width", type=int, default=960)
    p_collect.add_argument("--height", type=int, default=720)
    p_collect.add_argument("--out", type=Path, default=DEFAULT_DATASET_PATH)

    p_train = sub.add_parser("train", help="Train classifier on collected dataset")
    p_train.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p_train.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_train.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_train.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    p_train.add_argument("--no-calibration", action="store_true", help="Disable probability calibration")

    p_kaggle = sub.add_parser("import-kaggle", help="Import image dataset from Kaggle")
    p_kaggle.add_argument("--slug", required=True, help="Kaggle slug: owner/dataset-name")
    p_kaggle.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_IMAGES_DIR)
    p_kaggle.add_argument("--force", action="store_true", help="Force redownload")

    p_url = sub.add_parser("import-url", help="Import ZIP dataset from direct URL")
    p_url.add_argument("--url", required=True)
    p_url.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_IMAGES_DIR)

    p_zip = sub.add_parser("import-zip", help="Import local ZIP dataset")
    p_zip.add_argument("--zip-file", type=Path, required=True)
    p_zip.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_IMAGES_DIR)

    p_build = sub.add_parser("build-image-dataset", help="Build landmark CSV from image folders")
    p_build.add_argument("--images-root", type=Path, default=DEFAULT_RAW_IMAGES_DIR)
    p_build.add_argument("--out-csv", type=Path, default=DEFAULT_DATASET_PATH)
    p_build.add_argument("--max-per-class", type=int, default=0)
    p_build.add_argument("--min-det-conf", type=float, default=0.55)

    p_run = sub.add_parser("run", help="Run realtime gesture recognition")
    p_run.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_run.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_run.add_argument("--camera", type=int, default=0)
    p_run.add_argument("--width", type=int, default=960)
    p_run.add_argument("--height", type=int, default=720)
    p_run.add_argument("--threshold", type=float, default=0.60)
    p_run.add_argument("--smooth", type=int, default=7)
    p_run.add_argument("--session-log", type=Path, default=DEFAULT_SESSION_LOG_PATH)
    p_run.add_argument("--mode", choices=["rules", "ml", "hybrid"], default="hybrid")
    p_run.add_argument("--rule-threshold", type=float, default=0.78)
    p_run.add_argument("--infer-interval", type=int, default=1, help="Run heavy inference every N frames")
    p_run.add_argument("--infer-scale", type=float, default=0.75, help="Inference resize scale (0.4-1.0)")
    p_run.add_argument("--stage", action="store_true", help="Start in clean stage presentation mode")
    p_run.add_argument("--dev-ui", action="store_true", help="Start in detailed developer HUD mode")
    p_run.add_argument("--demo-script", action="store_true", help="Show guided sign prompts for stage demo")

    p_report = sub.add_parser("report", help="Generate markdown report from session log")
    p_report.add_argument("--session-log", type=Path, default=DEFAULT_SESSION_LOG_PATH)
    p_report.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)

    return parser


def main() -> None:
    parser = make_parser()
    # Convenience mode: if user runs `python src/main.py` with no args,
    # start realtime mode with defaults.
    if len(sys.argv) == 1:
        args = parser.parse_args(["run"])
    else:
        args = parser.parse_args()

    if args.cmd == "collect":
        cfg = CollectConfig(
            label=args.label,
            samples=args.samples,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            out_csv=args.out,
        )
        run_collection(cfg)
        return

    if args.cmd == "train":
        cfg = TrainConfig(
            dataset_csv=args.dataset,
            model_path=args.model,
            labels_path=args.labels,
            metadata_path=args.metadata,
            calibrate_probs=not args.no_calibration,
        )
        run_training(cfg)
        return

    if args.cmd == "import-kaggle":
        target = import_from_kaggle(args.slug, args.out_dir, force=args.force)
        print(f"Kaggle dataset imported into: {target}")
        return

    if args.cmd == "import-url":
        count = import_dataset_from_url(args.url, args.out_dir)
        print(f"Imported ZIP entries: {count}")
        print(f"Dataset directory: {args.out_dir}")
        return

    if args.cmd == "import-zip":
        count = import_zip_dataset(args.zip_file, args.out_dir)
        print(f"Imported ZIP entries: {count}")
        print(f"Dataset directory: {args.out_dir}")
        return

    if args.cmd == "build-image-dataset":
        cfg = BuildImageDatasetConfig(
            root_dir=args.images_root,
            out_csv=args.out_csv,
            max_images_per_class=args.max_per_class,
            min_detection_confidence=args.min_det_conf,
        )
        total, saved = build_dataset_from_images(cfg)
        print(f"Processed images: {total}")
        print(f"Saved samples: {saved}")
        print(f"Output CSV: {args.out_csv}")
        return

    if args.cmd == "run":
        cfg = RealtimeConfig(
            model_path=args.model,
            labels_path=args.labels,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            confidence_threshold=args.threshold,
            smoothing_window=args.smooth,
            session_log_path=args.session_log,
            mode=args.mode,
            rule_confidence_threshold=args.rule_threshold,
            inference_interval=args.infer_interval,
            inference_scale=args.infer_scale,
            stage_mode=(True if args.stage else (False if args.dev_ui else True)),
            demo_script=args.demo_script,
        )
        run_realtime(cfg)
        return

    if args.cmd == "report":
        out = build_session_report(ReportConfig(log_path=args.session_log, out_path=args.out))
        print(f"Session report generated: {out}")
        return

    raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
