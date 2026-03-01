from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import platform
import sys
import traceback
import warnings

# Reduce noisy TensorFlow/MediaPipe logs for cleaner console output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
)

from signifyai.collect import CollectConfig, run_collection
from signifyai.collect_sequence import CollectSequenceConfig, run_sequence_collection
from signifyai.bootstrap import BootstrapConfig, run_bootstrap
from signifyai.benchmark import run_benchmark
from signifyai.config import (
    DEFAULT_CONFUSION_CSV_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RAW_IMAGES_DIR,
    DEFAULT_REPORT_PATH,
    DEFAULT_SEQUENCE_DATASET_PATH,
    DEFAULT_SESSION_LOG_PATH,
    DEFAULT_TEMPORAL_LABELS_PATH,
    DEFAULT_TEMPORAL_METADATA_PATH,
    DEFAULT_TEMPORAL_MODEL_PATH,
)
from signifyai.doctor import print_results, run_doctor
from signifyai.external_data import import_dataset_from_url, import_from_kaggle, import_zip_dataset
from signifyai.image_dataset import BuildImageDatasetConfig, build_dataset_from_images
from signifyai.realtime import RealtimeConfig, run_realtime
from signifyai.preflight import run_preflight
from signifyai.report import ReportConfig, build_session_report
from signifyai.production_train import ProductionTrainConfig, run_production_training
from signifyai.release import ReleaseBundleConfig, build_release_bundle
from signifyai.sequence_dataset import build_sequence_dataset_from_frames
from signifyai.temporal_model import TemporalTrainConfig, run_temporal_training
from signifyai.train import TrainConfig, run_training
from signifyai.video_infer import VideoInferConfig, run_video_inference
from signifyai.phrase_map import load_phrase_map, set_phrase


def apply_run_profile(args: argparse.Namespace) -> None:
    profile = str(args.profile).lower()
    if profile == "speed":
        args.width = 960
        args.height = 540
        args.camera_fps = max(int(args.camera_fps), 30)
        args.infer_scale = 0.60
        args.smooth = 5
        args.threshold = 0.58
        args.rule_threshold = 0.74
        args.model_complexity = 0
        args.landmark_smoothing = 0.72
        args.target_fps = max(float(args.target_fps), 22.0)
    elif profile == "accuracy":
        args.width = max(int(args.width), 1280)
        args.height = max(int(args.height), 720)
        args.camera_fps = max(int(args.camera_fps), 30)
        args.infer_scale = 0.90
        args.smooth = 9
        args.threshold = 0.68
        args.rule_threshold = 0.82
        args.model_complexity = 1
        args.landmark_smoothing = 0.85
        args.target_fps = min(float(args.target_fps), 18.0)
    elif profile == "stage":
        args.mode = "rules"
        args.stage = True
        args.dev_ui = False
        args.demo_script = True
        args.width = 1280
        args.height = 720
        args.camera_fps = max(int(args.camera_fps), 30)
        args.infer_scale = 0.72
        args.smooth = 7
        args.threshold = 0.62
        args.rule_threshold = 0.78
        args.model_complexity = 0
        args.landmark_smoothing = 0.80
    elif profile == "production":
        args.mode = "hybrid"
        args.stage = False
        args.dev_ui = False
        args.demo_script = False
        args.width = 1280
        args.height = 720
        args.camera_fps = max(int(args.camera_fps), 60)
        args.infer_scale = 0.80
        args.smooth = 9
        args.threshold = 0.66
        args.rule_threshold = 0.82
        args.model_complexity = 0
        args.landmark_smoothing = 0.82
        args.target_fps = max(float(args.target_fps), 20.0)
    elif profile == "smoothhd":
        args.mode = "hybrid"
        args.stage = False
        args.dev_ui = False
        args.demo_script = False
        args.width = 1280
        args.height = 720
        args.camera_fps = max(int(args.camera_fps), 60)
        args.infer_scale = 0.72
        args.smooth = 7
        args.threshold = 0.62
        args.rule_threshold = 0.80
        args.model_complexity = 0
        args.landmark_smoothing = 0.86
        args.target_fps = max(float(args.target_fps), 24.0)
    # balanced: keep CLI/default values as-is.


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

    p_collect_seq = sub.add_parser("collect-seq", help="Collect temporal clips for one label")
    p_collect_seq.add_argument("--label", required=True, help="Label name (example: hello)")
    p_collect_seq.add_argument("--clips", type=int, default=120)
    p_collect_seq.add_argument("--seq-len", type=int, default=24)
    p_collect_seq.add_argument("--min-visible", type=int, default=14)
    p_collect_seq.add_argument("--auto", dest="auto", action="store_true", help="Auto-record clips continuously")
    p_collect_seq.add_argument("--no-auto", dest="auto", action="store_false", help="Manual recording mode")
    p_collect_seq.set_defaults(auto=True)
    p_collect_seq.add_argument("--clip-gap", type=float, default=1.2, help="Seconds between auto-recorded clips")
    p_collect_seq.add_argument("--camera", type=int, default=0)
    p_collect_seq.add_argument("--width", type=int, default=960)
    p_collect_seq.add_argument("--height", type=int, default=720)
    p_collect_seq.add_argument("--out", type=Path, default=DEFAULT_SEQUENCE_DATASET_PATH)

    p_train = sub.add_parser("train", help="Train classifier on collected dataset")
    p_train.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p_train.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_train.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_train.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    p_train.add_argument("--no-calibration", action="store_true", help="Disable probability calibration")
    p_train.add_argument("--automl", action="store_true", help="Run AutoML model selection and save confusion matrix")
    p_train.add_argument("--confusion-csv", type=Path, default=DEFAULT_CONFUSION_CSV_PATH)

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

    p_build_seq = sub.add_parser("build-seq-dataset", help="Build sequence dataset from frame-level CSV")
    p_build_seq.add_argument("--frame-csv", type=Path, default=DEFAULT_DATASET_PATH)
    p_build_seq.add_argument("--out-npz", type=Path, default=DEFAULT_SEQUENCE_DATASET_PATH)
    p_build_seq.add_argument("--seq-len", type=int, default=24)
    p_build_seq.add_argument("--stride", type=int, default=4)
    p_build_seq.add_argument("--per-label-limit", type=int, default=0)

    p_train_seq = sub.add_parser("train-seq", help="Train temporal sequence model")
    p_train_seq.add_argument("--dataset", type=Path, default=DEFAULT_SEQUENCE_DATASET_PATH)
    p_train_seq.add_argument("--model", type=Path, default=DEFAULT_TEMPORAL_MODEL_PATH)
    p_train_seq.add_argument("--labels", type=Path, default=DEFAULT_TEMPORAL_LABELS_PATH)
    p_train_seq.add_argument("--metadata", type=Path, default=DEFAULT_TEMPORAL_METADATA_PATH)

    p_phrase = sub.add_parser("set-phrase", help="Set spoken sentence for a gesture label")
    p_phrase.add_argument("--label", required=True, help="Gesture label (example: watching_you)")
    p_phrase.add_argument("--text", required=True, help="Sentence to speak (example: I'm watching you)")

    p_phrase_list = sub.add_parser("list-phrases", help="List custom gesture phrase mappings")

    p_record_combo = sub.add_parser("record-combo", help="Easy custom sequence recorder + phrase mapping")
    p_record_combo.add_argument("--label", required=True, help="Sequence label (example: watching_you)")
    p_record_combo.add_argument("--text", required=True, help="Sentence to speak for this label")
    p_record_combo.add_argument("--clips", type=int, default=80)
    p_record_combo.add_argument("--seq-len", type=int, default=24)
    p_record_combo.add_argument("--min-visible", type=int, default=14)
    p_record_combo.add_argument("--clip-gap", type=float, default=1.2)
    p_record_combo.add_argument("--camera", type=int, default=0)
    p_record_combo.add_argument("--width", type=int, default=960)
    p_record_combo.add_argument("--height", type=int, default=720)
    p_record_combo.add_argument("--out", type=Path, default=DEFAULT_SEQUENCE_DATASET_PATH)

    p_run = sub.add_parser("run", help="Run realtime gesture recognition")
    p_run.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_run.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_run.add_argument("--profile", choices=["balanced", "speed", "accuracy", "stage", "production", "smoothhd"], default="balanced")
    p_run.add_argument("--camera", type=int, default=0)
    p_run.add_argument("--width", type=int, default=1280)
    p_run.add_argument("--height", type=int, default=720)
    p_run.add_argument("--camera-fps", type=int, default=60)
    p_run.add_argument("--threshold", type=float, default=0.60)
    p_run.add_argument("--smooth", type=int, default=7)
    p_run.add_argument("--session-log", type=Path, default=DEFAULT_SESSION_LOG_PATH)
    p_run.add_argument("--mode", choices=["rules", "ml", "temporal", "hybrid"], default="hybrid")
    p_run.add_argument("--rule-threshold", type=float, default=0.78)
    p_run.add_argument("--infer-interval", type=int, default=1, help="Run heavy inference every N frames")
    p_run.add_argument("--infer-scale", type=float, default=0.75, help="Inference resize scale (0.4-1.0)")
    p_run.add_argument("--model-complexity", type=int, choices=[0, 1], default=0, help="MediaPipe hand model complexity")
    p_run.add_argument("--landmark-smoothing", type=float, default=0.78, help="Landmark smoothing factor (0.0-0.95)")
    p_run.add_argument("--enhance-frame", dest="enhance_frame", action="store_true", help="Enable lightweight video enhancement")
    p_run.add_argument("--no-enhance-frame", dest="enhance_frame", action="store_false", help="Disable video enhancement for max speed")
    p_run.set_defaults(enhance_frame=True)
    p_run.add_argument("--auto-speak", dest="auto_speak", action="store_true", help="Auto-speak stable labels")
    p_run.add_argument("--no-auto-speak", dest="auto_speak", action="store_false", help="Disable auto-speaking; use sentence controls manually")
    p_run.set_defaults(auto_speak=True)
    p_run.add_argument("--adaptive-perf", dest="adaptive_perf", action="store_true", help="Auto-adjust inference interval for stable FPS")
    p_run.add_argument("--no-adaptive-perf", dest="adaptive_perf", action="store_false", help="Disable adaptive inference interval tuning")
    p_run.set_defaults(adaptive_perf=True)
    p_run.add_argument("--target-fps", type=float, default=20.0, help="Target FPS for adaptive performance")
    p_run.add_argument("--stage", action="store_true", help="Start in clean stage presentation mode")
    p_run.add_argument("--dev-ui", action="store_true", help="Start in detailed developer HUD mode")
    p_run.add_argument("--demo-script", action="store_true", help="Show guided sign prompts for stage demo")
    p_run.add_argument("--temporal-model", type=Path, default=DEFAULT_TEMPORAL_MODEL_PATH)
    p_run.add_argument("--temporal-labels", type=Path, default=DEFAULT_TEMPORAL_LABELS_PATH)
    p_run.add_argument("--temporal-metadata", type=Path, default=DEFAULT_TEMPORAL_METADATA_PATH)
    p_run.add_argument("--temporal-threshold", type=float, default=0.60)

    p_report = sub.add_parser("report", help="Generate markdown report from session log")
    p_report.add_argument("--session-log", type=Path, default=DEFAULT_SESSION_LOG_PATH)
    p_report.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)

    p_pre = sub.add_parser("preflight", help="Run production preflight checks (doctor + model files)")
    p_pre.add_argument("--mode", choices=["rules", "ml", "temporal", "hybrid"], default="hybrid")
    p_pre.add_argument("--camera", type=int, default=0)
    p_pre.add_argument("--skip-camera", action="store_true")

    p_doctor = sub.add_parser("doctor", help="Check environment/import/camera health")
    p_doctor.add_argument("--camera", type=int, default=0)
    p_doctor.add_argument("--skip-camera", action="store_true")

    p_boot = sub.add_parser("bootstrap-ml", help="End-to-end Kaggle import + image dataset build + AutoML train")
    p_boot.add_argument("--slug", default="grassknoted/asl-alphabet")
    p_boot.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_IMAGES_DIR)
    p_boot.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p_boot.add_argument("--max-per-class", type=int, default=1200)
    p_boot.add_argument("--min-free-gb", type=float, default=20.0, help="Safety check to avoid filling disk during dataset import")
    p_boot.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_boot.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_boot.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)

    p_bench = sub.add_parser("benchmark", help="Measure camera and tracker FPS")
    p_bench.add_argument("--camera", type=int, default=0)
    p_bench.add_argument("--width", type=int, default=960)
    p_bench.add_argument("--height", type=int, default=720)
    p_bench.add_argument("--seconds", type=float, default=6.0)

    p_release = sub.add_parser("release-bundle", help="Package model/reports/logs into a deployable zip")
    p_release.add_argument("--out-dir", type=Path, default=Path("dist"))
    p_release.add_argument("--include-videos", action="store_true")

    p_video = sub.add_parser("infer-video", help="Run offline inference on a recorded video")
    p_video.add_argument("--input", type=Path, required=True, help="Input video file path")
    p_video.add_argument("--out", type=Path, default=Path("data/processed/video_infer.json"))
    p_video.add_argument("--mode", choices=["rules", "ml", "temporal", "hybrid"], default="hybrid")
    p_video.add_argument("--threshold", type=float, default=0.60)
    p_video.add_argument("--rule-threshold", type=float, default=0.78)
    p_video.add_argument("--temporal-threshold", type=float, default=0.60)
    p_video.add_argument("--smooth", type=int, default=7)
    p_video.add_argument("--infer-interval", type=int, default=1)
    p_video.add_argument("--infer-scale", type=float, default=0.75)

    p_prod = sub.add_parser("train-production", help="Train frame AutoML + temporal model in one command")
    p_prod.add_argument("--frame-dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p_prod.add_argument("--seq-npz", type=Path, default=DEFAULT_SEQUENCE_DATASET_PATH)
    p_prod.add_argument("--seq-len", type=int, default=24)
    p_prod.add_argument("--seq-stride", type=int, default=4)

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

    if args.cmd == "collect-seq":
        cfg = CollectSequenceConfig(
            label=args.label,
            clips=args.clips,
            seq_len=args.seq_len,
            min_visible_frames=args.min_visible,
            auto_mode=args.auto,
            clip_gap_sec=args.clip_gap,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            out_npz=args.out,
        )
        run_sequence_collection(cfg)
        return

    if args.cmd == "record-combo":
        set_phrase(args.label, args.text)
        print(f"Saved phrase mapping: {args.label} -> {args.text}")
        cfg = CollectSequenceConfig(
            label=args.label,
            clips=args.clips,
            seq_len=args.seq_len,
            min_visible_frames=args.min_visible,
            auto_mode=True,
            clip_gap_sec=args.clip_gap,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            out_npz=args.out,
        )
        run_sequence_collection(cfg)
        print("Next step: train temporal model with:")
        print("python -u .\\src\\main.py train-seq")
        return

    if args.cmd == "train":
        cfg = TrainConfig(
            dataset_csv=args.dataset,
            model_path=args.model,
            labels_path=args.labels,
            metadata_path=args.metadata,
            calibrate_probs=not args.no_calibration,
            automl=args.automl,
            confusion_csv_path=args.confusion_csv,
        )
        try:
            run_training(cfg)
        except ValueError as ex:
            msg = str(ex)
            print(f"[ERROR] {msg}")
            if "at least 2 labels" in msg.lower():
                print("You need samples from at least 2 different labels.")
                print("Example:")
                print("python -u .\\src\\main.py collect --label hello --samples 200")
                print("python -u .\\src\\main.py collect --label thanks --samples 200")
                print("Then train again.")
            raise SystemExit(1)
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

    if args.cmd == "build-seq-dataset":
        total, saved = build_sequence_dataset_from_frames(
            frame_csv=args.frame_csv,
            out_npz=args.out_npz,
            seq_len=args.seq_len,
            stride=args.stride,
            per_label_limit=args.per_label_limit,
        )
        print(f"Candidate windows: {total}")
        print(f"Saved sequence samples: {saved}")
        print(f"Output NPZ: {args.out_npz}")
        return

    if args.cmd == "train-seq":
        try:
            acc = run_temporal_training(
                TemporalTrainConfig(
                    dataset_npz=args.dataset,
                    model_path=args.model,
                    labels_path=args.labels,
                    metadata_path=args.metadata,
                )
            )
        except ValueError as ex:
            msg = str(ex)
            print(f"[ERROR] {msg}")
            if "at least 2 labels" in msg.lower():
                print("You need sequence clips from at least 2 different labels.")
                print("Use:")
                print("python -u .\\src\\main.py collect-seq --label hello --clips 80")
                print("python -u .\\src\\main.py collect-seq --label thanks --clips 80")
            raise SystemExit(1)
        print(f"Temporal accuracy: {acc:.4f}")
        return

    if args.cmd == "set-phrase":
        set_phrase(args.label, args.text)
        print(f"Saved phrase mapping: {args.label} -> {args.text}")
        return

    if args.cmd == "list-phrases":
        mapping = load_phrase_map()
        if not mapping:
            print("No custom phrases yet.")
            return
        print("Custom gesture phrases:")
        for k, v in sorted(mapping.items()):
            print(f"- {k}: {v}")
        return

    if args.cmd == "run":
        apply_run_profile(args)
        cfg = RealtimeConfig(
            model_path=args.model,
            labels_path=args.labels,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            camera_fps=args.camera_fps,
            confidence_threshold=args.threshold,
            smoothing_window=args.smooth,
            session_log_path=args.session_log,
            mode=args.mode,
            rule_confidence_threshold=args.rule_threshold,
            inference_interval=args.infer_interval,
            inference_scale=args.infer_scale,
            model_complexity=args.model_complexity,
            landmark_smoothing=args.landmark_smoothing,
            auto_speak=args.auto_speak,
            adaptive_performance=args.adaptive_perf,
            target_fps=args.target_fps,
            stage_mode=(True if args.stage else (False if args.dev_ui else True)),
            demo_script=args.demo_script,
            temporal_model_path=args.temporal_model,
            temporal_labels_path=args.temporal_labels,
            temporal_metadata_path=args.temporal_metadata,
            temporal_confidence_threshold=args.temporal_threshold,
            enhance_frame=args.enhance_frame,
        )
        run_realtime(cfg)
        return

    if args.cmd == "report":
        out = build_session_report(ReportConfig(log_path=args.session_log, out_path=args.out))
        print(f"Session report generated: {out}")
        return

    if args.cmd == "preflight":
        code = run_preflight(mode=args.mode, camera_index=args.camera, skip_camera=args.skip_camera)
        raise SystemExit(code)

    if args.cmd == "doctor":
        results = run_doctor(camera_index=args.camera, check_camera=not args.skip_camera)
        code = print_results(results)
        if code != 0:
            raise SystemExit(code)
        return

    if args.cmd == "bootstrap-ml":
        run_bootstrap(
            BootstrapConfig(
                kaggle_slug=args.slug,
                images_dir=args.out_dir,
                dataset_csv=args.dataset,
                model_path=args.model,
                labels_path=args.labels,
                metadata_path=args.metadata,
                max_per_class=args.max_per_class,
                min_free_gb=args.min_free_gb,
            )
        )
        return

    if args.cmd == "benchmark":
        result = run_benchmark(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            seconds=args.seconds,
        )
        print(f"Raw camera FPS: {result.raw_fps:.1f}")
        print(f"Tracker FPS: {result.tracker_fps:.1f}")
        print(f"Frames measured: {result.frames} over {result.seconds:.1f}s")
        return

    if args.cmd == "release-bundle":
        out = build_release_bundle(
            ReleaseBundleConfig(
                out_dir=args.out_dir,
                include_videos=args.include_videos,
            )
        )
        print(f"Release bundle created: {out}")
        return

    if args.cmd == "infer-video":
        out = run_video_inference(
            VideoInferConfig(
                input_video=args.input,
                out_json=args.out,
                mode=args.mode,
                confidence_threshold=args.threshold,
                rule_confidence_threshold=args.rule_threshold,
                temporal_confidence_threshold=args.temporal_threshold,
                smoothing_window=args.smooth,
                infer_interval=args.infer_interval,
                infer_scale=args.infer_scale,
            )
        )
        print(f"Video inference saved: {out}")
        return

    if args.cmd == "train-production":
        out = run_production_training(
            ProductionTrainConfig(
                frame_dataset_csv=args.frame_dataset,
                sequence_dataset_npz=args.seq_npz,
                sequence_len=args.seq_len,
                sequence_stride=args.seq_stride,
            )
        )
        print(f"Production training summary: {out}")
        return

    raise RuntimeError("Unknown command")


def _write_crash_log(ex: Exception) -> Path:
    crash_dir = Path("data/processed/crash_logs")
    crash_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    crash_path = crash_dir / f"crash_{ts}.log"

    details = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
        f"argv: {sys.argv}",
        "",
        "traceback:",
        traceback.format_exc(),
        f"error: {ex}",
    ]
    crash_path.write_text("\n".join(details), encoding="utf-8")
    return crash_path


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
    except Exception as ex:
        log_path = _write_crash_log(ex)
        print(f"[ERROR] {ex}")
        print(f"[ERROR] Crash log saved: {log_path}")
        raise SystemExit(1)
