from __future__ import annotations

import argparse
from datetime import datetime
import json
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
from signifyai.calibration import CalibrationConfig, run_calibration
from signifyai.dataset_check import run_dataset_check
from signifyai.data_help import get_data_help_text
from signifyai.bootstrap import BootstrapConfig, BootstrapURLConfig, run_bootstrap, run_bootstrap_from_url
from signifyai.benchmark import run_benchmark
from signifyai.safe_logging import redact_cli_args
from signifyai.final_test import FinalTestConfig, run_final_test
from signifyai.config import (
    DEFAULT_CONFUSION_CSV_PATH,
    DEFAULT_CALIBRATION_PROFILE_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_DEEP_LABELS_PATH,
    DEFAULT_DEEP_METADATA_PATH,
    DEFAULT_DEEP_MODEL_PATH,
    DEFAULT_DEEP_PREPROCESS_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROTOTYPE_DB_PATH,
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
from signifyai.video_dataset import BuildVideoDatasetConfig, build_dataset_from_videos
from signifyai.realtime import RealtimeConfig, run_realtime
from signifyai.preflight import run_preflight
from signifyai.report import ReportConfig, build_session_report
from signifyai.model_report import ModelReportConfig, build_model_report
from signifyai.qa import QAConfig, run_validate_all
from signifyai.production_train import ProductionTrainConfig, run_production_training
from signifyai.release import ReleaseBundleConfig, build_release_bundle
from signifyai.sequence_dataset import build_sequence_dataset_from_frames
from signifyai.temporal_model import TemporalTrainConfig, run_temporal_training
from signifyai.train import TrainConfig, run_training
from signifyai.teach_sign import TeachSignConfig, run_teach_sign
from signifyai.video_infer import VideoInferConfig, run_video_inference
from signifyai.phrase_map import load_phrase_map, set_phrase
from signifyai.prototype_adapt import (
    adapt_sign_from_images,
    adapt_signs_from_folder,
    extract_points_from_image,
)


def apply_run_profile(args: argparse.Namespace) -> None:
    profile = str(args.profile).lower()
    if profile == "balanced":
        args.mode = "hybrid"
        args.stage = False
        args.dev_ui = False
        args.demo_script = False
        args.width = min(int(args.width), 960)
        args.height = min(int(args.height), 540)
        args.camera_fps = max(int(args.camera_fps), 60)
        args.infer_scale = min(float(args.infer_scale), 0.64)
        args.infer_interval = max(1, int(args.infer_interval))
        args.smooth = min(int(args.smooth), 5)
        args.threshold = min(float(args.threshold), 0.60)
        args.rule_threshold = min(float(args.rule_threshold), 0.76)
        args.model_complexity = 0
        args.landmark_smoothing = min(float(args.landmark_smoothing), 0.75)
        args.target_fps = max(float(args.target_fps), 30.0)
        args.enhance_frame = False
        args.quality_gate = False
        args.use_deep_model = False
        args.ml_min_margin = min(float(args.ml_min_margin), 0.06)
        args.min_stable_frames = min(int(args.min_stable_frames), 2)
        args.label_hold_sec = min(float(args.label_hold_sec), 0.14)
        args.sentence_pause_sec = min(float(args.sentence_pause_sec), 1.0)
        args.sentence_append_cooldown = min(float(args.sentence_append_cooldown), 0.35)
    elif profile == "ultra-speed":
        args.mode = "hybrid"
        args.stage = False
        args.dev_ui = False
        args.demo_script = False
        args.width = 854
        args.height = 480
        args.camera_fps = max(int(args.camera_fps), 45)
        args.infer_scale = 0.55
        args.infer_interval = max(int(args.infer_interval), 2)
        args.smooth = 5
        args.threshold = 0.56
        args.rule_threshold = 0.74
        args.model_complexity = 0
        args.landmark_smoothing = 0.70
        args.target_fps = max(float(args.target_fps), 28.0)
        args.enhance_frame = False
        args.quality_gate = False
        args.ml_min_margin = min(float(args.ml_min_margin), 0.05)
        args.use_deep_model = False
    elif profile == "speed":
        args.mode = "hybrid"
        args.stage = False
        args.dev_ui = False
        args.demo_script = False
        args.width = 960
        args.height = 540
        args.camera_fps = max(int(args.camera_fps), 60)
        args.infer_scale = 0.58
        args.smooth = 5
        args.threshold = 0.58
        args.rule_threshold = 0.74
        args.model_complexity = 0
        args.landmark_smoothing = 0.70
        args.target_fps = max(float(args.target_fps), 36.0)
        args.enhance_frame = False
        args.quality_gate = False
        args.ml_min_margin = min(float(args.ml_min_margin), 0.06)
        args.use_deep_model = False
        args.min_stable_frames = min(int(args.min_stable_frames), 2)
        args.label_hold_sec = min(float(args.label_hold_sec), 0.12)
        args.sentence_pause_sec = min(float(args.sentence_pause_sec), 0.9)
        args.sentence_append_cooldown = min(float(args.sentence_append_cooldown), 0.30)
    elif profile == "ultra-accuracy":
        args.mode = "hybrid"
        args.stage = False
        args.dev_ui = False
        args.demo_script = False
        args.width = max(int(args.width), 1280)
        args.height = max(int(args.height), 720)
        args.camera_fps = max(int(args.camera_fps), 60)
        args.infer_scale = 0.92
        args.infer_interval = 1
        args.smooth = 13
        args.threshold = 0.72
        args.rule_threshold = 0.88
        args.model_complexity = 1
        args.landmark_smoothing = 0.92
        args.target_fps = max(float(args.target_fps), 20.0)
        args.quality_gate = True
        args.strict_consensus = True
        args.strict_override_conf = max(float(args.strict_override_conf), 0.96)
        args.label_hold_sec = max(float(args.label_hold_sec), 0.40)
        args.min_stable_frames = max(int(args.min_stable_frames), 4)
        args.ml_min_margin = max(float(args.ml_min_margin), 0.16)
        args.use_deep_model = True
        args.deep_threshold = max(float(args.deep_threshold), 0.70)
        args.deep_min_margin = max(float(args.deep_min_margin), 0.09)
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
        args.ml_min_margin = max(float(args.ml_min_margin), 0.10)
        args.use_deep_model = True
        args.deep_threshold = max(float(args.deep_threshold), 0.66)
        args.deep_min_margin = max(float(args.deep_min_margin), 0.07)
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
        args.ml_min_margin = max(float(args.ml_min_margin), 0.10)
        args.use_deep_model = True
        args.deep_threshold = max(float(args.deep_threshold), 0.65)
        args.deep_min_margin = max(float(args.deep_min_margin), 0.07)
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
        args.quality_gate = True
        args.ml_min_margin = max(float(args.ml_min_margin), 0.10)
        args.use_deep_model = True
        args.deep_threshold = max(float(args.deep_threshold), 0.64)
        args.deep_min_margin = max(float(args.deep_min_margin), 0.07)
    elif profile == "enterprise":
        args.mode = "hybrid"
        args.stage = False
        args.dev_ui = False
        args.demo_script = False
        args.width = 1280
        args.height = 720
        args.camera_fps = max(int(args.camera_fps), 60)
        args.infer_scale = 0.70
        args.smooth = 11
        args.threshold = 0.68
        args.rule_threshold = 0.84
        args.model_complexity = 1
        args.landmark_smoothing = 0.90
        args.target_fps = max(float(args.target_fps), 22.0)
        args.quality_gate = True
        args.min_brightness = max(float(args.min_brightness), 50.0)
        args.min_blur_var = max(float(args.min_blur_var), 65.0)
        args.min_hand_area = max(float(args.min_hand_area), 0.014)
        args.strict_consensus = True
        args.strict_override_conf = max(float(args.strict_override_conf), 0.95)
        args.label_hold_sec = max(float(args.label_hold_sec), 0.35)
        args.min_stable_frames = max(int(args.min_stable_frames), 4)
        args.ml_min_margin = max(float(args.ml_min_margin), 0.14)
        args.use_deep_model = True
        args.deep_threshold = max(float(args.deep_threshold), 0.70)
        args.deep_min_margin = max(float(args.deep_min_margin), 0.10)
    # balanced: keep CLI/default values as-is.


def apply_calibration_profile(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "use_calibration_profile", True)):
        return
    profile_path = Path(getattr(args, "calibration_profile", DEFAULT_CALIBRATION_PROFILE_PATH))
    if not profile_path.exists():
        return
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        rec = payload.get("recommended", {})
        if not isinstance(rec, dict):
            return
        for key in [
            "infer_interval",
            "infer_scale",
            "smooth",
            "landmark_smoothing",
            "threshold",
            "min_brightness",
            "min_blur_var",
            "min_hand_area",
            "target_fps",
        ]:
            if key in rec:
                setattr(args, key, rec[key])
        print(f"[INFO] Applied calibration profile: {profile_path}")
    except Exception as ex:
        print(f"[WARN] Failed to apply calibration profile: {ex}")


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
    p_collect.add_argument("--auto", dest="auto", action="store_true", help="Auto-capture while hand is visible")
    p_collect.add_argument("--no-auto", dest="auto", action="store_false", help="Manual capture only")
    p_collect.set_defaults(auto=True)
    p_collect.add_argument("--capture-interval", type=float, default=0.35)
    p_collect.add_argument("--min-hand-frames", type=int, default=2)
    p_collect.add_argument("--min-feature-delta", type=float, default=0.010)
    p_collect.add_argument("--flush-every", type=int, default=20)

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
    p_collect_seq.add_argument("--flush-every", type=int, default=8)

    p_calibrate = sub.add_parser("calibrate", help="Run camera/user calibration wizard and save runtime profile")
    p_calibrate.add_argument("--camera", type=int, default=0)
    p_calibrate.add_argument("--width", type=int, default=960)
    p_calibrate.add_argument("--height", type=int, default=720)
    p_calibrate.add_argument("--seconds", type=float, default=20.0)
    p_calibrate.add_argument("--out", type=Path, default=DEFAULT_CALIBRATION_PROFILE_PATH)
    p_calibrate.add_argument("--model-complexity", type=int, choices=[0, 1], default=0)
    p_calibrate.add_argument("--infer-scale", type=float, default=0.75)

    p_train = sub.add_parser("train", help="Train classifier on collected dataset")
    p_train.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p_train.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_train.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_train.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    p_train.add_argument("--no-calibration", action="store_true", help="Disable probability calibration")
    p_train.add_argument("--automl", action="store_true", help="Run AutoML model selection and save confusion matrix")
    p_train.add_argument("--confusion-csv", type=Path, default=DEFAULT_CONFUSION_CSV_PATH)
    p_train.add_argument("--min-samples-per-label", type=int, default=5, help="Drop labels with too few samples before training")

    p_train_deep = sub.add_parser("train-deep", help="Train TensorFlow deep model on landmark dataset")
    p_train_deep.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p_train_deep.add_argument("--model", type=Path, default=DEFAULT_DEEP_MODEL_PATH)
    p_train_deep.add_argument("--labels", type=Path, default=DEFAULT_DEEP_LABELS_PATH)
    p_train_deep.add_argument("--metadata", type=Path, default=DEFAULT_DEEP_METADATA_PATH)
    p_train_deep.add_argument("--preprocess", type=Path, default=DEFAULT_DEEP_PREPROCESS_PATH)
    p_train_deep.add_argument("--epochs", type=int, default=140)
    p_train_deep.add_argument("--batch-size", type=int, default=64)
    p_train_deep.add_argument("--patience", type=int, default=18)
    p_train_deep.add_argument("--min-samples-per-label", type=int, default=6)
    p_train_deep.add_argument("--seed", type=int, default=42)

    p_train_all = sub.add_parser("train-all", help="Train frame AutoML + deep TF + temporal models in one command")
    p_train_all.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p_train_all.add_argument("--frame-model", type=Path, default=DEFAULT_MODEL_PATH)
    p_train_all.add_argument("--frame-labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_train_all.add_argument("--frame-metadata", type=Path, default=DEFAULT_METADATA_PATH)
    p_train_all.add_argument("--deep-model", type=Path, default=DEFAULT_DEEP_MODEL_PATH)
    p_train_all.add_argument("--deep-labels", type=Path, default=DEFAULT_DEEP_LABELS_PATH)
    p_train_all.add_argument("--deep-metadata", type=Path, default=DEFAULT_DEEP_METADATA_PATH)
    p_train_all.add_argument("--deep-preprocess", type=Path, default=DEFAULT_DEEP_PREPROCESS_PATH)
    p_train_all.add_argument("--seq-npz", type=Path, default=DEFAULT_SEQUENCE_DATASET_PATH)
    p_train_all.add_argument("--seq-len", type=int, default=24)
    p_train_all.add_argument("--seq-stride", type=int, default=4)
    p_train_all.add_argument("--temporal-model", type=Path, default=DEFAULT_TEMPORAL_MODEL_PATH)
    p_train_all.add_argument("--temporal-labels", type=Path, default=DEFAULT_TEMPORAL_LABELS_PATH)
    p_train_all.add_argument("--temporal-metadata", type=Path, default=DEFAULT_TEMPORAL_METADATA_PATH)
    p_train_all.add_argument("--summary", type=Path, default=Path("models/train_all_summary.json"))
    p_train_all.add_argument("--frame-min-samples", type=int, default=5)
    p_train_all.add_argument("--deep-min-samples", type=int, default=6)
    p_train_all.add_argument("--deep-epochs", type=int, default=140)
    p_train_all.add_argument("--deep-batch-size", type=int, default=64)
    p_train_all.add_argument("--deep-patience", type=int, default=18)

    p_check = sub.add_parser("check-dataset", help="Validate dataset CSV before training")
    p_check.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p_check.add_argument("--min-samples-per-label", type=int, default=5)

    sub.add_parser("data-help", help="Print simple data import/collection instructions")

    p_kaggle = sub.add_parser("import-kaggle", help="Import image dataset from Kaggle")
    p_kaggle.add_argument("--slug", required=True, help="Kaggle slug: owner/dataset-name")
    p_kaggle.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_IMAGES_DIR)
    p_kaggle.add_argument("--force", action="store_true", help="Force redownload")

    p_url = sub.add_parser("import-url", help="Import ZIP dataset from direct URL")
    p_url.add_argument("--url", required=True)
    p_url.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_IMAGES_DIR)
    p_url.add_argument("--allow-private-url", dest="allow_private_url", action="store_true", help="Allow localhost/private IP URLs")
    p_url.add_argument("--no-allow-private-url", dest="allow_private_url", action="store_false", help="Block localhost/private IP URLs")
    p_url.set_defaults(allow_private_url=True)
    p_url.add_argument("--allow-file-url", dest="allow_file_url", action="store_true", help="Allow file:// local dataset URLs")
    p_url.add_argument("--no-allow-file-url", dest="allow_file_url", action="store_false", help="Block file:// local dataset URLs")
    p_url.set_defaults(allow_file_url=True)

    p_zip = sub.add_parser("import-zip", help="Import local ZIP dataset")
    p_zip.add_argument("--zip-file", type=Path, required=True)
    p_zip.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_IMAGES_DIR)

    p_build = sub.add_parser("build-image-dataset", help="Build landmark CSV from image folders")
    p_build.add_argument("--images-root", type=Path, default=DEFAULT_RAW_IMAGES_DIR)
    p_build.add_argument("--out-csv", type=Path, default=DEFAULT_DATASET_PATH)
    p_build.add_argument("--max-per-class", type=int, default=0)
    p_build.add_argument("--min-det-conf", type=float, default=0.55)

    p_build_video = sub.add_parser("build-video-dataset", help="Build landmark CSV from class-wise videos")
    p_build_video.add_argument("--videos-root", type=Path, default=Path("data/raw/videos"))
    p_build_video.add_argument("--out-csv", type=Path, default=DEFAULT_DATASET_PATH)
    p_build_video.add_argument("--max-videos-per-class", type=int, default=0)
    p_build_video.add_argument("--max-frames-per-video", type=int, default=0)
    p_build_video.add_argument("--frame-stride", type=int, default=3)
    p_build_video.add_argument("--min-det-conf", type=float, default=0.55)

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

    p_teach = sub.add_parser("teach-sign", help="Teach one new sign: collect samples and retrain automatically")
    p_teach.add_argument("--label", required=True, help="Sign label to teach")
    p_teach.add_argument("--phrase", default="", help="Optional spoken phrase mapping")
    p_teach.add_argument("--samples", type=int, default=180)
    p_teach.add_argument("--camera", type=int, default=0)
    p_teach.add_argument("--width", type=int, default=960)
    p_teach.add_argument("--height", type=int, default=720)
    p_teach.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p_teach.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_teach.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_teach.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    p_teach.add_argument("--min-samples-per-label", type=int, default=5)
    p_teach.add_argument("--deep", dest="run_deep", action="store_true", help="Also train deep model (requires requirements-deep.txt)")
    p_teach.add_argument("--no-deep", dest="run_deep", action="store_false")
    p_teach.set_defaults(run_deep=False)
    p_teach.add_argument("--temporal", dest="run_temporal", action="store_true", help="Also rebuild and train temporal model")
    p_teach.add_argument("--no-temporal", dest="run_temporal", action="store_false")
    p_teach.set_defaults(run_temporal=False)
    p_teach.add_argument("--summary", type=Path, default=Path("data/processed/teach_sign_summary.json"))

    p_run = sub.add_parser("run", help="Run realtime gesture recognition")
    p_run.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_run.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_run.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    p_run.add_argument("--prototype-db", type=Path, default=DEFAULT_PROTOTYPE_DB_PATH)
    p_run.add_argument("--profile", choices=["balanced", "ultra-speed", "speed", "accuracy", "ultra-accuracy", "stage", "production", "smoothhd", "enterprise"], default="balanced")
    p_run.add_argument("--camera", type=int, default=0)
    p_run.add_argument("--width", type=int, default=1280)
    p_run.add_argument("--height", type=int, default=720)
    p_run.add_argument("--camera-fps", type=int, default=60)
    p_run.add_argument("--threshold", type=float, default=0.60)
    p_run.add_argument("--smooth", type=int, default=7)
    p_run.add_argument("--session-log", type=Path, default=DEFAULT_SESSION_LOG_PATH)
    p_run.add_argument("--calibration-profile", type=Path, default=DEFAULT_CALIBRATION_PROFILE_PATH, help="Calibration profile JSON generated by calibrate command")
    p_run.add_argument("--use-calibration-profile", dest="use_calibration_profile", action="store_true", help="Apply calibration profile when available")
    p_run.add_argument("--no-calibration-profile", dest="use_calibration_profile", action="store_false", help="Ignore calibration profile")
    p_run.set_defaults(use_calibration_profile=True)
    p_run.add_argument("--mode", choices=["rules", "ml", "temporal", "hybrid"], default="hybrid")
    p_run.add_argument("--rule-threshold", type=float, default=0.78)
    p_run.add_argument("--infer-interval", type=int, default=1, help="Run heavy inference every N frames")
    p_run.add_argument("--infer-scale", type=float, default=0.75, help="Inference resize scale (0.4-1.0)")
    p_run.add_argument("--model-complexity", type=int, choices=[0, 1], default=0, help="MediaPipe hand model complexity")
    p_run.add_argument("--landmark-smoothing", type=float, default=0.78, help="Landmark smoothing factor (0.0-0.95)")
    p_run.add_argument("--enhance-frame", dest="enhance_frame", action="store_true", help="Enable lightweight video enhancement")
    p_run.add_argument("--no-enhance-frame", dest="enhance_frame", action="store_false", help="Disable video enhancement for max speed")
    p_run.set_defaults(enhance_frame=False)
    p_run.add_argument("--quality-gate", dest="quality_gate", action="store_true", help="Gate predictions on frame/hand quality")
    p_run.add_argument("--no-quality-gate", dest="quality_gate", action="store_false", help="Disable quality gate")
    p_run.set_defaults(quality_gate=True)
    p_run.add_argument("--min-brightness", type=float, default=45.0, help="Minimum brightness for valid prediction")
    p_run.add_argument("--min-blur-var", type=float, default=55.0, help="Minimum Laplacian variance for valid prediction")
    p_run.add_argument("--min-hand-area", type=float, default=0.012, help="Minimum normalized hand bbox area for valid prediction")
    p_run.add_argument("--strict-consensus", action="store_true", help="Require multi-source agreement in hybrid mode")
    p_run.add_argument("--strict-override-conf", type=float, default=0.92, help="Confidence to bypass strict consensus disagreement")
    p_run.add_argument("--ml-min-margin", type=float, default=0.08, help="Minimum top1-top2 ML probability gap")
    p_run.add_argument("--prototypes", dest="use_prototypes", action="store_true", help="Enable prototype matcher")
    p_run.add_argument("--no-prototypes", dest="use_prototypes", action="store_false", help="Disable prototype matcher")
    p_run.set_defaults(use_prototypes=True)
    p_run.add_argument("--prototype-threshold", type=float, default=0.84, help="Prototype cosine similarity threshold")
    p_run.add_argument("--prototype-margin", type=float, default=0.03, help="Prototype top1-top2 margin")
    p_run.add_argument("--deep-runtime", dest="use_deep_model", action="store_true", help="Enable TensorFlow deep model at runtime")
    p_run.add_argument("--no-deep-runtime", dest="use_deep_model", action="store_false", help="Disable TensorFlow deep model at runtime")
    p_run.set_defaults(use_deep_model=False)
    p_run.add_argument("--deep-model", type=Path, default=DEFAULT_DEEP_MODEL_PATH)
    p_run.add_argument("--deep-labels", type=Path, default=DEFAULT_DEEP_LABELS_PATH)
    p_run.add_argument("--deep-metadata", type=Path, default=DEFAULT_DEEP_METADATA_PATH)
    p_run.add_argument("--deep-preprocess", type=Path, default=DEFAULT_DEEP_PREPROCESS_PATH)
    p_run.add_argument("--deep-threshold", type=float, default=0.62, help="Deep model confidence threshold")
    p_run.add_argument("--deep-min-margin", type=float, default=0.06, help="Deep model top1-top2 probability margin")
    p_run.add_argument("--label-hold-sec", type=float, default=0.28, help="Debounce hold time before accepting label")
    p_run.add_argument("--min-stable-frames", type=int, default=3, help="Stable frame count required before speech")
    p_run.add_argument("--auto-speak", dest="auto_speak", action="store_true", help="Auto-speak stable labels")
    p_run.add_argument("--no-auto-speak", dest="auto_speak", action="store_false", help="Disable auto-speaking; use sentence controls manually")
    p_run.set_defaults(auto_speak=True)
    p_run.add_argument("--continuous-sentence", dest="continuous_sentence", action="store_true", help="Auto-build sentence from stable signs and speak on pause")
    p_run.add_argument("--no-continuous-sentence", dest="continuous_sentence", action="store_false", help="Disable continuous sentence mode")
    p_run.set_defaults(continuous_sentence=False)
    p_run.add_argument("--sentence-pause-sec", type=float, default=1.0, help="Pause before auto-speaking built sentence")
    p_run.add_argument("--sentence-append-cooldown", type=float, default=0.35, help="Minimum gap between auto-added words")
    p_run.add_argument("--sentence-max-tokens", type=int, default=14, help="Max tokens kept in live sentence buffer")
    p_run.add_argument("--adaptive-perf", dest="adaptive_perf", action="store_true", help="Auto-adjust inference interval for stable FPS")
    p_run.add_argument("--no-adaptive-perf", dest="adaptive_perf", action="store_false", help="Disable adaptive inference interval tuning")
    p_run.set_defaults(adaptive_perf=True)
    p_run.add_argument("--async-inference", dest="async_inference", action="store_true", help="Run hand tracking on background thread for smoother rendering")
    p_run.add_argument("--sync-inference", dest="async_inference", action="store_false", help="Run hand tracking on main thread")
    p_run.set_defaults(async_inference=True)
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

    p_model_report = sub.add_parser("model-report", help="Generate markdown report from model metadata files")
    p_model_report.add_argument("--out", type=Path, default=Path("data/processed/model_report.md"))
    p_model_report.add_argument("--frame-metadata", type=Path, default=DEFAULT_METADATA_PATH)
    p_model_report.add_argument("--deep-metadata", type=Path, default=DEFAULT_DEEP_METADATA_PATH)
    p_model_report.add_argument("--temporal-metadata", type=Path, default=DEFAULT_TEMPORAL_METADATA_PATH)
    p_model_report.add_argument("--frame-labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_model_report.add_argument("--deep-labels", type=Path, default=DEFAULT_DEEP_LABELS_PATH)
    p_model_report.add_argument("--temporal-labels", type=Path, default=DEFAULT_TEMPORAL_LABELS_PATH)

    p_validate = sub.add_parser("validate-all", help="Run full QA validation benchmark suite")
    p_validate.add_argument("--out", type=Path, default=Path("data/processed/qa_validation_report.json"))
    p_validate.add_argument("--pytest", dest="include_pytest", action="store_true", help="Run full pytest suite")
    p_validate.add_argument("--no-pytest", dest="include_pytest", action="store_false", help="Skip full pytest")
    p_validate.set_defaults(include_pytest=True)
    p_validate.add_argument("--cli-help", dest="include_cli_help", action="store_true", help="Validate all CLI help commands")
    p_validate.add_argument("--no-cli-help", dest="include_cli_help", action="store_false", help="Skip CLI help checks")
    p_validate.set_defaults(include_cli_help=True)
    p_validate.add_argument("--release", dest="include_release", action="store_true", help="Run release bundle smoke check")
    p_validate.add_argument("--no-release", dest="include_release", action="store_false", help="Skip release bundle check")
    p_validate.set_defaults(include_release=True)

    p_final = sub.add_parser("final-test", help="Run final dataset+QA gate and write final report")
    p_final.add_argument("--out-json", type=Path, default=Path("data/processed/final_test_report.json"))
    p_final.add_argument("--out-md", type=Path, default=Path("data/processed/final_test_report.md"))
    p_final.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p_final.add_argument("--min-samples-per-label", type=int, default=5)
    p_final.add_argument("--pytest", dest="include_pytest", action="store_true", help="Run full pytest suite")
    p_final.add_argument("--no-pytest", dest="include_pytest", action="store_false", help="Skip full pytest")
    p_final.set_defaults(include_pytest=True)
    p_final.add_argument("--cli-help", dest="include_cli_help", action="store_true", help="Validate all CLI help commands")
    p_final.add_argument("--no-cli-help", dest="include_cli_help", action="store_false", help="Skip CLI help checks")
    p_final.set_defaults(include_cli_help=True)
    p_final.add_argument("--release", dest="include_release", action="store_true", help="Run release bundle smoke check")
    p_final.add_argument("--no-release", dest="include_release", action="store_false", help="Skip release bundle check")
    p_final.set_defaults(include_release=True)

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

    p_boot_url = sub.add_parser("bootstrap-url", help="End-to-end URL import + image dataset build + AutoML train")
    p_boot_url.add_argument("--url", required=True, help="Direct URL to dataset zip")
    p_boot_url.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_IMAGES_DIR)
    p_boot_url.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p_boot_url.add_argument("--max-per-class", type=int, default=1200)
    p_boot_url.add_argument("--min-free-gb", type=float, default=20.0)
    p_boot_url.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_boot_url.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_boot_url.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    p_boot_url.add_argument("--allow-private-url", dest="allow_private_url", action="store_true", help="Allow localhost/private IP URLs")
    p_boot_url.add_argument("--no-allow-private-url", dest="allow_private_url", action="store_false", help="Block localhost/private IP URLs")
    p_boot_url.set_defaults(allow_private_url=True)
    p_boot_url.add_argument("--allow-file-url", dest="allow_file_url", action="store_true", help="Allow file:// local dataset URLs")
    p_boot_url.add_argument("--no-allow-file-url", dest="allow_file_url", action="store_false", help="Block file:// local dataset URLs")
    p_boot_url.set_defaults(allow_file_url=True)

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
    p_video.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_video.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_video.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    p_video.add_argument("--ml-min-margin", type=float, default=0.08)
    p_video.add_argument("--prototype-db", type=Path, default=DEFAULT_PROTOTYPE_DB_PATH)
    p_video.add_argument("--prototypes", dest="use_prototypes", action="store_true")
    p_video.add_argument("--no-prototypes", dest="use_prototypes", action="store_false")
    p_video.set_defaults(use_prototypes=True)
    p_video.add_argument("--prototype-threshold", type=float, default=0.84)
    p_video.add_argument("--prototype-margin", type=float, default=0.03)
    p_video.add_argument("--deep-runtime", dest="use_deep_model", action="store_true")
    p_video.add_argument("--no-deep-runtime", dest="use_deep_model", action="store_false")
    p_video.set_defaults(use_deep_model=True)
    p_video.add_argument("--deep-model", type=Path, default=DEFAULT_DEEP_MODEL_PATH)
    p_video.add_argument("--deep-labels", type=Path, default=DEFAULT_DEEP_LABELS_PATH)
    p_video.add_argument("--deep-metadata", type=Path, default=DEFAULT_DEEP_METADATA_PATH)
    p_video.add_argument("--deep-preprocess", type=Path, default=DEFAULT_DEEP_PREPROCESS_PATH)
    p_video.add_argument("--deep-threshold", type=float, default=0.62)
    p_video.add_argument("--deep-min-margin", type=float, default=0.06)

    p_adapt_one = sub.add_parser("adapt-sign", help="Learn one sign from reference images/steps")
    p_adapt_one.add_argument("--label", required=True)
    p_adapt_one.add_argument("--images", nargs="+", required=True, help="Image files and/or folders")
    p_adapt_one.add_argument("--phrase", default="", help="Optional spoken phrase mapping for this label")
    p_adapt_one.add_argument("--prototype-db", type=Path, default=DEFAULT_PROTOTYPE_DB_PATH)
    p_adapt_one.add_argument("--min-det-conf", type=float, default=0.35)

    p_adapt_folder = sub.add_parser("adapt-signs-folder", help="Learn multiple signs from label subfolders")
    p_adapt_folder.add_argument("--images-root", type=Path, required=True, help="Root with label subfolders")
    p_adapt_folder.add_argument("--prototype-db", type=Path, default=DEFAULT_PROTOTYPE_DB_PATH)
    p_adapt_folder.add_argument("--max-per-label", type=int, default=0)
    p_adapt_folder.add_argument("--min-det-conf", type=float, default=0.35)

    p_points = sub.add_parser("image-points", help="Read hand points from one image and save overlay")
    p_points.add_argument("--image", type=Path, required=True)
    p_points.add_argument("--out", type=Path, default=Path("data/processed/image_points_overlay.png"))
    p_points.add_argument("--min-det-conf", type=float, default=0.35)

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
            auto_mode=args.auto,
            capture_interval_sec=args.capture_interval,
            min_hand_frames=args.min_hand_frames,
            min_feature_delta=args.min_feature_delta,
            flush_every=args.flush_every,
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
            flush_every=args.flush_every,
        )
        run_sequence_collection(cfg)
        return

    if args.cmd == "calibrate":
        out = run_calibration(
            CalibrationConfig(
                camera_index=args.camera,
                width=args.width,
                height=args.height,
                seconds=args.seconds,
                out_json=args.out,
                model_complexity=args.model_complexity,
                inference_scale=args.infer_scale,
            )
        )
        print(f"Calibration profile saved: {out}")
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

    if args.cmd == "teach-sign":
        summary = run_teach_sign(
            TeachSignConfig(
                label=args.label,
                phrase_text=(args.phrase.strip() if args.phrase else None),
                samples=args.samples,
                camera_index=args.camera,
                width=args.width,
                height=args.height,
                dataset_csv=args.dataset,
                model_path=args.model,
                labels_path=args.labels,
                metadata_path=args.metadata,
                min_samples_per_label=args.min_samples_per_label,
                run_deep=args.run_deep,
                run_temporal=args.run_temporal,
                summary_path=args.summary,
            )
        )
        print(f"Teach-sign summary: {summary}")
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
            min_samples_per_label=args.min_samples_per_label,
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

    if args.cmd == "train-deep":
        try:
            from signifyai.deep_model import DeepTrainConfig, run_deep_training
        except Exception as ex:
            print(f"[ERROR] Deep training dependencies unavailable: {ex}")
            print("Install/upgrade dependencies and retry:")
            print("python -m pip install -r requirements.txt")
            print("python -m pip install -r requirements-deep.txt")
            raise SystemExit(1)
        try:
            result = run_deep_training(
                DeepTrainConfig(
                    dataset_csv=args.dataset,
                    model_path=args.model,
                    labels_path=args.labels,
                    metadata_path=args.metadata,
                    preprocess_path=args.preprocess,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    patience=args.patience,
                    min_samples_per_label=args.min_samples_per_label,
                    seed=args.seed,
                )
            )
        except ValueError as ex:
            msg = str(ex)
            print(f"[ERROR] {msg}")
            if "at least 2 labels" in msg.lower():
                print("You need samples from at least 2 labels with enough count.")
                print("Example:")
                print("python -u .\\src\\main.py collect --label hello --samples 200")
                print("python -u .\\src\\main.py collect --label thanks --samples 200")
            raise SystemExit(1)
        print(f"Deep model accuracy: {result.accuracy:.4f}")
        print(f"Deep model macro F1: {result.f1_macro:.4f}")
        print(f"Epochs trained: {result.epochs_trained}")
        if result.dropped_labels:
            print(f"Dropped low-sample labels: {', '.join(result.dropped_labels)}")
        print("Classification report:")
        print(result.report)
        return

    if args.cmd == "train-all":
        try:
            from signifyai.train_all import TrainAllConfig, run_train_all
        except Exception as ex:
            print(f"[ERROR] Train-all dependencies unavailable: {ex}")
            print("Install/upgrade dependencies and retry:")
            print("python -m pip install -r requirements.txt")
            print("python -m pip install -r requirements-deep.txt")
            raise SystemExit(1)
        try:
            summary = run_train_all(
                TrainAllConfig(
                    dataset_csv=args.dataset,
                    frame_model_path=args.frame_model,
                    frame_labels_path=args.frame_labels,
                    frame_metadata_path=args.frame_metadata,
                    deep_model_path=args.deep_model,
                    deep_labels_path=args.deep_labels,
                    deep_metadata_path=args.deep_metadata,
                    deep_preprocess_path=args.deep_preprocess,
                    sequence_dataset_npz=args.seq_npz,
                    seq_len=args.seq_len,
                    seq_stride=args.seq_stride,
                    temporal_model_path=args.temporal_model,
                    temporal_labels_path=args.temporal_labels,
                    temporal_metadata_path=args.temporal_metadata,
                    summary_path=args.summary,
                    frame_min_samples_per_label=args.frame_min_samples,
                    deep_min_samples_per_label=args.deep_min_samples,
                    deep_epochs=args.deep_epochs,
                    deep_batch_size=args.deep_batch_size,
                    deep_patience=args.deep_patience,
                )
            )
        except ValueError as ex:
            print(f"[ERROR] {ex}")
            raise SystemExit(1)
        print(f"Train-all summary: {summary}")
        return

    if args.cmd == "check-dataset":
        result = run_dataset_check(
            dataset_csv=args.dataset,
            min_samples_per_label=args.min_samples_per_label,
        )
        print(f"Dataset: {args.dataset}")
        print(f"Rows: {result.rows}")
        print(f"Labels: {result.labels}")
        print(f"Min/Max label count: {result.min_count}/{result.max_count}")
        print(result.detail)
        raise SystemExit(0 if result.ok else 1)

    if args.cmd == "data-help":
        print(get_data_help_text())
        return

    if args.cmd == "import-kaggle":
        target = import_from_kaggle(args.slug, args.out_dir, force=args.force)
        print(f"Kaggle dataset imported into: {target}")
        return

    if args.cmd == "import-url":
        count = import_dataset_from_url(
            args.url,
            args.out_dir,
            allow_private_or_local_host=args.allow_private_url,
            allow_file_url=args.allow_file_url,
        )
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

    if args.cmd == "build-video-dataset":
        videos, frames, saved = build_dataset_from_videos(
            BuildVideoDatasetConfig(
                root_dir=args.videos_root,
                out_csv=args.out_csv,
                max_videos_per_class=args.max_videos_per_class,
                max_frames_per_video=args.max_frames_per_video,
                frame_stride=max(1, int(args.frame_stride)),
                min_detection_confidence=args.min_det_conf,
            )
        )
        print(f"Processed videos: {videos}")
        print(f"Processed frames: {frames}")
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
        apply_calibration_profile(args)
        cfg = RealtimeConfig(
            model_path=args.model,
            labels_path=args.labels,
            metadata_path=args.metadata,
            prototype_db_path=args.prototype_db,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            camera_fps=args.camera_fps,
            confidence_threshold=args.threshold,
            smoothing_window=args.smooth,
            min_stable_frames_for_speech=args.min_stable_frames,
            session_log_path=args.session_log,
            mode=args.mode,
            rule_confidence_threshold=args.rule_threshold,
            inference_interval=args.infer_interval,
            inference_scale=args.infer_scale,
            model_complexity=args.model_complexity,
            landmark_smoothing=args.landmark_smoothing,
            auto_speak=args.auto_speak,
            continuous_sentence=args.continuous_sentence,
            sentence_pause_speak_sec=args.sentence_pause_sec,
            sentence_append_cooldown_sec=args.sentence_append_cooldown,
            sentence_max_tokens=args.sentence_max_tokens,
            adaptive_performance=args.adaptive_perf,
            async_inference=args.async_inference,
            target_fps=args.target_fps,
            stage_mode=bool(args.stage),
            label_hold_sec=args.label_hold_sec,
            demo_script=args.demo_script,
            temporal_model_path=args.temporal_model,
            temporal_labels_path=args.temporal_labels,
            temporal_metadata_path=args.temporal_metadata,
            temporal_confidence_threshold=args.temporal_threshold,
            enhance_frame=args.enhance_frame,
            quality_gate=args.quality_gate,
            min_brightness=args.min_brightness,
            min_blur_var=args.min_blur_var,
            min_hand_area=args.min_hand_area,
            strict_consensus=args.strict_consensus,
            strict_override_conf=args.strict_override_conf,
            ml_min_margin=args.ml_min_margin,
            use_prototypes=args.use_prototypes,
            prototype_threshold=args.prototype_threshold,
            prototype_margin=args.prototype_margin,
            use_deep_model=args.use_deep_model,
            deep_model_path=args.deep_model,
            deep_labels_path=args.deep_labels,
            deep_metadata_path=args.deep_metadata,
            deep_preprocess_path=args.deep_preprocess,
            deep_confidence_threshold=args.deep_threshold,
            deep_min_margin=args.deep_min_margin,
        )
        run_realtime(cfg)
        return

    if args.cmd == "report":
        out = build_session_report(ReportConfig(log_path=args.session_log, out_path=args.out))
        print(f"Session report generated: {out}")
        return

    if args.cmd == "model-report":
        out = build_model_report(
            ModelReportConfig(
                out_path=args.out,
                frame_metadata=args.frame_metadata,
                deep_metadata=args.deep_metadata,
                temporal_metadata=args.temporal_metadata,
                frame_labels=args.frame_labels,
                deep_labels=args.deep_labels,
                temporal_labels=args.temporal_labels,
            )
        )
        print(f"Model report generated: {out}")
        return

    if args.cmd == "validate-all":
        out = run_validate_all(
            QAConfig(
                out_json=args.out,
                include_pytest=args.include_pytest,
                include_cli_help_checks=args.include_cli_help,
                include_release_bundle_check=args.include_release,
            )
        )
        print(f"QA validation report: {out}")
        return

    if args.cmd == "final-test":
        out_json, out_md = run_final_test(
            FinalTestConfig(
                out_json=args.out_json,
                out_md=args.out_md,
                include_pytest=args.include_pytest,
                include_cli_help_checks=args.include_cli_help,
                include_release_bundle_check=args.include_release,
                dataset_csv=args.dataset,
                min_samples_per_label=args.min_samples_per_label,
            )
        )
        print(f"Final test JSON report: {out_json}")
        print(f"Final test markdown report: {out_md}")
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

    if args.cmd == "bootstrap-url":
        run_bootstrap_from_url(
            BootstrapURLConfig(
                dataset_url=args.url,
                images_dir=args.out_dir,
                dataset_csv=args.dataset,
                model_path=args.model,
                labels_path=args.labels,
                metadata_path=args.metadata,
                max_per_class=args.max_per_class,
                min_free_gb=args.min_free_gb,
                allow_private_url=args.allow_private_url,
                allow_file_url=args.allow_file_url,
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
                model_path=args.model,
                labels_path=args.labels,
                metadata_path=args.metadata,
                ml_min_margin=args.ml_min_margin,
                prototype_db_path=args.prototype_db,
                use_prototypes=args.use_prototypes,
                prototype_threshold=args.prototype_threshold,
                prototype_margin=args.prototype_margin,
                use_deep_model=args.use_deep_model,
                deep_model_path=args.deep_model,
                deep_labels_path=args.deep_labels,
                deep_metadata_path=args.deep_metadata,
                deep_preprocess_path=args.deep_preprocess,
                deep_confidence_threshold=args.deep_threshold,
                deep_min_margin=args.deep_min_margin,
            )
        )
        print(f"Video inference saved: {out}")
        return

    if args.cmd == "adapt-sign":
        image_paths: list[Path] = []
        for raw in args.images:
            p = Path(raw)
            if p.is_dir():
                image_paths.extend([x for x in sorted(p.rglob("*")) if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
            else:
                image_paths.append(p)
        stats = adapt_sign_from_images(
            label=args.label.strip().lower().replace(" ", "_"),
            image_paths=image_paths,
            out_db=args.prototype_db,
            min_detection_confidence=args.min_det_conf,
            phrase_text=(args.phrase.strip() or None),
        )
        print(f"Images scanned: {stats.total_images}")
        print(f"Images with detected hands: {stats.detected_images}")
        print(f"Prototype vectors saved: {stats.saved_vectors}")
        print(f"Prototype DB: {args.prototype_db}")
        return

    if args.cmd == "adapt-signs-folder":
        stats = adapt_signs_from_folder(
            images_root=args.images_root,
            out_db=args.prototype_db,
            max_per_label=args.max_per_label,
            min_detection_confidence=args.min_det_conf,
        )
        print(f"Images scanned: {stats.total_images}")
        print(f"Images with detected hands: {stats.detected_images}")
        print(f"Prototype vectors saved: {stats.saved_vectors}")
        print(f"Labels added: {', '.join(stats.labels_added) if stats.labels_added else '(none)'}")
        print(f"Prototype DB: {args.prototype_db}")
        return

    if args.cmd == "image-points":
        _, info = extract_points_from_image(
            image_path=args.image,
            min_detection_confidence=args.min_det_conf,
            save_overlay_to=args.out,
        )
        print(f"Hand count detected: {info.hand_count}")
        print(f"Best processing variant: {info.best_variant}")
        print(f"Overlay image: {info.out_image}")
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
        f"argv: {redact_cli_args(sys.argv)}",
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
