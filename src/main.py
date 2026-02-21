from __future__ import annotations

import argparse
from pathlib import Path

from signifyai.collect import CollectConfig, run_collection
from signifyai.config import DEFAULT_DATASET_PATH, DEFAULT_LABELS_PATH, DEFAULT_MODEL_PATH, DEFAULT_SESSION_LOG_PATH
from signifyai.realtime import RealtimeConfig, run_realtime
from signifyai.train import TrainConfig, run_training


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SignifyAI command runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

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

    p_run = sub.add_parser("run", help="Run realtime gesture recognition")
    p_run.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_run.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    p_run.add_argument("--camera", type=int, default=0)
    p_run.add_argument("--width", type=int, default=960)
    p_run.add_argument("--height", type=int, default=720)
    p_run.add_argument("--threshold", type=float, default=0.60)
    p_run.add_argument("--smooth", type=int, default=7)
    p_run.add_argument("--session-log", type=Path, default=DEFAULT_SESSION_LOG_PATH)

    return parser


def main() -> None:
    parser = make_parser()
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
        )
        run_training(cfg)
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
        )
        run_realtime(cfg)
        return

    raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
