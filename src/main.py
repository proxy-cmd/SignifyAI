from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

VERSIONS_DIR = Path("data/landmarks/versions")
MODELS_DIR = Path("data/models")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SignifyAI command runner")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_run_args(sub)
    add_record_args(sub)
    add_dataset_args(sub)
    add_train_args(sub)
    add_eval_args(sub)
    add_promote_args(sub)
    add_api_args(sub)
    return parser


def add_run_args(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    cmd = sub.add_parser("run", help="Run realtime streaming translator")
    cmd.add_argument("--camera", type=int, default=0)
    cmd.add_argument("--width", type=int, default=960)
    cmd.add_argument("--height", type=int, default=540)
    cmd.add_argument("--fps", type=int, default=30)
    cmd.add_argument("--seq-len", type=int, default=24)
    cmd.add_argument("--model-name", type=str, default="")
    cmd.add_argument("--mode", choices=["default", "demo"], default="default")
    cmd.add_argument("--voice", dest="voice", action="store_true")
    cmd.add_argument("--no-voice", dest="voice", action="store_false")
    cmd.set_defaults(voice=True)


def add_record_args(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    cmd = sub.add_parser("record", help="Record continuous intent clips")
    cmd.add_argument("--intent", required=True)
    cmd.add_argument("--clips", type=int, default=8)
    cmd.add_argument("--clip-seconds", type=float, default=1.2)
    cmd.add_argument("--signer", type=str, default="anonymous")
    cmd.add_argument("--consent-raw-video", action="store_true")
    cmd.add_argument("--camera", type=int, default=0)
    cmd.add_argument("--width", type=int, default=960)
    cmd.add_argument("--height", type=int, default=540)
    cmd.add_argument("--fps", type=int, default=30)


def add_dataset_args(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    cmd = sub.add_parser("build-dataset", help="Build signer-aware dataset version manifests")
    cmd.add_argument("--version", required=True)


def add_train_args(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    cmd = sub.add_parser("train-seq", help="Train baseline sequence model")
    cmd.add_argument("--version", required=True)
    cmd.add_argument("--model-name", required=True)


def add_eval_args(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    cmd = sub.add_parser("evaluate", help="Evaluate trained model")
    cmd.add_argument("--version", required=True)
    cmd.add_argument("--model-name", required=True)


def add_promote_args(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    cmd = sub.add_parser("promote", help="Promote trained model as active")
    cmd.add_argument("--model-name", required=True)
    cmd.add_argument("--notes", default="")


def add_api_args(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    cmd = sub.add_parser("serve-api", help="Serve API + web dashboard")
    cmd.add_argument("--host", default="127.0.0.1")
    cmd.add_argument("--port", type=int, default=8000)


def run_cmd(args: argparse.Namespace) -> None:
    from signifyai.runtime.stream import RuntimeConfig, StreamingRuntime

    cfg = RuntimeConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        seq_len=args.seq_len,
        model_name=(args.model_name or None),
        mode=args.mode,
        voice_enabled=bool(args.voice),
    )
    runner = StreamingRuntime(cfg)
    runner.run()


def record_cmd(args: argparse.Namespace) -> None:
    from signifyai.runtime.record_intent import IntentRecorder, RecordConfig

    cfg = RecordConfig(
        intent_id=args.intent,
        clips=args.clips,
        clip_seconds=args.clip_seconds,
        signer_id=args.signer,
        consent_raw_video=bool(args.consent_raw_video),
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    rec = IntentRecorder(cfg)
    print(rec.run())


def build_dataset_cmd(args: argparse.Namespace) -> None:
    from signifyai.data.dataset_version import DsBuilder, DsCfg

    ds = DsBuilder(DsCfg())
    print(ds.build(args.version))


def train_seq_cmd(args: argparse.Namespace) -> None:
    from signifyai.model.sequence_model import SeqModel, TrainCfg

    model = SeqModel()
    cfg = TrainCfg(
        version_dir=VERSIONS_DIR / args.version,
        model_name=args.model_name,
        out_dir=MODELS_DIR,
    )
    out = model.train(cfg)
    print(out)


def evaluate_cmd(args: argparse.Namespace) -> None:
    from signifyai.model.sequence_model import SeqModel

    model = SeqModel()
    res = model.eval(
        version_dir=VERSIONS_DIR / args.version,
        model_name=args.model_name,
        out_dir=MODELS_DIR,
    )
    print({"accuracy": res.accuracy, "samples": res.samples})
    print(res.report)


def promote_cmd(args: argparse.Namespace) -> None:
    from signifyai.model.registry import ModelRegistry

    reg = ModelRegistry()
    print(reg.promote_model(args.model_name, notes=args.notes))


def serve_api_cmd(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("signifyai.api.app:create_app", host=args.host, port=args.port, reload=False, factory=True)


def main() -> None:
    args = make_parser().parse_args()
    handlers: dict[str, Callable[[argparse.Namespace], None]] = {
        "run": run_cmd,
        "record": record_cmd,
        "build-dataset": build_dataset_cmd,
        "train-seq": train_seq_cmd,
        "evaluate": evaluate_cmd,
        "promote": promote_cmd,
        "serve-api": serve_api_cmd,
    }
    handlers[args.cmd](args)


if __name__ == "__main__":
    main()
