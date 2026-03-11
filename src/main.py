from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SignifyAI command runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run realtime streaming translator")
    p_run.add_argument("--camera", type=int, default=0)
    p_run.add_argument("--width", type=int, default=960)
    p_run.add_argument("--height", type=int, default=540)
    p_run.add_argument("--fps", type=int, default=30)
    p_run.add_argument("--seq-len", type=int, default=24)
    p_run.add_argument("--model-name", type=str, default="")
    p_run.add_argument("--voice", dest="voice", action="store_true")
    p_run.add_argument("--no-voice", dest="voice", action="store_false")
    p_run.set_defaults(voice=True)

    p_rec = sub.add_parser("record", help="Record continuous intent clips")
    p_rec.add_argument("--intent", required=True)
    p_rec.add_argument("--clips", type=int, default=8)
    p_rec.add_argument("--clip-seconds", type=float, default=1.2)
    p_rec.add_argument("--signer", type=str, default="anonymous")
    p_rec.add_argument("--consent-raw-video", action="store_true")
    p_rec.add_argument("--camera", type=int, default=0)
    p_rec.add_argument("--width", type=int, default=960)
    p_rec.add_argument("--height", type=int, default=540)
    p_rec.add_argument("--fps", type=int, default=30)

    p_ds = sub.add_parser("build-dataset", help="Build signer-aware dataset version manifests")
    p_ds.add_argument("--version", required=True)

    p_train = sub.add_parser("train-seq", help="Train baseline sequence model")
    p_train.add_argument("--version", required=True)
    p_train.add_argument("--model-name", required=True)

    p_eval = sub.add_parser("evaluate", help="Evaluate trained model")
    p_eval.add_argument("--version", required=True)
    p_eval.add_argument("--model-name", required=True)

    p_promote = sub.add_parser("promote", help="Promote trained model as active")
    p_promote.add_argument("--model-name", required=True)
    p_promote.add_argument("--notes", default="")

    p_api = sub.add_parser("serve-api", help="Serve API + web dashboard")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8000)

    return parser


def _run_cmd(args: argparse.Namespace) -> None:
    from signifyai.runtime.stream import RuntimeConfig as RunCfg
    from signifyai.runtime.stream import StreamingRuntime as Runner

    app = Runner(
        RunCfg(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seq_len=args.seq_len,
            model_name=(args.model_name or None),
            voice_enabled=bool(args.voice),
        )
    )
    app.run()


def _record_cmd(args: argparse.Namespace) -> None:
    from signifyai.runtime.record_intent import IntentRecorder as Recorder
    from signifyai.runtime.record_intent import RecordConfig as RecCfg

    rec = Recorder(
        RecCfg(
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
    )
    print(rec.run())


def _build_dataset_cmd(args: argparse.Namespace) -> None:
    from signifyai.data.dataset_version import DsBuilder, DsCfg

    ds = DsBuilder(DsCfg())
    print(ds.build(args.version))


def _train_seq_cmd(args: argparse.Namespace) -> None:
    from signifyai.model.sequence_model import SeqModel, TrainCfg

    model = SeqModel()
    print(
        model.train(
            TrainCfg(
                version_dir=Path("data/landmarks/versions") / args.version,
                model_name=args.model_name,
                out_dir=Path("data/models"),
            )
        )
    )


def _evaluate_cmd(args: argparse.Namespace) -> None:
    from signifyai.model.sequence_model import SeqModel

    model = SeqModel()
    res = model.eval(
        version_dir=Path("data/landmarks/versions") / args.version,
        model_name=args.model_name,
        out_dir=Path("data/models"),
    )
    print({"accuracy": res.accuracy, "samples": res.samples})
    print(res.report)


def _promote_cmd(args: argparse.Namespace) -> None:
    from signifyai.model.registry import ModelRegistry as Reg

    reg = Reg()
    print(reg.promote_model(args.model_name, notes=args.notes))


def _serve_api_cmd(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("signifyai.api.app:create_app", host=args.host, port=args.port, reload=False, factory=True)


def main() -> None:
    args = make_parser().parse_args()
    handlers: dict[str, Callable[[argparse.Namespace], None]] = {
        "run": _run_cmd,
        "record": _record_cmd,
        "build-dataset": _build_dataset_cmd,
        "train-seq": _train_seq_cmd,
        "evaluate": _evaluate_cmd,
        "promote": _promote_cmd,
        "serve-api": _serve_api_cmd,
    }
    handlers[args.cmd](args)


if __name__ == "__main__":
    main()
