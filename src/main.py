import argparse
import json
from pathlib import Path

VER_DIR = Path("data/landmarks/versions")
MODEL_DIR = Path("data/models")
GLOBAL_MODEL = "signifyai_global"


# ---------- parser ----------
def make_parser():
    parser = argparse.ArgumentParser(description="SignifyAI command runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    add_run_cmd(sub)
    add_record_cmd(sub)
    add_build_cmd(sub)
    add_health_cmd(sub)
    add_train_cmd(sub)
    add_eval_cmd(sub)
    add_promote_cmd(sub)

    return parser


def add_run_cmd(sub):
    cmd = sub.add_parser("run", help="Run realtime streaming translator")
    cmd.add_argument("--camera", type=int, default=0)
    cmd.add_argument("--width", type=int, default=960)
    cmd.add_argument("--height", type=int, default=540)
    cmd.add_argument("--fps", type=int, default=30)
    cmd.add_argument("--seq-len", type=int, default=24)
    cmd.add_argument("--model-name", type=str, default="")
    cmd.add_argument("--mode", choices=["default", "hybrid", "demo", "aid"], default="hybrid")
    cmd.add_argument("--voice", dest="voice", action="store_true")
    cmd.add_argument("--no-voice", dest="voice", action="store_false")
    cmd.set_defaults(voice=True)


def add_record_cmd(sub):
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


def add_build_cmd(sub):
    cmd = sub.add_parser("build-dataset", help="Build signer-aware dataset version manifests")
    cmd.add_argument("--version", required=True)


def add_health_cmd(sub):
    cmd = sub.add_parser("dataset-health", help="Show dataset health report")
    cmd.add_argument("--version", required=True)


def add_train_cmd(sub):
    cmd = sub.add_parser("train-seq", help="Train sequence model")
    cmd.add_argument("--version", required=True)
    cmd.add_argument("--model-name", default=GLOBAL_MODEL)
    cmd.add_argument("--seq-len", type=int, default=24)


def add_eval_cmd(sub):
    cmd = sub.add_parser("evaluate", help="Evaluate trained model")
    cmd.add_argument("--version", required=True)
    cmd.add_argument("--model-name", default=GLOBAL_MODEL)


def add_promote_cmd(sub):
    cmd = sub.add_parser("promote", help="Promote trained model as active")
    cmd.add_argument("--model-name", default=GLOBAL_MODEL)
    cmd.add_argument("--notes", default="")


# ---------- actions ----------
def do_run(args):
    from modes.realtime_translator import LiveCfg, LiveRunner

    cfg = LiveCfg(
        cam_idx=args.camera,
        w=args.width,
        h=args.height,
        fps=args.fps,
        seq_len=args.seq_len,
        model_name=(args.model_name or None),
        mode=args.mode,
        voice=bool(args.voice),
    )
    runner = LiveRunner(cfg)
    runner.run()


def do_record(args):
    from dataset.recording import RecCfg, RecSession

    cfg = RecCfg(
        intent=args.intent,
        clips=args.clips,
        clip_sec=args.clip_seconds,
        signer=args.signer,
        consent_raw=bool(args.consent_raw_video),
        cam_idx=args.camera,
        w=args.width,
        h=args.height,
        fps=args.fps,
    )
    rec = RecSession(cfg)
    print(rec.run())


def do_build(args):
    from dataset.dataset_builder import DataBuildCfg, DataBuilder

    cfg = DataBuildCfg()
    builder = DataBuilder(cfg)
    print(builder.build(args.version))


def do_health(args):
    from dataset.dataset_builder import check_dataset

    print(check_dataset(VER_DIR / args.version))


def do_train(args):
    from dataset.dataset_builder import check_dataset
    from model.sequence_model import SeqCfg, SeqTrainer

    health = check_dataset(VER_DIR / args.version)
    print({"dataset_health": health})
    if not health.get("can_train", False):
        print("Train blocked: dataset is not ready. Fix warnings above and rebuild dataset.")
        return

    model_name = args.model_name or GLOBAL_MODEL
    cfg = SeqCfg(
        version_dir=VER_DIR / args.version,
        model_name=model_name,
        out_dir=MODEL_DIR,
        seq_len=args.seq_len,
    )

    trainer = SeqTrainer()
    try:
        out = trainer.train(cfg)
        print(out)
        new_acc = float(out.get("val_accuracy", 0.0))
        auto_promote_if_better(model_name, new_acc)
    except ValueError as ex:
        print(f"Train failed: {ex}")


def do_eval(args):
    from model.sequence_model import SeqTrainer

    model_name = args.model_name or GLOBAL_MODEL
    trainer = SeqTrainer()
    try:
        res = trainer.eval(version_dir=VER_DIR / args.version, model_name=model_name, out_dir=MODEL_DIR)
        print({"accuracy": res.acc, "samples": res.samples})
        print(res.report)
    except FileNotFoundError:
        print(f"Model not found. Train first with: train-seq --version {args.version} --model-name {model_name}")


def do_promote(args):
    from model.model_manager import ModelHub

    model_name = args.model_name or GLOBAL_MODEL
    hub = ModelHub()
    print(hub.promote(model_name, notes=args.notes))


# ---------- helpers ----------
def read_val_acc(model_name):
    path = MODEL_DIR / f"{model_name}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("val_accuracy", 0.0))
    except Exception:
        return None


def auto_promote_if_better(model_name, new_acc, min_gain=0.01):
    from model.model_manager import ModelHub

    hub = ModelHub()
    active = hub.active()

    if active is None:
        hub.promote(model_name, notes="auto-promote: first global model")
        print(f"Auto-promoted {model_name} (first model).")
        return

    if active == model_name:
        print(f"Model {model_name} is already active.")
        return

    old_acc = read_val_acc(active)
    if old_acc is None:
        hub.promote(model_name, notes="auto-promote: previous model had no metadata")
        print(f"Auto-promoted {model_name} (previous model metadata missing).")
        return

    # promote if new model is at least min_gain better
    if new_acc >= (old_acc + min_gain):
        note = f"auto-promote: val_accuracy {new_acc:.3f} > {old_acc:.3f}"
        hub.promote(model_name, notes=note)
        print(f"Auto-promoted {model_name}: {new_acc:.3f} vs active {old_acc:.3f}.")
    else:
        print(f"Kept active model {active}: {old_acc:.3f} vs new {new_acc:.3f}.")


def main():
    args = make_parser().parse_args()
    if args.cmd == "run":
        do_run(args)
    elif args.cmd == "record":
        do_record(args)
    elif args.cmd == "build-dataset":
        do_build(args)
    elif args.cmd == "dataset-health":
        do_health(args)
    elif args.cmd == "train-seq":
        do_train(args)
    elif args.cmd == "evaluate":
        do_eval(args)
    elif args.cmd == "promote":
        do_promote(args)
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
