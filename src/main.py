import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import warnings

VER_DIR = Path("data/landmarks/versions")
MODEL_DIR = Path("data/models")
CUSTOM_DATASET = "custom"
GLOBAL_DATASET = "global"
CUSTOM_MODEL = "custom"
GLOBAL_MODEL = "global"
LEGACY_CUSTOM_DATASET = "v2"
LEGACY_CUSTOM_MODEL = "signifyai_global"
LEGACY_GLOBAL_MODEL = "signifyai_global_kaggle"
DEFAULT_SEQ_LEN = 24
EXTERNAL_DATA_DIR = Path("data/external")
GLOBAL_RAW_DIR = Path("data/landmarks/raw/external_global")


def ensure_project_python():
    if os.environ.get("SIGNIFYAI_SKIP_REEXEC") == "1":
        return

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")

    venv_python = Path(".venv") / "Scripts" / "python.exe"
    if not venv_python.exists():
        return

    cur = Path(sys.executable).resolve()
    target = venv_python.resolve()
    if cur == target:
        return

    env = dict(os.environ)
    env["SIGNIFYAI_SKIP_REEXEC"] = "1"
    cmd = [str(target)] + sys.argv
    raise SystemExit(subprocess.call(cmd, env=env))


def make_parser():
    parser = argparse.ArgumentParser(description="SignifyAI command runner")
    sub = parser.add_subparsers(dest="cmd", required=False)

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
    cmd.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    cmd.add_argument("--model-name", type=str, default=CUSTOM_MODEL)
    cmd.add_argument("--global-model-name", type=str, default=GLOBAL_MODEL)
    cmd.add_argument("--mode", choices=["default", "hybrid", "demo", "aid", "eye"], default="hybrid")
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
    cmd.add_argument("--model-name", default=CUSTOM_MODEL)
    cmd.add_argument("--seq-len", type=int, default=24)
    cmd.add_argument("--algo", choices=["auto", "logreg"], default="auto")


def add_eval_cmd(sub):
    cmd = sub.add_parser("evaluate", help="Evaluate trained model")
    cmd.add_argument("--version", required=True)
    cmd.add_argument("--model-name", default=CUSTOM_MODEL)


def add_promote_cmd(sub):
    cmd = sub.add_parser("promote", help="Promote trained model as active")
    cmd.add_argument("--model-name", default=CUSTOM_MODEL)
    cmd.add_argument("--notes", default="")


def setup_simple_layout():
    VER_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    legacy_custom = VER_DIR / LEGACY_CUSTOM_DATASET
    custom_dir = VER_DIR / CUSTOM_DATASET
    if (not custom_dir.exists()) and legacy_custom.exists():
        custom_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "summary.json"):
            src = legacy_custom / name
            dst = custom_dir / name
            if src.exists() and (not dst.exists()):
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    global_dir = VER_DIR / GLOBAL_DATASET
    if not global_dir.exists():
        global_dir.mkdir(parents=True, exist_ok=True)


    model_map = [
        (CUSTOM_MODEL, LEGACY_CUSTOM_MODEL),
        (GLOBAL_MODEL, LEGACY_GLOBAL_MODEL),
    ]
    for simple_name, legacy_name in model_map:
        dst_job = MODEL_DIR / f"{simple_name}.joblib"
        dst_json = MODEL_DIR / f"{simple_name}.json"
        src_job = MODEL_DIR / f"{legacy_name}.joblib"
        src_json = MODEL_DIR / f"{legacy_name}.json"
        if (not dst_job.exists()) and src_job.exists():
            dst_job.write_bytes(src_job.read_bytes())
        if (not dst_json.exists()) and src_json.exists():
            dst_json.write_text(src_json.read_text(encoding="utf-8"), encoding="utf-8")



def do_run(args):
    from modes.realtime_translator import LiveCfg, LiveRunner

    cfg = LiveCfg(
        cam_idx=int(getattr(args, "camera", 0)),
        w=int(getattr(args, "width", 960)),
        h=int(getattr(args, "height", 540)),
        fps=int(getattr(args, "fps", 30)),
        seq_len=int(getattr(args, "seq_len", DEFAULT_SEQ_LEN)),
        model_name=(getattr(args, "model_name", CUSTOM_MODEL) or CUSTOM_MODEL),
        global_model_name=str(getattr(args, "global_model_name", GLOBAL_MODEL)),
        mode=str(getattr(args, "mode", "default")),
        voice=bool(getattr(args, "voice", True)),
    )
    runner = LiveRunner(cfg)
    runner.run()


def _input_int(prompt, default):
    raw = input(prompt).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _input_float(prompt, default):
    raw = input(prompt).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def run_menu():
    while True:
        print("\n=== SignifyAI Menu ===")
        print("1) Realtime translation")
        print("2) Demo mode (hardcoded signs)")
        print("3) Emergency mode (hardcoded emergency signs)")
        print("4) Eye assist mode (separate test mode)")
        print("A) Advanced menu (record/train/evaluate)")
        print("Q) Quit")

        choice = input("Select option (1-4, A, Q): ").strip().lower()

        if choice == "1":
            args = argparse.Namespace(
                camera=_input_int("Camera index [0]: ", 0),
                width=_input_int("Width [960]: ", 960),
                height=_input_int("Height [540]: ", 540),
                fps=_input_int("FPS [30]: ", 30),
                seq_len=DEFAULT_SEQ_LEN,
                model_name=CUSTOM_MODEL,
                global_model_name=GLOBAL_MODEL,
                mode="default",
                voice=True,
            )
            do_run(args)
            continue

        if choice == "2":
            args = argparse.Namespace(
                camera=_input_int("Camera index [0]: ", 0),
                width=_input_int("Width [960]: ", 960),
                height=_input_int("Height [540]: ", 540),
                fps=_input_int("FPS [30]: ", 30),
                seq_len=DEFAULT_SEQ_LEN,
                model_name=CUSTOM_MODEL,
                global_model_name=GLOBAL_MODEL,
                mode="demo",
                voice=True,
            )
            do_run(args)
            continue

        if choice == "3":
            args = argparse.Namespace(
                camera=_input_int("Camera index [0]: ", 0),
                width=_input_int("Width [960]: ", 960),
                height=_input_int("Height [540]: ", 540),
                fps=_input_int("FPS [30]: ", 30),
                seq_len=DEFAULT_SEQ_LEN,
                model_name=CUSTOM_MODEL,
                global_model_name=GLOBAL_MODEL,
                mode="aid",
                voice=True,
            )
            do_run(args)
            continue

        if choice == "4":
            args = argparse.Namespace(
                camera=_input_int("Camera index [0]: ", 0),
                width=_input_int("Width [960]: ", 960),
                height=_input_int("Height [540]: ", 540),
                fps=_input_int("FPS [30]: ", 30),
                seq_len=DEFAULT_SEQ_LEN,
                model_name=CUSTOM_MODEL,
                global_model_name=GLOBAL_MODEL,
                mode="eye",
                voice=True,
            )
            do_run(args)
            continue

        if choice == "a":
            run_advanced_menu()
            continue

        if choice == "q":
            print("Exiting SignifyAI menu.")
            return

        print("Invalid option. Please run again.")


def run_advanced_menu():
    while True:
        print("\n=== Advanced Menu ===")
        print("4) Record new sign clips (custom)")
        print("5) Build custom dataset")
        print("6) Train custom model (log-reg)")
        print("7) Build + train global model from external data (log-reg)")
        print("8) Evaluate custom + global")
        print("B) Back to main menu")

        choice = input("Select option (4-8 or B): ").strip().lower()

        if choice == "4":
            args = argparse.Namespace(
                intent=input("Intent text: ").strip(),
                clips=_input_int("Clips [8]: ", 8),
                clip_seconds=_input_float("Clip seconds [1.2]: ", 1.2),
                signer=input("Signer id [anonymous]: ").strip() or "anonymous",
                consent_raw_video=False,
                camera=0,
                width=960,
                height=540,
                fps=30,
            )
            do_record(args)
            continue

        if choice == "5":
            args = argparse.Namespace(version=CUSTOM_DATASET)
            do_build(args)
            continue

        if choice == "6":
            args = argparse.Namespace(
                version=CUSTOM_DATASET,
                model_name=CUSTOM_MODEL,
                seq_len=DEFAULT_SEQ_LEN,
                algo="logreg",
            )
            do_train(args)
            continue

        if choice == "7":
            summary = build_global_dataset_from_external()
            if summary is None:
                print("Global dataset build failed. Check data/external folder.")
                continue
            args = argparse.Namespace(
                version=GLOBAL_DATASET,
                model_name=GLOBAL_MODEL,
                seq_len=1,
                algo="auto",
            )
            do_train(args)
            continue

        if choice == "8":
            print("\n[custom model report]")
            do_eval(argparse.Namespace(version=CUSTOM_DATASET, model_name=CUSTOM_MODEL))
            print("\n[global model report]")
            do_eval(argparse.Namespace(version=GLOBAL_DATASET, model_name=GLOBAL_MODEL))
            continue

        if choice == "b":
            return

        print("Invalid option. Please run again.")


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
    from sklearn.exceptions import ConvergenceWarning

    health = check_dataset(VER_DIR / args.version)
    print("\n=== Training Summary ===")
    print(f"Dataset: {args.version}")
    print(f"Model: {args.model_name}")
    print(f"Clips: train={health['clips']['train']} | val={health['clips']['val']} | test={health['clips']['test']}")
    if health.get("warnings"):
        for w in health["warnings"]:
            print(f"[warn] {w}")
    if not health.get("can_train", False):
        print("Train blocked: dataset is not ready. Fix warnings above and rebuild dataset.")
        return

    model_name = args.model_name or GLOBAL_MODEL
    cfg = SeqCfg(
        version_dir=VER_DIR / args.version,
        model_name=model_name,
        out_dir=MODEL_DIR,
        seq_len=args.seq_len,
        algo=str(getattr(args, "algo", "auto")),
    )

    trainer = SeqTrainer()
    try:
        print("Training model... please wait")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            out = trainer.train(cfg)
        print("=== Train Result ===")
        print(f"Saved: {out['model_path']}")
        print(f"Best algo: {out['best_algo']}")
        print(f"Accuracy: val={out['val_accuracy'] * 100:.2f}% | test={out['test_accuracy'] * 100:.2f}%")
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
        print("\n=== Evaluation Result ===")
        print(f"Accuracy: {res.acc * 100:.2f}%")
        print(f"Samples: {res.samples}")
        print(res.report)
    except FileNotFoundError:
        print(f"Model not found. Train first with: train-seq --version {args.version} --model-name {model_name}")


def _iter_external_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def _split_rows_by_label(rows, train_ratio=0.7, val_ratio=0.15):
    by_label = {}
    for row in rows:
        by_label.setdefault(str(row.get("intent_id", "unknown")), []).append(row)

    splits = {"train": [], "val": [], "test": []}
    rng = random.Random(42)
    for label_rows in by_label.values():
        items = label_rows[:]
        rng.shuffle(items)
        n = len(items)
        if n < 3:
            splits["train"].extend(items)
            continue

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        n_test = n - n_train - n_val

        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val : n_train + n_val + n_test])
    return splits


def build_global_dataset_from_external():
    import cv2
    import numpy as np

    from core.hand_detection import HandCfg, HandDetector
    from dataset.recording import frame_to_vec

    images = _iter_external_images(EXTERNAL_DATA_DIR)
    if not images:
        print("No images found in data/external")
        return None

    print("\n=== Global Dataset Build ===")
    print(f"Found external images: {len(images)}")
    print("Applying strict quality filter for cleaner global model...")

    if GLOBAL_RAW_DIR.exists():
        shutil.rmtree(GLOBAL_RAW_DIR)
    GLOBAL_RAW_DIR.mkdir(parents=True, exist_ok=True)

    det = HandDetector(HandCfg(scale=1.0))
    rows = []
    kept = 0
    skipped = 0
    try:
        for i, img_path in enumerate(images, start=1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                skipped += 1
                continue

            data = det.process(frame)
            if data.left is None and data.right is None:
                skipped += 1
                continue

            label = img_path.parent.name.strip().lower().replace(" ", "_")
            clip_id = f"clip_{kept + 1:06d}"
            npz_path = GLOBAL_RAW_DIR / f"{clip_id}.npz"

            seq = np.asarray([frame_to_vec(data)], dtype=np.float32)
            ts = np.asarray([data.ts_ms], dtype=np.int64)
            np.savez_compressed(npz_path, sequence=seq, timestamps=ts)

            # quality filter: keep clear and visible hand samples only
            q = dict(data.quality)
            if float(q.get("hand_area", 0.0)) < 0.07:
                skipped += 1
                continue
            if float(q.get("blur", 0.0)) < 150.0:
                skipped += 1
                continue

            rows.append(
                {
                    "session_id": "external_global",
                    "clip_id": clip_id,
                    "intent_id": label,
                    "signer_id": "external",
                    "consent_raw_video": False,
                    "npz_path": str(npz_path),
                    "frames": int(seq.shape[0]),
                    "quality": q,
                }
            )
            kept += 1

            if i % 200 == 0:
                print(f"Progress: {i}/{len(images)} images")
    finally:
        det.close()

    if not rows:
        print("No usable hand landmarks extracted from external images.")
        return None

    # keep only labels with enough samples for stable training
    by_label = {}
    for row in rows:
        by_label.setdefault(str(row.get("intent_id", "unknown")), []).append(row)
    filtered_rows = []
    for label_rows in by_label.values():
        if len(label_rows) >= 14:
            filtered_rows.extend(label_rows)

    if not filtered_rows:
        print("No labels have enough high-quality samples after filtering.")
        return None

    splits = _split_rows_by_label(filtered_rows)
    out_dir = VER_DIR / GLOBAL_DATASET
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows in splits.items():
        payload = "\n".join(json.dumps(r) for r in split_rows)
        (out_dir / f"{split_name}.jsonl").write_text(payload, encoding="utf-8")

    summary = {
        "version": GLOBAL_DATASET,
        "source": "external_images",
        "total_samples": len(filtered_rows),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "labels": len({r["intent_id"] for r in filtered_rows}),
        "skipped_images": skipped,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== Global Dataset Ready ===")
    print(f"Usable samples: {summary['total_samples']}")
    print(f"Split: train={summary['train']} | val={summary['val']} | test={summary['test']}")
    print(f"Labels kept: {summary['labels']}")
    print(f"Skipped images: {summary['skipped_images']}")
    return summary


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
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")

    ensure_project_python()
    setup_simple_layout()

    parser = make_parser()
    args = parser.parse_args()

    if not args.cmd:
        run_menu()
        return

    actions = {
        "run": do_run,
        "record": do_record,
        "build-dataset": do_build,
        "dataset-health": do_health,
        "train-seq": do_train,
        "evaluate": do_eval,
        "promote": do_promote,
    }
    action = actions.get(args.cmd)
    if action is None:
        raise RuntimeError(f"Unknown command: {args.cmd}")
    action(args)


if __name__ == "__main__":
    main()
