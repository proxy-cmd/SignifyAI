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
LIVE_TEACH_DIR = Path("data/landmarks/raw/live_teach")
SIGN_PROTO_PATH = Path("data/models/sign_prototypes.json")


def ensure_venv_python():
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
    add_rollback_cmd(sub)

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
    cmd.add_argument("--mode", choices=["default", "hybrid", "aid", "eye", "teach"], default="hybrid")
    cmd.add_argument("--uncertainty-min-conf", type=float, default=0.48)
    cmd.add_argument("--speech-repeat-cooldown-sec", type=float, default=1.8)
    cmd.add_argument("--speech-global-cooldown-sec", type=float, default=0.35)
    cmd.add_argument("--watchdog-reset-sec", type=float, default=5.0)
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
    cmd.add_argument("--min-val-acc", type=float, default=0.0)
    cmd.add_argument("--min-gain", type=float, default=0.0)
    cmd.add_argument("--force", action="store_true")


def add_rollback_cmd(sub):
    cmd = sub.add_parser("rollback", help="Rollback active model to previous promoted model")
    cmd.add_argument("--notes", default="manual rollback")


def setup_layout():
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
        uncertainty_min_conf=float(getattr(args, "uncertainty_min_conf", 0.48)),
        speech_repeat_cooldown_sec=float(getattr(args, "speech_repeat_cooldown_sec", 1.8)),
        speech_global_cooldown_sec=float(getattr(args, "speech_global_cooldown_sec", 0.35)),
        watchdog_reset_sec=float(getattr(args, "watchdog_reset_sec", 5.0)),
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


def _norm_sign(name):
    return str(name).strip().lower().replace(" ", "_")


def _read_json_file(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_live_rows():
    clips = LIVE_TEACH_DIR / "clips.jsonl"
    if not clips.exists():
        return []
    rows = []
    for line in clips.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _write_live_rows(rows):
    clips = LIVE_TEACH_DIR / "clips.jsonl"
    LIVE_TEACH_DIR.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(r) for r in rows)
    if payload:
        payload += "\n"
    clips.write_text(payload, encoding="utf-8")


def _list_taught_signs():
    proto = _read_json_file(SIGN_PROTO_PATH, {})
    rows = _read_live_rows()
    names = set()
    for k in proto.keys():
        nk = _norm_sign(k)
        if nk:
            names.add(nk)
    for r in rows:
        nk = _norm_sign(r.get("intent_id", ""))
        if nk:
            names.add(nk)
    return sorted(names)


def _delete_sign(name):
    key = _norm_sign(name)

    proto = _read_json_file(SIGN_PROTO_PATH, {})
    removed_proto = False
    if key in proto:
        del proto[key]
        _write_json_file(SIGN_PROTO_PATH, proto)
        removed_proto = True

    rows = _read_live_rows()
    keep = []
    removed_rows = 0
    removed_files = 0
    for row in rows:
        if _norm_sign(row.get("intent_id", "")) == key:
            removed_rows += 1
            npz_path = Path(str(row.get("npz_path", "")))
            if npz_path.exists():
                try:
                    npz_path.unlink()
                    removed_files += 1
                except Exception:
                    pass
        else:
            keep.append(row)
    _write_live_rows(keep)

    print(f"Deleted sign '{key}': prototype_removed={removed_proto}, clips_removed={removed_rows}, files_removed={removed_files}")


def _rename_sign(old_name, new_name):
    old_key = _norm_sign(old_name)
    new_key = _norm_sign(new_name)
    if not new_key:
        print("New name is empty. Cancelled.")
        return
    if old_key == new_key:
        print("Old and new names are the same. Nothing changed.")
        return

    proto = _read_json_file(SIGN_PROTO_PATH, {})
    if old_key in proto:
        proto[new_key] = proto[old_key]
        del proto[old_key]
        _write_json_file(SIGN_PROTO_PATH, proto)

    rows = _read_live_rows()
    changed = 0
    for row in rows:
        if _norm_sign(row.get("intent_id", "")) == old_key:
            row["intent_id"] = new_key
            changed += 1
    _write_live_rows(rows)

    print(f"Renamed sign '{old_key}' -> '{new_key}' (updated clips: {changed})")


def run_signs_menu():
    while True:
        print("\n=== Manage Taught Signs ===")
        print("1) Delete sign")
        print("2) Modify sign name")
        print("B) Back")
        action = input("Select (1/2/B): ").strip().lower()

        if action == "b":
            return
        if action not in {"1", "2"}:
            print("Invalid option.")
            continue

        names = _list_taught_signs()
        if not names:
            print("No taught signs found.")
            continue

        print("\nCurrent taught signs:")
        for i, name in enumerate(names, start=1):
            print(f"{i}) {name}")

        idx_raw = input("Choose sign number: ").strip()
        try:
            idx = int(idx_raw)
        except ValueError:
            print("Invalid number.")
            continue
        if idx < 1 or idx > len(names):
            print("Out of range.")
            continue

        chosen = names[idx - 1]
        if action == "1":
            confirm = input(f"Delete '{chosen}'? (y/N): ").strip().lower()
            if confirm == "y":
                _delete_sign(chosen)
            else:
                print("Cancelled.")
            continue

        new_name = input(f"New name for '{chosen}': ").strip()
        _rename_sign(chosen, new_name)


def run_menu():
    while True:
        print("\n=== SignifyAI Menu ===")
        print("1) Realtime translation")
        print("2) Emergency mode (hardcoded emergency signs)")
        print("3) Eye assist mode (separate test mode)")
        print("4) Fast sign record mode (manual teach via T)")
        print("5) Manage taught signs (delete/modify)")
        print("A) Advanced menu (record/train/evaluate)")
        print("Q) Quit")

        choice = input("Select option (1-5, A, Q): ").strip().lower()

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
                mode="aid",
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
                mode="eye",
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
                mode="teach",
                voice=True,
            )
            do_run(args)
            continue

        if choice == "5":
            run_signs_menu()
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
        print("4) Record custom clips")
        print("5) Build + train CUSTOM (custom dataset -> custom model)")
        print("6) Build + train GLOBAL (external dataset -> global model)")
        print("7) Evaluate CUSTOM model")
        print("8) Evaluate GLOBAL model")
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
            args = argparse.Namespace(
                version=CUSTOM_DATASET,
                model_name=CUSTOM_MODEL,
                seq_len=DEFAULT_SEQ_LEN,
                algo="auto",
            )
            do_build(argparse.Namespace(version=CUSTOM_DATASET))
            do_train(args)
            continue

        if choice == "6":
            summary = build_global()
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

        if choice == "7":
            do_eval(argparse.Namespace(version=CUSTOM_DATASET, model_name=CUSTOM_MODEL))
            continue

        if choice == "8":
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
        auto_promote(model_name, new_acc)
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


def _list_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def _get_letter(path_obj):
    # Prefer directory names like A, B, C..., then fallback to filename stem.
    parts = [str(p).strip() for p in path_obj.parts]
    for part in reversed(parts):
        if len(part) == 1 and part.isalpha():
            return part.lower()

    stem = path_obj.stem.strip()
    if stem:
        ch = stem[0]
        if ch.isalpha():
            return ch.lower()
    return None


def _pick_hand(detector, frame):
    """Try multiple pre-processing variants and keep the best hand detection."""
    import cv2
    import numpy as np

    if frame is None:
        return None

    variants = [frame]
    h, w = frame.shape[:2]

    # Upscaling helps tiny ISL images where hand occupies very few pixels.
    variants.append(cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))
    variants.append(cv2.resize(frame, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC))

    # Contrast + sharpen variant for low-detail inputs.
    up3 = variants[-1]
    lab = cv2.cvtColor(up3, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    up3_clahe = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    sharp = cv2.GaussianBlur(up3_clahe, (0, 0), 1.2)
    up3_sharp = cv2.addWeighted(up3_clahe, 1.45, sharp, -0.45, 0)
    variants.append(np.clip(up3_sharp, 0, 255).astype(np.uint8))

    best = None
    best_score = -1.0
    for vf in variants:
        try:
            d = detector.process(vf)
        except Exception:
            continue
        if d is None:
            continue
        if d.left is None and d.right is None:
            continue

        q = getattr(d, "quality", {}) or {}
        area = float(q.get("hand_area", 0.0))
        blur = float(q.get("blur", 0.0))
        score = area + 0.0002 * blur
        if score > best_score:
            best = d
            best_score = score

    return best


def _balance_rows(rows, raw_dir, min_target_per_label=24):
    """Balance class counts by generating lightweight landmark-jitter variants."""
    import numpy as np

    by_label = {}
    for row in rows:
        by_label.setdefault(str(row.get("intent_id", "unknown")), []).append(row)

    rng = np.random.default_rng(42)
    out = list(rows)
    next_id = len(rows) + 1

    for label, items in sorted(by_label.items()):
        need = int(min_target_per_label) - len(items)
        if need <= 0:
            continue

        for _ in range(need):
            src = items[int(rng.integers(0, len(items)))]
            src_npz = Path(str(src["npz_path"]))
            if not src_npz.exists():
                continue

            try:
                blob = np.load(src_npz)
                seq = np.asarray(blob["sequence"], dtype=np.float32).copy()
                ts = np.asarray(blob["timestamps"], dtype=np.int64).copy()
            except Exception:
                continue

            # frame_to_vec layout: left(63), right(63), quality(3)
            if seq.ndim == 2 and seq.shape[1] >= 126:
                jitter = rng.normal(0.0, 0.003, size=(seq.shape[0], 126)).astype(np.float32)
                seq[:, :126] = seq[:, :126] + jitter

            clip_id = f"clip_{next_id:06d}"
            next_id += 1
            out_npz = raw_dir / f"{clip_id}.npz"
            np.savez_compressed(out_npz, sequence=seq, timestamps=ts)

            new_row = dict(src)
            new_row["clip_id"] = clip_id
            new_row["npz_path"] = str(out_npz)
            q = dict(new_row.get("quality", {}))
            q["augmented"] = True
            new_row["quality"] = q
            out.append(new_row)

    return out


def _split_by_label(rows, train_ratio=0.7, val_ratio=0.15):
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


def build_global():
    import cv2
    import numpy as np

    from core.hand_detection import HandCfg, HandDetector
    from dataset.recording import frame_to_vec

    images = _list_images(EXTERNAL_DATA_DIR)
    if not images:
        print("No images found in data/external")
        return None

    print("\n=== Global Dataset Build ===")
    print(f"Found external images: {len(images)}")
    print("Building A-Z landmark dataset from external images...")

    # relaxed thresholds preserve class coverage; quality is recovered by split + retraining.
    min_hand_area = 0.008
    min_blur = 20.0
    min_per_label = 6
    min_target_per_label = 24

    if GLOBAL_RAW_DIR.exists():
        shutil.rmtree(GLOBAL_RAW_DIR)
    GLOBAL_RAW_DIR.mkdir(parents=True, exist_ok=True)

    det = HandDetector(HandCfg(scale=1.0, min_det=0.45, min_track=0.45))
    rows = []
    kept = 0
    skipped = 0
    skipped_no_label = 0
    skipped_non_letter = 0
    try:
        for i, img_path in enumerate(images, start=1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                skipped += 1
                continue

            label = _get_letter(img_path)
            if label is None:
                skipped_no_label += 1
                continue
            if not (len(label) == 1 and label.isalpha()):
                skipped_non_letter += 1
                continue

            data = _pick_hand(det, frame)
            if data is None:
                skipped += 1
                continue
            if data.left is None and data.right is None:
                skipped += 1
                continue

            clip_id = f"clip_{kept + 1:06d}"
            npz_path = GLOBAL_RAW_DIR / f"{clip_id}.npz"

            seq = np.asarray([frame_to_vec(data)], dtype=np.float32)
            ts = np.asarray([data.ts_ms], dtype=np.int64)
            np.savez_compressed(npz_path, sequence=seq, timestamps=ts)

            # quality filter: keep usable hand samples without over-pruning rare letters
            q = dict(data.quality)
            if float(q.get("hand_area", 0.0)) < min_hand_area:
                skipped += 1
                continue
            if float(q.get("blur", 0.0)) < min_blur:
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
        if len(label_rows) >= min_per_label:
            filtered_rows.extend(label_rows)

    if not filtered_rows:
        print("No labels have enough high-quality samples after filtering.")
        return None

    # Balance labels so weak classes are not ignored by the model.
    filtered_rows = _balance_rows(
        filtered_rows,
        raw_dir=GLOBAL_RAW_DIR,
        min_target_per_label=min_target_per_label,
    )

    splits = _split_by_label(filtered_rows)
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
        "skipped_no_label": skipped_no_label,
        "skipped_non_letter": skipped_non_letter,
        "min_hand_area": min_hand_area,
        "min_blur": min_blur,
        "min_per_label": min_per_label,
        "min_target_per_label": min_target_per_label,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    present = sorted({str(r["intent_id"]).lower() for r in filtered_rows})
    target = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    missing = [x for x in target if x not in present]
    print("=== Global Dataset Ready ===")
    print(f"Usable samples: {summary['total_samples']}")
    print(f"Split: train={summary['train']} | val={summary['val']} | test={summary['test']}")
    print(f"Labels kept: {summary['labels']}")
    print(f"Skipped images: {summary['skipped_images']}")
    if missing:
        print(f"Missing letters after extraction: {', '.join(missing)}")
    else:
        print("All A-Z letters are present after extraction.")
    return summary


def do_promote(args):
    from model.model_manager import ModelHub

    model_name = args.model_name or GLOBAL_MODEL
    force = bool(getattr(args, "force", False))
    min_val_acc = float(getattr(args, "min_val_acc", 0.0))
    min_gain = float(getattr(args, "min_gain", 0.0))
    hub = ModelHub()
    active = hub.active()

    new_acc = read_val_acc(model_name)
    if new_acc is None and not force:
        print(f"Promote blocked: metadata missing for model '{model_name}'. Use --force to override.")
        return

    if new_acc is not None and new_acc < min_val_acc and not force:
        print(f"Promote blocked: val_accuracy {new_acc:.3f} < min required {min_val_acc:.3f}.")
        return

    if active and active != model_name and not force:
        old_acc = read_val_acc(active)
        if old_acc is not None and new_acc is not None:
            if (new_acc - old_acc) < min_gain:
                print(
                    f"Promote blocked: gain {new_acc - old_acc:.3f} < min gain {min_gain:.3f}. "
                    f"(active={active}:{old_acc:.3f}, new={model_name}:{new_acc:.3f})"
                )
                return

    out = hub.promote(model_name, notes=args.notes)
    print(out)


def do_rollback(args):
    from model.model_manager import ModelHub

    hub = ModelHub()
    out = hub.rollback(notes=str(getattr(args, "notes", "manual rollback")))
    print(out)


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


def auto_promote(model_name, new_acc, min_gain=0.01):
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

    ensure_venv_python()
    setup_layout()

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
        "rollback": do_rollback,
    }
    action = actions.get(args.cmd)
    if action is None:
        raise RuntimeError(f"Unknown command: {args.cmd}")
    action(args)


if __name__ == "__main__":
    main()
