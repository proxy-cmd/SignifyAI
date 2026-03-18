import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external" / "kaggle"
MODELS_DIR = DATA_DIR / "models"
LIVE_TEACH_DIR = DATA_DIR / "landmarks" / "raw" / "live_teach"


def ensure_dirs_and_placeholders():
    (DATA_DIR / "exports").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_TEACH_DIR.mkdir(parents=True, exist_ok=True)

    proto = MODELS_DIR / "sign_prototypes.json"
    if not proto.exists():
        proto.write_text("{}\n", encoding="utf-8")

    gitkeep = LIVE_TEACH_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")


def has_kaggle_auth():
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def run_kaggle_download(dataset_id):
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_id,
        "-p",
        str(EXTERNAL_DIR),
        "--unzip",
    ]
    print(f"[data] Downloading: {dataset_id}")
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[warn] Failed: {dataset_id}")
        err = (res.stderr or res.stdout or "").strip()
        if err:
            print(err)
        return False
    print(f"[ok] Downloaded: {dataset_id}")
    return True


def validate_isl_hjy():
    isl_root = EXTERNAL_DIR / "indian-sign-language-dataset" / "ISL_Dataset"
    labels = ["H", "J", "Y"]
    if not isl_root.exists():
        print("[note] ISL dataset folder not found yet. If you use ISL, place it under data/external/kaggle/indian-sign-language-dataset/ISL_Dataset")
        return

    print("[data] Checking ISL letters H/J/Y...")
    for label in labels:
        p = isl_root / label
        if not p.exists():
            print(f"[warn] Missing folder: {p}")
            continue
        count = sum(1 for f in p.rglob("*") if f.is_file())
        print(f"[ok] {label}: {count} files")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap lightweight SignifyAI data setup")
    parser.add_argument("--download-kaggle", action="store_true", help="Download datasets via Kaggle CLI")
    parser.add_argument("--dataset", action="append", default=[], help="Kaggle dataset id, repeatable")
    args = parser.parse_args()

    ensure_dirs_and_placeholders()

    if args.download_kaggle:
        if not has_kaggle_auth():
            print("[warn] Kaggle auth not found. Set KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json first.")
            return 1
        if not args.dataset:
            print("[warn] No dataset ids provided. Use --dataset <owner/name>.")
            return 1
        for ds in args.dataset:
            run_kaggle_download(ds)

    validate_isl_hjy()
    print("[done] Data bootstrap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
