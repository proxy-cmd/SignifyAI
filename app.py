from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def run_cmd(args: list[str]) -> int:
    cmd = [sys.executable, "-u", *args]
    print("\nRunning:", " ".join(cmd))
    print("-" * 72)
    return subprocess.call(cmd, cwd=str(ROOT))


def menu() -> None:
    while True:
        print("\n=== SignifyAI Launcher ===")
        print("1) Stage demo (recommended)")
        print("2) Normal realtime")
        print("3) Realtime (rules only)")
        print("4) Collect samples")
        print("5) Train model")
        print("6) Record custom sequence + phrase (easy)")
        print("7) Build session report")
        print("8) Open GUI app")
        print("9) Run final test gate")
        print("10) More tools (advanced)")
        print("11) Exit")
        choice = input("Choose 1-11: ").strip()

        if choice == "1":
            run_cmd([str(SRC / "stage_demo.py")])
        elif choice == "2":
            run_cmd([str(SRC / "main.py"), "run", "--profile", "balanced", "--mode", "hybrid"])
        elif choice == "3":
            run_cmd([str(SRC / "main.py"), "run", "--mode", "rules"])
        elif choice == "4":
            label = input("Enter label (example: hello): ").strip().lower() or "hello"
            samples = input("How many samples? (default 250): ").strip() or "250"
            run_cmd([str(SRC / "main.py"), "collect", "--label", label, "--samples", samples])
        elif choice == "5":
            print("Training mode: 1) AutoML  2) Deep (TensorFlow)  3) AutoML + Deep  4) Full pipeline (AutoML + Deep + Temporal)")
            tmode = input("Choose 1-4 (default 1): ").strip() or "1"
            if tmode == "2":
                run_cmd([str(SRC / "main.py"), "train-deep"])
            elif tmode == "3":
                run_cmd([str(SRC / "main.py"), "train", "--automl"])
                run_cmd([str(SRC / "main.py"), "train-deep"])
            elif tmode == "4":
                run_cmd([str(SRC / "main.py"), "train-all"])
            else:
                run_cmd([str(SRC / "main.py"), "train", "--automl"])
        elif choice == "6":
            label = input("Sequence label (example: watching_you): ").strip().lower().replace(" ", "_")
            text = input("Sentence to speak (example: I'm watching you): ").strip()
            clips = input("How many clips? (default 80): ").strip() or "80"
            if not label or not text:
                print("Label and sentence are required.")
            else:
                run_cmd(
                    [
                        str(SRC / "main.py"),
                        "record-combo",
                        "--label",
                        label,
                        "--text",
                        text,
                        "--clips",
                        clips,
                    ]
                )
        elif choice == "7":
            run_cmd([str(SRC / "main.py"), "report"])
        elif choice == "8":
            run_cmd([str(SRC / "gui.py")])
        elif choice == "9":
            run_cmd([str(SRC / "main.py"), "final-test"])
        elif choice == "10":
            print("\nAdvanced tools (run these directly in terminal):")
            print("python -u .\\src\\main.py calibrate --seconds 20")
            print("python -u .\\src\\main.py final-test")
            print("python -u .\\src\\main.py train-deep")
            print("python -u .\\src\\main.py train-all")
            print("python -u .\\src\\main.py teach-sign --label hello --phrase \"Hello\" --samples 180")
            print("python -u .\\src\\main.py check-dataset --dataset .\\data\\processed\\dataset.csv")
            print("python -u .\\src\\main.py run --profile ultra-speed")
            print("python -u .\\src\\main.py run --profile ultra-accuracy")
            print("python -u .\\src\\main.py doctor")
            print("python -u .\\src\\main.py benchmark")
            print("python -u .\\src\\main.py collect-seq --label hello --clips 80")
            print("python -u .\\src\\main.py record-combo --label watching_you --text \"I am watching you.\" --clips 80")
            print("python -u .\\src\\main.py image-points --image .\\path\\to\\image.jpg")
            print("python -u .\\src\\main.py adapt-sign --label hello --images .\\path\\to\\steps_folder")
            print("python -u .\\src\\main.py adapt-signs-folder --images-root .\\path\\to\\label_folders")
            print("python -u .\\src\\main.py train-seq")
            print("python -u .\\src\\main.py run --mode temporal")
            print("python -u .\\src\\main.py list-phrases")
            print("python -u .\\src\\main.py train-production")
            print("python -u .\\src\\main.py infer-video --input .\\data\\raw\\demo.mp4")
            print("python -u .\\src\\main.py release-bundle")
            print("python -u .\\src\\main.py bootstrap-ml")
        elif choice == "11":
            print("Goodbye.")
            return
        else:
            print("Invalid choice. Please pick 1-11.")


if __name__ == "__main__":
    menu()
