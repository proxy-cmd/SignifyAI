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
        print("2) Normal realtime (balanced)")
        print("3) Realtime (speed profile for old PCs)")
        print("4) Realtime (accuracy profile)")
        print("5) Realtime (rules only)")
        print("6) Collect samples")
        print("7) Train model")
        print("8) Build session report")
        print("9) Run doctor checks")
        print("10) Bootstrap full ML pipeline (Kaggle -> Train)")
        print("11) Benchmark camera/tracker FPS")
        print("12) Run autonomous pipeline script")
        print("13) Build sequence dataset (from frame CSV)")
        print("14) Train temporal model")
        print("15) Run realtime (temporal mode)")
        print("16) Collect sequence clips")
        print("17) Build release bundle zip")
        print("18) Infer recorded video (offline)")
        print("19) Train production models (frame + temporal)")
        print("20) Exit")
        choice = input("Choose 1-20: ").strip()

        if choice == "1":
            run_cmd([str(SRC / "stage_demo.py")])
        elif choice == "2":
            run_cmd([str(SRC / "main.py"), "run", "--profile", "balanced", "--mode", "hybrid"])
        elif choice == "3":
            run_cmd([str(SRC / "main.py"), "run", "--profile", "speed", "--mode", "hybrid"])
        elif choice == "4":
            run_cmd([str(SRC / "main.py"), "run", "--profile", "accuracy", "--mode", "hybrid"])
        elif choice == "5":
            run_cmd([str(SRC / "main.py"), "run", "--mode", "rules"])
        elif choice == "6":
            label = input("Enter label (example: hello): ").strip().lower() or "hello"
            samples = input("How many samples? (default 250): ").strip() or "250"
            run_cmd([str(SRC / "main.py"), "collect", "--label", label, "--samples", samples])
        elif choice == "7":
            mode = (input("Train mode [normal/automl] (default automl): ").strip().lower() or "automl")
            if mode.startswith("n"):
                run_cmd([str(SRC / "main.py"), "train"])
            else:
                run_cmd([str(SRC / "main.py"), "train", "--automl"])
        elif choice == "8":
            run_cmd([str(SRC / "main.py"), "report"])
        elif choice == "9":
            run_cmd([str(SRC / "main.py"), "doctor"])
        elif choice == "10":
            run_cmd([str(SRC / "main.py"), "bootstrap-ml"])
        elif choice == "11":
            run_cmd([str(SRC / "main.py"), "benchmark"])
        elif choice == "12":
            subprocess.call(
                [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(ROOT / "scripts" / "autonomous_pipeline.ps1"),
                ],
                cwd=str(ROOT),
            )
        elif choice == "13":
            run_cmd([str(SRC / "main.py"), "build-seq-dataset"])
        elif choice == "14":
            run_cmd([str(SRC / "main.py"), "train-seq"])
        elif choice == "15":
            run_cmd([str(SRC / "main.py"), "run", "--mode", "temporal", "--profile", "balanced"])
        elif choice == "16":
            label = input("Enter sequence label (example: hello): ").strip().lower() or "hello"
            clips = input("How many clips? (default 80): ").strip() or "80"
            run_cmd([str(SRC / "main.py"), "collect-seq", "--label", label, "--clips", clips])
        elif choice == "17":
            run_cmd([str(SRC / "main.py"), "release-bundle"])
        elif choice == "18":
            video_path = input("Enter input video path: ").strip()
            if video_path:
                run_cmd([str(SRC / "main.py"), "infer-video", "--input", video_path])
            else:
                print("No video path provided.")
        elif choice == "19":
            run_cmd([str(SRC / "main.py"), "train-production"])
        elif choice == "20":
            print("Goodbye.")
            return
        else:
            print("Invalid choice. Please pick 1-20.")


if __name__ == "__main__":
    menu()
