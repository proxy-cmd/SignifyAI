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
        print("5) Train model (AutoML)")
        print("6) Build session report")
        print("7) More tools (advanced)")
        print("8) Exit")
        choice = input("Choose 1-8: ").strip()

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
            run_cmd([str(SRC / "main.py"), "train", "--automl"])
        elif choice == "6":
            run_cmd([str(SRC / "main.py"), "report"])
        elif choice == "7":
            print("\nAdvanced tools (run these directly in terminal):")
            print("python -u .\\src\\main.py doctor")
            print("python -u .\\src\\main.py benchmark")
            print("python -u .\\src\\main.py collect-seq --label hello --clips 80")
            print("python -u .\\src\\main.py train-seq")
            print("python -u .\\src\\main.py run --mode temporal")
            print("python -u .\\src\\main.py train-production")
            print("python -u .\\src\\main.py infer-video --input .\\data\\raw\\demo.mp4")
            print("python -u .\\src\\main.py release-bundle")
            print("python -u .\\src\\main.py bootstrap-ml")
        elif choice == "8":
            print("Goodbye.")
            return
        else:
            print("Invalid choice. Please pick 1-8.")


if __name__ == "__main__":
    menu()
