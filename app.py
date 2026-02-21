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
        print("10) Exit")
        choice = input("Choose 1-10: ").strip()

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
            print("Goodbye.")
            return
        else:
            print("Invalid choice. Please pick 1-10.")


if __name__ == "__main__":
    menu()
