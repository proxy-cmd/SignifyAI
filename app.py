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
        print("2) Normal realtime (hybrid)")
        print("3) Realtime (rules only)")
        print("4) Collect samples")
        print("5) Train model")
        print("6) Exit")
        choice = input("Choose 1-6: ").strip()

        if choice == "1":
            run_cmd([str(SRC / "stage_demo.py")])
        elif choice == "2":
            run_cmd([str(SRC / "main.py"), "run", "--mode", "hybrid"])
        elif choice == "3":
            run_cmd([str(SRC / "main.py"), "run", "--mode", "rules"])
        elif choice == "4":
            label = input("Enter label (example: hello): ").strip().lower() or "hello"
            samples = input("How many samples? (default 250): ").strip() or "250"
            run_cmd([str(SRC / "main.py"), "collect", "--label", label, "--samples", samples])
        elif choice == "5":
            run_cmd([str(SRC / "main.py"), "train"])
        elif choice == "6":
            print("Goodbye.")
            return
        else:
            print("Invalid choice. Please pick 1-6.")


if __name__ == "__main__":
    menu()

