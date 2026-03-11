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
        print("\n=== SignifyAI (Rebuild) ===")
        print("1) Start realtime sign recognizer")
        print("2) Exit")
        choice = input("Choose 1-2: ").strip()

        if choice == "1":
            run_cmd([str(SRC / "main.py"), "run"])
        elif choice == "2":
            print("Goodbye.")
            return
        else:
            print("Invalid choice. Please pick 1-2.")


if __name__ == "__main__":
    menu()
