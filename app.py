from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def run_cmd(args: list[str]) -> int:
    cmd = [sys.executable, "-u", *args]
    print("\nRunning:", " ".join(cmd))
    print("-" * 88)
    return subprocess.call(cmd, cwd=str(ROOT))


def menu() -> None:
    while True:
        print("\n=== SignifyAI ===")
        print("1) Realtime translator")
        print("2) Record intent clips")
        print("3) Build dataset version")
        print("4) Train sequence model")
        print("5) Evaluate model")
        print("6) Promote model")
        print("7) Serve API + Web")
        print("8) Exit")
        choice = input("Choose 1-8: ").strip()

        if choice == "1":
            run_cmd([str(SRC / "main.py"), "run"])
        elif choice == "2":
            intent = input("Intent id (example: hospital_help): ").strip() or "hospital_help"
            clips = input("Clips (default 8): ").strip() or "8"
            run_cmd([str(SRC / "main.py"), "record", "--intent", intent, "--clips", clips])
        elif choice == "3":
            version = input("Dataset version name (default v1): ").strip() or "v1"
            run_cmd([str(SRC / "main.py"), "build-dataset", "--version", version])
        elif choice == "4":
            version = input("Dataset version (default v1): ").strip() or "v1"
            model_name = input("Model name (default isl_intent_v1): ").strip() or "isl_intent_v1"
            run_cmd([str(SRC / "main.py"), "train-seq", "--version", version, "--model-name", model_name])
        elif choice == "5":
            version = input("Dataset version (default v1): ").strip() or "v1"
            model_name = input("Model name (default isl_intent_v1): ").strip() or "isl_intent_v1"
            run_cmd([str(SRC / "main.py"), "evaluate", "--version", version, "--model-name", model_name])
        elif choice == "6":
            model_name = input("Model name (default isl_intent_v1): ").strip() or "isl_intent_v1"
            run_cmd([str(SRC / "main.py"), "promote", "--model-name", model_name])
        elif choice == "7":
            run_cmd([str(SRC / "main.py"), "serve-api", "--port", "8000"])
        elif choice == "8":
            print("Goodbye.")
            return
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    menu()
