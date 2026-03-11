from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
MAIN = SRC / "main.py"


def run_py(args: list[str]) -> int:
    cmd = [sys.executable, "-u", str(MAIN), *args]
    print("\nRunning:", " ".join(str(x) for x in cmd))
    print("-" * 88)
    return subprocess.call(cmd, cwd=str(ROOT))


def ask(prompt: str, default: str) -> str:
    value = input(prompt).strip()
    return value or default


def run_live() -> None:
    run_py(["run"])


def run_record() -> None:
    intent = ask("Intent id (default hospital_help): ", "hospital_help")
    clips = ask("Clips (default 8): ", "8")
    run_py(["record", "--intent", intent, "--clips", clips])


def run_build_ds() -> None:
    version = ask("Dataset version (default v1): ", "v1")
    run_py(["build-dataset", "--version", version])


def run_train() -> None:
    version = ask("Dataset version (default v1): ", "v1")
    model = ask("Model name (default isl_intent_v1): ", "isl_intent_v1")
    run_py(["train-seq", "--version", version, "--model-name", model])


def run_eval() -> None:
    version = ask("Dataset version (default v1): ", "v1")
    model = ask("Model name (default isl_intent_v1): ", "isl_intent_v1")
    run_py(["evaluate", "--version", version, "--model-name", model])


def run_promote() -> None:
    model = ask("Model name (default isl_intent_v1): ", "isl_intent_v1")
    run_py(["promote", "--model-name", model])


def run_api() -> None:
    run_py(["serve-api", "--port", "8000"])


def menu() -> None:
    actions = {
        "1": run_live,
        "2": run_record,
        "3": run_build_ds,
        "4": run_train,
        "5": run_eval,
        "6": run_promote,
        "7": run_api,
    }

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

        if choice == "8":
            print("Goodbye.")
            return
        action = actions.get(choice)
        if action is None:
            print("Invalid choice.")
            continue
        action()


if __name__ == "__main__":
    menu()
