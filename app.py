from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
MAIN = SRC / "main.py"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
GLOBAL_MODEL = "signifyai_global"


def run_py(args: list[str]) -> int:
    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    cmd = [py, "-u", str(MAIN), *args]
    print("\nRunning:", " ".join(str(x) for x in cmd))
    print("-" * 88)
    return subprocess.call(cmd, cwd=str(ROOT))


def ask(prompt: str, default: str) -> str:
    value = input(prompt).strip()
    return value or default


def normalize_intent_id(text: str) -> str:
    text = text.strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    intent = "".join(out)
    while "__" in intent:
        intent = intent.replace("__", "_")
    return intent.strip("_") or "custom_intent"


def run_live() -> None:
    run_py(["run", "--mode", "hybrid"])


def run_demo() -> None:
    run_py(["run", "--mode", "demo"])


def run_aid() -> None:
    run_py(["run", "--mode", "aid"])


def run_record() -> None:
    raw_intent = ask("Intent text/id (default hospital_help): ", "hospital_help")
    intent = normalize_intent_id(raw_intent)
    signer = ask("Signer id (default signer_1): ", "signer_1")
    clips = ask("Clips (default 8): ", "8")
    run_py(["record", "--intent", intent, "--signer", signer, "--clips", clips])


def run_build_ds() -> None:
    version = ask("Dataset version (default v1): ", "v1")
    run_py(["build-dataset", "--version", version])


def run_train() -> None:
    version = ask("Dataset version (default v1): ", "v1")
    seq_len = ask("Sequence length (default 24): ", "24")
    run_py(["train-seq", "--version", version, "--model-name", GLOBAL_MODEL, "--seq-len", seq_len])


def run_eval() -> None:
    version = ask("Dataset version (default v1): ", "v1")
    run_py(["evaluate", "--version", version, "--model-name", GLOBAL_MODEL])


def run_promote() -> None:
    run_py(["promote", "--model-name", GLOBAL_MODEL])


def run_api() -> None:
    run_py(["serve-api", "--port", "8000"])


def menu() -> None:
    actions = {
        "1": run_live,
        "2": run_demo,
        "3": run_aid,
        "4": run_record,
        "5": run_build_ds,
        "6": run_train,
        "7": run_eval,
        "8": run_promote,
        "9": run_api,
    }

    while True:
        print("\n=== SignifyAI ===")
        print("1) Realtime translator (hybrid: demo + model)")
        print("2) Demo use (limited hardcoded signs)")
        print("3) Quick aid mode (simple emergency signs + side guide)")
        print("4) Record intent clips")
        print("5) Build dataset version")
        print("6) Train sequence model")
        print("7) Evaluate model")
        print("8) Promote global model")
        print("9) Serve API + Web")
        print("10) Exit")
        choice = input("Choose 1-10: ").strip()

        if choice == "10":
            print("Goodbye.")
            return
        action = actions.get(choice)
        if action is None:
            print("Invalid choice.")
            continue
        action()


if __name__ == "__main__":
    menu()
