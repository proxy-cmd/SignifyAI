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


def _ask_label(prompt: str, default: str = "hello") -> str:
    return input(prompt).strip().lower().replace(" ", "_") or default


def _run_realtime_default() -> int:
    return run_cmd(
        [
            str(SRC / "main.py"),
            "run",
            "--profile",
            "lite",
            "--mini-runtime",
            "--mode",
            "hybrid",
            "--continuous-sentence",
            "--sentence-pause-sec",
            "0.55",
            "--sentence-append-cooldown",
            "0.12",
            "--min-stable-frames",
            "1",
            "--label-hold-sec",
            "0.06",
            "--tts-rate",
            "190",
            "--tts-min-gap-sec",
            "0.10",
            "--tts-dedup-sec",
            "0.20",
            "--prediction-interval",
            "2",
            "--sync-inference",
            "--no-deep-runtime",
            "--no-prototypes",
            "--no-enhance-frame",
        ]
    )


def _run_live_teach() -> None:
    label = _ask_label("Live teach label (example: hello): ")
    if not label:
        print("Label is required.")
        return
    run_cmd(
        [
            str(SRC / "main.py"),
            "run",
            "--profile",
            "balanced",
            "--mode",
            "hybrid",
            "--continuous-sentence",
            "--sentence-pause-sec",
            "0.55",
            "--sentence-append-cooldown",
            "0.12",
            "--prediction-interval",
            "2",
            "--sync-inference",
            "--no-prototypes",
            "--no-enhance-frame",
            "--teach-label",
            label,
            "--live-capture",
            "--live-auto-retrain",
            "--live-retrain-every",
            "40",
        ]
    )


def _collect_samples() -> None:
    label = _ask_label("Enter label (example: hello): ")
    samples = input("How many samples? (default 250): ").strip() or "250"
    run_cmd([str(SRC / "main.py"), "collect", "--label", label, "--samples", samples])


def _record_sentence_clip() -> None:
    label = _ask_label("Sentence label (example: watching_you): ", default="")
    text = input("Sentence to speak (example: I am watching you): ").strip()
    clips = input("How many clips? (default 80): ").strip() or "80"
    if not label or not text:
        print("Label and sentence are required.")
        return
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


def _train_models() -> None:
    print("Training:")
    print("1) Fast frame model")
    print("2) Full pipeline")
    mode = input("Choose 1-2 (default 1): ").strip() or "1"
    if mode == "2":
        run_cmd([str(SRC / "main.py"), "train-all"])
    else:
        run_cmd([str(SRC / "main.py"), "train", "--automl"])


def _maintenance() -> None:
    print("\nMaintenance:")
    print("1) Doctor check")
    print("2) Benchmark camera")
    print("3) Final test gate")
    print("4) Back")
    choice = input("Choose 1-4: ").strip()
    if choice == "1":
        run_cmd([str(SRC / "main.py"), "doctor"])
    elif choice == "2":
        run_cmd([str(SRC / "main.py"), "benchmark", "--seconds", "5"])
    elif choice == "3":
        run_cmd([str(SRC / "main.py"), "final-test"])


def menu() -> None:
    while True:
        print("\n=== SignifyAI ===")
        print("1) Start realtime translator")
        print("2) Live teach + learn new sign")
        print("3) Collect sign samples")
        print("4) Record sentence clip")
        print("5) Train models")
        print("6) Stage demo")
        print("7) Maintenance")
        print("8) Exit")
        choice = input("Choose 1-8: ").strip()

        if choice == "1":
            _run_realtime_default()
        elif choice == "2":
            _run_live_teach()
        elif choice == "3":
            _collect_samples()
        elif choice == "4":
            _record_sentence_clip()
        elif choice == "5":
            _train_models()
        elif choice == "6":
            run_cmd([str(SRC / "stage_demo.py")])
        elif choice == "7":
            _maintenance()
        elif choice == "8":
            print("Goodbye.")
            return
        else:
            print("Invalid choice. Please pick 1-8.")


if __name__ == "__main__":
    menu()
