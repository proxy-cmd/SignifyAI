from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BRIDGE = ROOT / "src" / "web_bridge.py"
WIZARD = ROOT / "ux" / "app.py"


def _python_exe() -> str:
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _wait_bridge_ready(timeout_sec: float = 60.0) -> str | None:
    candidates = [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8010",
        "http://127.0.0.1:8020",
        "http://127.0.0.1:8030",
    ]
    end = time.time() + timeout_sec
    while time.time() < end:
        for base in candidates:
            try:
                with urllib.request.urlopen(f"{base}/api/health", timeout=0.6) as res:
                    if res.status == 200:
                        return base
            except Exception:
                pass
        time.sleep(0.25)
    return None


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            time.sleep(0.6)
    except Exception:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except Exception:
            proc.kill()


def main() -> int:
    if not BRIDGE.exists() or not WIZARD.exists():
        print("Missing required files. Run from project root.")
        return 1

    py = _python_exe()
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    # If bridge is already running, reuse it.
    base = _wait_bridge_ready(timeout_sec=1.2)
    started_here = False
    bridge_proc: subprocess.Popen | None = None
    if base is None:
        bridge_proc = subprocess.Popen(
            [py, "-u", str(BRIDGE)],
            cwd=str(ROOT),
            creationflags=creationflags,
        )
        started_here = True
        print("Waiting for bridge to boot (up to 60s)...")
        base = _wait_bridge_ready(timeout_sec=60.0)
        if base is None:
            print("Bridge did not become ready in time.")
            if bridge_proc is not None:
                _terminate(bridge_proc)
            return 1

    print(f"Bridge ready at {base}. Starting wizard...")

    try:
        wizard_code = subprocess.call([py, str(WIZARD)], cwd=str(ROOT))
    finally:
        if started_here and bridge_proc is not None:
            _terminate(bridge_proc)

    return int(wizard_code)


if __name__ == "__main__":
    raise SystemExit(main())
