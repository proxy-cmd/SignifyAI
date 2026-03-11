from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shlex
import subprocess
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "webui"


@dataclass(frozen=True)
class Preset:
    id: str
    title: str
    description: str
    cmd: list[str]


PRESETS: list[Preset] = [
    Preset(
        id="run_balanced",
        title="Run Realtime (Balanced)",
        description="Main hybrid realtime mode.",
        cmd=["src/main.py", "run", "--profile", "balanced", "--mode", "hybrid"],
    ),
    Preset(
        id="run_stage",
        title="Run Stage Demo",
        description="Presentation mode with guided script.",
        cmd=["src/stage_demo.py"],
    ),
    Preset(
        id="run_enterprise",
        title="Run Enterprise Profile",
        description="Strict confidence and consensus profile.",
        cmd=["src/main.py", "run", "--profile", "enterprise"],
    ),
    Preset(
        id="doctor",
        title="Doctor Check",
        description="Environment and dependency diagnostics.",
        cmd=["src/main.py", "doctor"],
    ),
    Preset(
        id="data_help",
        title="Data Help",
        description="Human-friendly data ingestion help.",
        cmd=["src/main.py", "data-help"],
    ),
    Preset(
        id="train_all",
        title="Train All Models",
        description="AutoML + Deep + Temporal training pipeline.",
        cmd=["src/main.py", "train-all"],
    ),
    Preset(
        id="validate_all",
        title="Validate All",
        description="QA benchmark suite.",
        cmd=["src/main.py", "validate-all", "--no-release"],
    ),
    Preset(
        id="final_test",
        title="Final Test Gate",
        description="Final readiness report.",
        cmd=["src/main.py", "final-test", "--no-release"],
    ),
    Preset(
        id="model_report",
        title="Model Report",
        description="Generate frame/deep/temporal report.",
        cmd=["src/main.py", "model-report"],
    ),
]


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._running = False
        self._command: list[str] = []
        self._started_at = 0.0
        self._ended_at = 0.0
        self._exit_code: int | None = None
        self._log_seq = 0
        self._logs: deque[dict[str, Any]] = deque(maxlen=6000)

    def _append_log(self, line: str) -> None:
        with self._lock:
            self._log_seq += 1
            self._logs.append(
                {
                    "seq": self._log_seq,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "line": line.rstrip("\n"),
                }
            )

    def start(self, cmd: list[str]) -> tuple[bool, str]:
        with self._lock:
            if self._running:
                return False, "A command is already running. Stop it first."

            full_cmd = [os.sys.executable, "-u", *cmd]
            try:
                proc = subprocess.Popen(
                    full_cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as ex:
                return False, f"Failed to start command: {ex}"

            self._proc = proc
            self._running = True
            self._command = full_cmd
            self._started_at = time.time()
            self._ended_at = 0.0
            self._exit_code = None

        self._append_log(f"$ {' '.join(full_cmd)}")
        threading.Thread(target=self._reader_loop, daemon=True).start()
        threading.Thread(target=self._wait_loop, daemon=True).start()
        return True, "Started"

    def _reader_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                self._append_log(line)
        except Exception as ex:
            self._append_log(f"[log-reader-error] {ex}")

    def _wait_loop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        code = proc.wait()
        with self._lock:
            self._running = False
            self._ended_at = time.time()
            self._exit_code = int(code)
        self._append_log(f"[process-exit] code={code}")

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            proc = self._proc
            if proc is None or not self._running:
                return False, "No running command."
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            self._append_log("[process-stop] requested by user")
            return True, "Stopped"
        except Exception as ex:
            return False, f"Stop failed: {ex}"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "command": self._command,
                "started_at": self._started_at,
                "ended_at": self._ended_at,
                "exit_code": self._exit_code,
                "log_seq": self._log_seq,
            }

    def logs_since(self, seq: int, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            items = [x for x in self._logs if int(x["seq"]) > seq]
        if len(items) > limit:
            return items[-limit:]
        return items


JOB = JobManager()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def _catalog_payload() -> dict[str, Any]:
    return {
        "presets": [
            {
                "id": p.id,
                "title": p.title,
                "description": p.description,
                "cmd": p.cmd,
            }
            for p in PRESETS
        ],
        "hints": [
            "Use Presets for one-click flows.",
            "Use Advanced command to run any src/main.py subcommand.",
            "Only one process runs at a time in this UI.",
        ],
    }


def _build_metrics() -> dict[str, Any]:
    dataset = ROOT / "data" / "processed" / "dataset.csv"
    frame_model = ROOT / "models" / "gesture_model.joblib"
    deep_model = ROOT / "models" / "gesture_deep_model.keras"
    temporal_model = ROOT / "models" / "temporal_gesture_model.joblib"
    return {
        "dataset_exists": dataset.exists(),
        "dataset_size_mb": round(dataset.stat().st_size / (1024 * 1024), 2) if dataset.exists() else 0.0,
        "frame_model": frame_model.exists(),
        "deep_model": deep_model.exists(),
        "temporal_model": temporal_model.exists(),
    }


def _safe_main_args(raw: str) -> tuple[bool, list[str] | str]:
    try:
        parts = shlex.split(raw)
    except Exception as ex:
        return False, f"Invalid command args: {ex}"
    if not parts:
        return False, "No command provided."
    banned = {"&&", "||", ";", "|", ">", "<"}
    if any(tok in banned for tok in parts):
        return False, "Shell operators are not allowed."
    return True, parts


class UiHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/catalog":
            self._send_json({"ok": True, **_catalog_payload()})
            return
        if parsed.path == "/api/status":
            self._send_json({"ok": True, "status": JOB.status(), "metrics": _build_metrics()})
            return
        if parsed.path == "/api/logs":
            qs = parse_qs(parsed.query)
            try:
                since = int(qs.get("since", ["0"])[0])
            except Exception:
                since = 0
            logs = JOB.logs_since(since)
            self._send_json({"ok": True, "logs": logs})
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        if parsed.path == "/api/run":
            mode = str(payload.get("mode", "")).strip().lower()
            if mode == "preset":
                preset_id = str(payload.get("id", "")).strip()
                preset = next((p for p in PRESETS if p.id == preset_id), None)
                if preset is None:
                    self._send_json({"ok": False, "error": "Preset not found."}, status=400)
                    return
                ok, msg = JOB.start(preset.cmd)
                self._send_json({"ok": ok, "message": msg, "command": preset.cmd}, status=200 if ok else 409)
                return

            if mode == "main_args":
                raw = str(payload.get("args", "")).strip()
                ok_args, result = _safe_main_args(raw)
                if not ok_args:
                    self._send_json({"ok": False, "error": str(result)}, status=400)
                    return
                cmd = ["src/main.py", *result]  # type: ignore[list-item]
                ok, msg = JOB.start(cmd)
                self._send_json({"ok": ok, "message": msg, "command": cmd}, status=200 if ok else 409)
                return

            if mode == "script":
                script = str(payload.get("script", "")).strip()
                allowed = {
                    "app": ["app.py"],
                    "stage": ["src/stage_demo.py"],
                }
                cmd = allowed.get(script)
                if cmd is None:
                    self._send_json({"ok": False, "error": "Unknown script."}, status=400)
                    return
                ok, msg = JOB.start(cmd)
                self._send_json({"ok": ok, "message": msg, "command": cmd}, status=200 if ok else 409)
                return

            self._send_json({"ok": False, "error": "Unknown run mode."}, status=400)
            return

        if parsed.path == "/api/stop":
            ok, msg = JOB.stop()
            self._send_json({"ok": ok, "message": msg}, status=200 if ok else 409)
            return

        self._send_json({"ok": False, "error": "Not found."}, status=404)

    def log_message(self, format: str, *args) -> None:
        # Keep terminal output clean; process logs are shown in web terminal.
        return


def main() -> None:
    if not STATIC_DIR.exists():
        raise SystemExit(f"Missing UI directory: {STATIC_DIR}")

    host = "127.0.0.1"
    port = 8787
    server = ThreadingHTTPServer((host, port), UiHandler)
    url = f"http://{host}:{port}"

    print("SignifyAI Web UI")
    print(f"Serving: {STATIC_DIR}")
    print(f"Open: {url}")
    print("Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
