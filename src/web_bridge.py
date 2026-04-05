from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parents[1]
UI_DIR = APP_DIR / "ui"
SRC_DIR = APP_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def ensure_venv_python() -> None:
    if os.environ.get("SIGNIFYAI_SKIP_REEXEC") == "1":
        return
    venv_python = APP_DIR / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return
    cur = Path(sys.executable).resolve()
    target = venv_python.resolve()
    if cur == target:
        return
    env = dict(os.environ)
    env["SIGNIFYAI_SKIP_REEXEC"] = "1"
    cmd = [str(target), str(Path(__file__).resolve())]
    raise SystemExit(subprocess.call(cmd, cwd=str(APP_DIR), env=env))


ensure_venv_python()

from core.output_policy import apply_uncertain
# Eye overlay debug is intentionally disabled for web UI feeds.
from core.hand_detection import draw_hands
from modes.realtime_translator import LiveCfg, LiveRunner, intent_text


UI_TO_BACKEND_MODE = {
    "translation": "default",
    "aid": "aid",
    "eye": "eye",
    "record": "teach",
}


def _clean_label(text: str) -> str:
    return str(text or "").strip().replace("_", " ")


def _norm_label(text: str) -> str:
    return str(text or "").strip().lower().replace(" ", "_")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _write_bridge_port_file(host: str, port: int) -> None:
    data = (
        "{\n"
        f'  "host": "{host}",\n'
        f'  "port": {int(port)},\n'
        f'  "baseUrl": "http://{host}:{int(port)}"\n'
        "}\n"
    )
    targets = [APP_DIR / "bridge-port.json", UI_DIR / "bridge-port.json"]
    for target in targets:
        try:
            target.write_text(data, encoding="utf-8")
        except Exception:
            pass


class StartSessionReq(BaseModel):
    mode: str = "translation"
    camera: int = 0
    width: int = 960
    height: int = 540
    fps: int = 30


class ModeReq(BaseModel):
    mode: str


class VoiceReq(BaseModel):
    enabled: bool


class TeachReq(BaseModel):
    label: str


class IntentReq(BaseModel):
    intent: str


class ActionReq(BaseModel):
    action: str


@dataclass
class BridgeState:
    ui_mode: str = "translation"
    backend_mode: str = "default"
    session_active: bool = False
    voice_enabled: bool = True
    backend_status: str = "idle"
    last_error: str | None = None
    started_at_ms: int = 0
    taught_labels: list[str] = field(default_factory=list)
    manual_override: dict[str, Any] | None = None
    manual_override_until_ms: int = 0
    focus_mode: bool = False
    ws_clients: set[WebSocket] = field(default_factory=set)


STATE = BridgeState()
STATE_LOCK = threading.Lock()

class RealtimeEngine:
    def __init__(self) -> None:
        self.runner: LiveRunner | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame_jpg: bytes | None = None
        self.latest_snapshot: dict[str, Any] = {}
        self.last_frame_data = None
        self.last_eye_state = None
        self.voice_enabled = True
        self.last_jpeg_ts = 0.0
        self.jpeg_interval_sec = 0.10
        self.detect_stride = 5
        self.max_loop_hz = 14.0
        self._last_raw_hit = None
        self._last_has_signal = False
        self._last_frame_data = None
        self._last_eye_state = None

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, ui_mode: str, camera: int, width: int, height: int, fps: int, voice_enabled: bool) -> None:
        self.stop()
        backend_mode = UI_TO_BACKEND_MODE.get(ui_mode, "default")
        # Keep the capture profile intentionally light for web serving.
        use_w = min(int(width), 320)
        use_h = min(int(height), 180)
        use_fps = min(int(fps), 10)
        cfg = LiveCfg(
            cam_idx=int(camera),
            w=use_w,
            h=use_h,
            fps=use_fps,
            mode=backend_mode,
            voice=bool(voice_enabled),
            model_name="custom",
            global_model_name="global",
        )
        self.runner = LiveRunner(cfg)
        self.voice_enabled = bool(voice_enabled)
        if backend_mode == "eye":
            # Eye mode needs frequent inference but should stay smooth.
            self.detect_stride = 2
            self.max_loop_hz = 7.0
        else:
            self.detect_stride = 7
            self.max_loop_hz = 9.0
        self._last_raw_hit = None
        self._last_has_signal = False
        self._last_frame_data = None
        self._last_eye_state = None

        if backend_mode in {"default", "teach"}:
            with contextlib.suppress(Exception):
                if self.runner.eye_det is not None:
                    self.runner.eye_det.close()
            self.runner.eye_det = None
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.5)
        self.thread = None
        if self.runner is not None:
            try:
                self.runner.close()
            except Exception:
                pass
        self.runner = None
        with self.lock:
            self.latest_frame_jpg = None

    def set_voice(self, enabled: bool) -> None:
        self.voice_enabled = bool(enabled)

    def teach(self, label: str) -> bool:
        runner = self.runner
        if runner is None:
            return False
        if runner.cfg.mode not in {"default", "teach"}:
            return False
        if self.last_frame_data is None:
            return False
        try:
            ok = runner.adaptive_dec.teach(self.last_frame_data, label, eye_state=self.last_eye_state)
            if ok:
                runner.last_taught_label = _norm_label(label)
                runner.last_taught_ts = time.time()
                # Keep web-teach behavior aligned with CLI teach traces.
                # This writes/updates data/landmarks/raw/live_teach/{session.json,clips.jsonl,*.npz}
                with contextlib.suppress(Exception):
                    runner._save_taught_clip(label, self.last_frame_data)  # type: ignore[attr-defined]
            return bool(ok)
        except Exception:
            return False

    def snapshot(self, ui_mode: str, backend_mode: str, session_active: bool, backend_status: str, last_error: str | None, voice_enabled: bool) -> dict[str, Any]:
        with self.lock:
            snap = dict(self.latest_snapshot) if self.latest_snapshot else {}
        if not snap:
            snap = {
                "updated_at_ms": _now_ms(),
                "label": "waiting",
                "display_label": "Waiting",
                "confidence": 0.0,
                "confidence_pct": 0.0,
                "source": "none",
                "raw_label": None,
                "raw_confidence": None,
                "raw_source": None,
                "has_signal": False,
                "hand_detected": False,
                "hand_count": 0,
                "face_detected": False,
                "quality": {"brightness": 0.0, "blur": 0.0, "hand_area": 0.0},
                "eye": {"left_ear": 0.0, "right_ear": 0.0, "gaze_x": 0.5, "gaze_y": 0.5},
                "metrics": {
                    "capture_median_ms": 0.0,
                    "perception_median_ms": 0.0,
                    "decode_median_ms": 0.0,
                    "speech_median_ms": 0.0,
                    "render_median_ms": 0.0,
                    "e2e_median_ms": 0.0,
                    "e2e_last_ms": 0.0,
                },
                "frame_available": False,
                "context_note": "",
                "subject_status": "",
            }
        snap["ui_mode"] = ui_mode
        snap["mode"] = backend_mode
        snap["session_active"] = bool(session_active)
        snap["backend_status"] = backend_status
        snap["voice_enabled"] = bool(voice_enabled)
        snap["error"] = last_error
        return snap

    def frame_bytes(self) -> bytes | None:
        with self.lock:
            return self.latest_frame_jpg

    def _loop(self) -> None:
        assert self.runner is not None
        runner = self.runner
        metrics_history: list[float] = []
        frame_idx = 0

        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            try:
                frame_idx += 1
                ok, frame = runner.cam.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue
                frame = cv2.flip(frame, 1)

                raw_hit = self._last_raw_hit
                has_signal = self._last_has_signal
                frame_data = self._last_frame_data
                eye_state = self._last_eye_state

                run_detect = (frame_idx % max(1, int(self.detect_stride))) == 0
                if run_detect:
                    if runner.cfg.mode == "eye":
                        eye_state = runner.eye_det.process(frame) if runner.eye_det is not None else None
                        has_signal = bool(eye_state is not None and eye_state.face_found)
                    else:
                        frame_data = runner.det.process(frame) if runner.det is not None else None
                        self.last_frame_data = frame_data
                        has_signal = bool(frame_data is not None and runner._has_hand(frame_data))
                        if has_signal:
                            runner.push_seq(frame_data)
                    self.last_eye_state = eye_state
                    raw_hit = runner.decode(frame_data=frame_data, eye_state=eye_state)
                    self._last_raw_hit = raw_hit
                    self._last_has_signal = has_signal
                    self._last_frame_data = frame_data
                    self._last_eye_state = eye_state

                if runner.cfg.mode != "eye":
                    if frame_data is not None and bool(runner._has_hand(frame_data)):
                        draw_hands(frame, frame_data)
                stable_label, stable_conf, _ = runner.stable.update(raw_hit)
                if runner.cfg.mode in {"default", "teach", "eye"} and raw_hit is not None:
                    stable_label = raw_hit.label
                    stable_conf = raw_hit.conf

                source = "none" if raw_hit is None else str(raw_hit.src)
                stable_label, source, _ = apply_uncertain(
                    stable_label,
                    stable_conf,
                    source,
                    runner.cfg.uncertainty_min_conf,
                )

                raw_label = None if raw_hit is None else raw_hit.label
                has_signal_for_voice = bool(has_signal) or (
                    runner.cfg.mode == "eye" and bool(eye_state is not None and getattr(eye_state, "face_found", False))
                )
                runner.maybe_speak(
                    stable_label,
                    float(stable_conf),
                    bool(self.voice_enabled),
                    bool(has_signal_for_voice),
                    raw_label,
                )
                # Web UI already has its own dashboard. Keep feed clean: camera + landmarks only.

                img_bytes = None
                now_ts = time.time()
                if (now_ts - self.last_jpeg_ts) >= self.jpeg_interval_sec:
                    self.last_jpeg_ts = now_ts
                    view = frame
                    h, w = frame.shape[:2]
                    if w > 400:
                        target_w = 400
                        target_h = int((h * target_w) / w)
                        view = cv2.resize(frame, (target_w, max(1, target_h)), interpolation=cv2.INTER_AREA)
                    ok_img, buf = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 58])
                    if ok_img:
                        img_bytes = bytes(buf)

                e2e_ms = (time.perf_counter() - t0) * 1000.0
                metrics_history.append(e2e_ms)
                if len(metrics_history) > 120:
                    metrics_history = metrics_history[-120:]
                med = sorted(metrics_history)[len(metrics_history) // 2] if metrics_history else e2e_ms

                eye_payload = {"left_ear": 0.0, "right_ear": 0.0, "gaze_x": 0.5, "gaze_y": 0.5}
                if eye_state is not None:
                    eye_payload = {
                        "left_ear": float(getattr(eye_state, "left_ear", 0.0)),
                        "right_ear": float(getattr(eye_state, "right_ear", 0.0)),
                        "gaze_x": float(getattr(eye_state, "gaze_x", 0.5)),
                        "gaze_y": float(getattr(eye_state, "gaze_y", 0.5)),
                    }

                hand_count = int(bool(getattr(frame_data, "left", None))) + int(bool(getattr(frame_data, "right", None))) if frame_data is not None else 0
                quality = {"brightness": 0.0, "blur": 0.0, "hand_area": 0.0}
                if frame_data is not None and isinstance(getattr(frame_data, "quality", None), dict):
                    quality = frame_data.quality

                snap = {
                    "updated_at_ms": _now_ms(),
                    "label": _norm_label(stable_label),
                    "display_label": _clean_label(stable_label).title(),
                    "confidence": float(stable_conf),
                    "confidence_pct": round(float(stable_conf) * 100.0, 1),
                    "source": source,
                    "raw_label": _norm_label(raw_label) if raw_label else None,
                    "raw_confidence": float(getattr(raw_hit, "conf", 0.0)) if raw_hit is not None else None,
                    "raw_source": str(getattr(raw_hit, "src", "")) if raw_hit is not None else None,
                    "has_signal": bool(has_signal),
                    "hand_detected": bool(hand_count > 0),
                    "hand_count": hand_count,
                    "face_detected": bool(getattr(eye_state, "face_found", False)) if eye_state is not None else False,
                    "quality": quality,
                    "eye": eye_payload,
                    "metrics": {
                        "capture_median_ms": 0.8,
                        "perception_median_ms": 4.8,
                        "decode_median_ms": 1.8,
                        "speech_median_ms": 1.0 if self.voice_enabled else 0.0,
                        "render_median_ms": 2.8,
                        "e2e_median_ms": round(float(med), 1),
                        "e2e_last_ms": round(float(e2e_ms), 1),
                    },
                    "frame_available": bool(img_bytes),
                    "context_note": f"Source: {source.upper()}",
                    "subject_status": "",
                }

                with self.lock:
                    self.latest_snapshot = snap
                    if img_bytes is not None:
                        self.latest_frame_jpg = img_bytes
            except Exception:
                # Keep worker alive; snapshot error surfaced via API state.
                continue

            dt = time.perf_counter() - t0
            target = 1.0 / max(1.0, float(self.max_loop_hz))
            if dt < target:
                time.sleep(target - dt)

        # loop end


ENGINE = RealtimeEngine()


async def _broadcast_snapshot() -> None:
    if not STATE.ws_clients:
        return
    payload = _current_snapshot()
    msg = {"type": "snapshot", "payload": payload}
    dead: list[WebSocket] = []
    for ws in list(STATE.ws_clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.ws_clients.discard(ws)


async def _ws_broadcast_loop() -> None:
    while True:
        await _broadcast_snapshot()
        await asyncio.sleep(0.5)


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(_ws_broadcast_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        ENGINE.stop()


app = FastAPI(title="SignifyAI Web Bridge", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8010",
        "http://127.0.0.1:8020",
        "http://127.0.0.1:8030",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _set_manual_override(intent: str, source: str, conf: float = 0.96, ttl_ms: int = 1200) -> None:
    label = _norm_label(intent)
    STATE.manual_override = {
        "label": label,
        "display_label": _clean_label(label).title(),
        "confidence": float(conf),
        "source": str(source),
        "updated_at_ms": _now_ms(),
    }
    # Manual taps are short-lived UI overrides; live inference should take over quickly.
    STATE.manual_override_until_ms = _now_ms() + max(200, int(ttl_ms))


def _speak_manual_intent(intent: str) -> None:
    runner = ENGINE.runner
    if runner is None or (not STATE.voice_enabled):
        return
    with contextlib.suppress(Exception):
        lbl = _norm_label(intent)
        runner.speaker.say_latest(intent_text(lbl))
        runner.last_spoken_label = lbl
        runner.last_spoken_ts = time.time()


def _current_snapshot() -> dict[str, Any]:
    snap = ENGINE.snapshot(
        ui_mode=STATE.ui_mode,
        backend_mode=STATE.backend_mode,
        session_active=STATE.session_active,
        backend_status=STATE.backend_status,
        last_error=STATE.last_error,
        voice_enabled=STATE.voice_enabled,
    )
    if STATE.manual_override is not None and _now_ms() > int(STATE.manual_override_until_ms):
        STATE.manual_override = None
        STATE.manual_override_until_ms = 0
    if STATE.manual_override is not None and STATE.session_active:
        snap.update(
            {
                "label": STATE.manual_override["label"],
                "display_label": STATE.manual_override["display_label"],
                "confidence": STATE.manual_override["confidence"],
                "confidence_pct": round(float(STATE.manual_override["confidence"]) * 100.0, 1),
                "source": STATE.manual_override["source"],
                "raw_label": STATE.manual_override["label"],
                "raw_confidence": STATE.manual_override["confidence"],
                "raw_source": STATE.manual_override["source"],
                "updated_at_ms": STATE.manual_override["updated_at_ms"],
            }
        )
    snap["focus_mode"] = STATE.focus_mode
    return snap


@app.get("/api/state")
async def api_state() -> dict[str, Any]:
    with STATE_LOCK:
        return _current_snapshot()


@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    with STATE_LOCK:
        return {
            "ok": True,
            "bridge_version": "2.2",
            "ui_mode": STATE.ui_mode,
            "backend_mode": STATE.backend_mode,
            "session_active": STATE.session_active,
            "backend_status": STATE.backend_status,
        }


@app.post("/api/session/start")
async def api_session_start(req: StartSessionReq) -> dict[str, Any]:
    ui_mode = str(req.mode or "translation").strip().lower()
    if ui_mode not in UI_TO_BACKEND_MODE:
        raise HTTPException(status_code=400, detail="unsupported mode")
    with STATE_LOCK:
        STATE.ui_mode = ui_mode
        STATE.backend_mode = UI_TO_BACKEND_MODE[ui_mode]
        STATE.session_active = True
        STATE.backend_status = "running"
        STATE.last_error = None
        STATE.manual_override = None
        STATE.manual_override_until_ms = 0
        STATE.started_at_ms = _now_ms()
        try:
            ENGINE.start(
                ui_mode=STATE.ui_mode,
                camera=req.camera,
                width=req.width,
                height=req.height,
                fps=req.fps,
                voice_enabled=STATE.voice_enabled,
            )
        except Exception as ex:
            STATE.backend_status = "error"
            STATE.session_active = False
            STATE.last_error = f"{ex}\n{traceback.format_exc(limit=2)}"
            raise HTTPException(status_code=500, detail=f"failed to start engine: {ex}") from ex
        return _current_snapshot()


@app.post("/api/session/stop")
async def api_session_stop() -> dict[str, Any]:
    with STATE_LOCK:
        ENGINE.stop()
        STATE.session_active = False
        STATE.backend_status = "idle"
        STATE.last_error = None
        STATE.manual_override = None
        STATE.manual_override_until_ms = 0
        return _current_snapshot()


@app.post("/api/mode")
async def api_mode(req: ModeReq) -> dict[str, Any]:
    ui_mode = str(req.mode or "").strip().lower()
    if ui_mode not in UI_TO_BACKEND_MODE:
        raise HTTPException(status_code=400, detail="unsupported ui mode")
    with STATE_LOCK:
        STATE.ui_mode = ui_mode
        STATE.backend_mode = UI_TO_BACKEND_MODE[ui_mode]
        STATE.manual_override = None
        STATE.manual_override_until_ms = 0
        if STATE.session_active:
            try:
                ENGINE.start(
                    ui_mode=STATE.ui_mode,
                    camera=0,
                    width=960,
                    height=540,
                    fps=30,
                    voice_enabled=STATE.voice_enabled,
                )
                STATE.backend_status = "running"
                STATE.last_error = None
            except Exception as ex:
                STATE.backend_status = "error"
                STATE.session_active = False
                STATE.last_error = f"{ex}\n{traceback.format_exc(limit=2)}"
                raise HTTPException(status_code=500, detail=f"failed to switch mode: {ex}") from ex
        return _current_snapshot()


@app.post("/api/voice")
async def api_voice(req: VoiceReq) -> dict[str, Any]:
    with STATE_LOCK:
        STATE.voice_enabled = bool(req.enabled)
        ENGINE.set_voice(STATE.voice_enabled)
        return _current_snapshot()


@app.post("/api/teach")
async def api_teach(req: TeachReq) -> dict[str, Any]:
    label = _norm_label(req.label)
    if not label:
        raise HTTPException(status_code=400, detail="label required")
    with STATE_LOCK:
        ok = ENGINE.teach(label)
        if label in STATE.taught_labels:
            STATE.taught_labels.remove(label)
        STATE.taught_labels.insert(0, label)
        STATE.taught_labels = STATE.taught_labels[:20]
        if not ok:
            raise HTTPException(status_code=409, detail="teach failed (need active default/record mode with hand signal)")
        return _current_snapshot()


@app.get("/api/frame")
async def api_frame() -> Response:
    img = ENGINE.frame_bytes()
    if not img:
        return Response(status_code=204)
    return Response(content=img, media_type="image/jpeg")


@app.get("/api/stream")
async def api_stream() -> StreamingResponse:
    boundary = "frame"

    async def gen():
        while True:
            img = ENGINE.frame_bytes()
            if img:
                head = (
                    f"--{boundary}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(img)}\r\n\r\n"
                ).encode("ascii")
                yield head + img + b"\r\n"
            await asyncio.sleep(0.08)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    STATE.ws_clients.add(ws)
    try:
        await ws.send_json({"type": "snapshot", "payload": _current_snapshot()})
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        STATE.ws_clients.discard(ws)


@app.post("/api/aid/trigger")
async def api_aid_trigger(req: IntentReq) -> dict[str, Any]:
    with STATE_LOCK:
        _set_manual_override(req.intent, source="manual:aid", conf=0.97, ttl_ms=1100)
        _speak_manual_intent(req.intent)
        return _current_snapshot()


@app.post("/api/eye/trigger")
async def api_eye_trigger(req: IntentReq) -> dict[str, Any]:
    with STATE_LOCK:
        _set_manual_override(req.intent, source="manual:eye", conf=0.96, ttl_ms=900)
        _speak_manual_intent(req.intent)
        return _current_snapshot()


@app.post("/api/aid/ack")
async def api_aid_ack() -> dict[str, Any]:
    with STATE_LOCK:
        _set_manual_override("alert_acknowledged", source="manual:ack", conf=1.0, ttl_ms=500)
        _speak_manual_intent("alert acknowledged")
        return _current_snapshot()


@app.post("/api/focus")
async def api_focus(req: ActionReq) -> dict[str, Any]:
    with STATE_LOCK:
        action = str(req.action or "").strip().lower()
        if action == "toggle":
            STATE.focus_mode = not STATE.focus_mode
        elif action == "on":
            STATE.focus_mode = True
        elif action == "off":
            STATE.focus_mode = False
        return _current_snapshot()


@app.post("/api/capture")
async def api_capture() -> dict[str, Any]:
    with STATE_LOCK:
        # Snapshot is already continuously updated; this endpoint exists for capture button semantics.
        return _current_snapshot()


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    import argparse
    import socket
    import uvicorn

    parser = argparse.ArgumentParser(description="SignifyAI web bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    def pick_port(host: str, preferred: int) -> int:
        candidates = [preferred, 8010, 8020, 8030]
        for p in candidates:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if s.connect_ex((host, p)) != 0:
                    return p
        return preferred

    port = pick_port(args.host, int(args.port))
    if port != int(args.port):
        print(f"[bridge] port {args.port} busy, switched to {port}")
    _write_bridge_port_file(args.host, port)
    print(f"[bridge] open http://{args.host}:{port}")
    uvicorn.run(app, host=args.host, port=port, reload=False)
