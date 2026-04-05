from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


ROOT_DIR = Path(__file__).resolve().parents[1]
BRIDGE_PORT_FILE = ROOT_DIR / "bridge-port.json"
PROTOTYPES_FILE = ROOT_DIR / "data" / "models" / "sign_prototypes.json"


def now_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def label_to_key(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def key_to_label(key: str) -> str:
    return str(key).replace("_", " ").title()


@dataclass
class AppState:
    connected: bool = False
    base_url: str = ""
    poll_running: bool = False
    session_active: bool = False
    speech_enabled: bool = True
    active_mode: str = "realtime"
    current_intent: str = "Waiting"
    confidence: float = 0.0
    logs: list[str] = field(default_factory=list)
    taught_sign_keys: list[str] = field(default_factory=list)


class BridgeClient:
    def __init__(self) -> None:
        self.base_url = self._discover_base_url()

    def _discover_base_url(self) -> str:
        candidates: list[str] = []
        if BRIDGE_PORT_FILE.exists():
            try:
                data = json.loads(BRIDGE_PORT_FILE.read_text(encoding="utf-8"))
                base = str(data.get("baseUrl", "")).strip()
                if base:
                    candidates.append(base)
            except Exception:
                pass

        for port in (8000, 8010, 8020, 8030):
            candidates.append(f"http://127.0.0.1:{port}")

        seen = set()
        for base in candidates:
            if base in seen:
                continue
            seen.add(base)
            try:
                payload = self._request_json("GET", f"{base}/api/health", None, timeout=0.65)
                if payload.get("ok") is True:
                    return base
            except Exception:
                continue
        return ""

    @staticmethod
    def _request_json(method: str, url: str, data: dict | None, timeout: float = 1.2) -> dict:
        body = None
        headers = {"Content-Type": "application/json"}
        if data is not None:
            body = json.dumps(data).encode("utf-8")
        req = Request(url=url, method=method, data=body, headers=headers)
        with urlopen(req, timeout=timeout) as res:
            raw = res.read().decode("utf-8")
            if not raw.strip():
                return {}
            return json.loads(raw)

    @staticmethod
    def _request_bytes(method: str, url: str, timeout: float = 1.2) -> bytes | None:
        req = Request(url=url, method=method)
        with urlopen(req, timeout=timeout) as res:
            raw = res.read()
            return raw or None

    def connected(self) -> bool:
        return bool(self.base_url)

    def refresh_base(self) -> bool:
        self.base_url = self._discover_base_url()
        return bool(self.base_url)

    def get_state(self) -> dict:
        if not self.base_url:
            raise RuntimeError("Bridge not connected")
        return self._request_json("GET", f"{self.base_url}/api/state", None)

    def start_session(self, mode: str) -> dict:
        if not self.base_url:
            raise RuntimeError("Bridge not connected")
        return self._request_json("POST", f"{self.base_url}/api/session/start", {"mode": mode})

    def stop_session(self) -> dict:
        if not self.base_url:
            raise RuntimeError("Bridge not connected")
        return self._request_json("POST", f"{self.base_url}/api/session/stop", {})

    def set_mode(self, mode: str) -> dict:
        if not self.base_url:
            raise RuntimeError("Bridge not connected")
        return self._request_json("POST", f"{self.base_url}/api/mode", {"mode": mode})

    def set_voice(self, enabled: bool) -> dict:
        if not self.base_url:
            raise RuntimeError("Bridge not connected")
        return self._request_json("POST", f"{self.base_url}/api/voice", {"enabled": bool(enabled)})

    def teach(self, label: str) -> dict:
        if not self.base_url:
            raise RuntimeError("Bridge not connected")
        return self._request_json("POST", f"{self.base_url}/api/teach", {"label": label})

    def aid_trigger(self, intent: str) -> dict:
        if not self.base_url:
            raise RuntimeError("Bridge not connected")
        return self._request_json("POST", f"{self.base_url}/api/aid/trigger", {"intent": intent})

    def eye_trigger(self, intent: str) -> dict:
        if not self.base_url:
            raise RuntimeError("Bridge not connected")
        return self._request_json("POST", f"{self.base_url}/api/eye/trigger", {"intent": intent})

    def aid_ack(self) -> dict:
        if not self.base_url:
            raise RuntimeError("Bridge not connected")
        return self._request_json("POST", f"{self.base_url}/api/aid/ack", {})

    def capture(self) -> dict:
        if not self.base_url:
            raise RuntimeError("Bridge not connected")
        return self._request_json("POST", f"{self.base_url}/api/capture", {})

    def get_frame_bytes(self) -> bytes | None:
        if not self.base_url:
            return None
        try:
            return self._request_bytes("GET", f"{self.base_url}/api/frame", timeout=0.9)
        except HTTPError as ex:
            if ex.code == 204:
                return None
            raise


class SignifyWizard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SignifyAI - Desktop Wizard")
        self.geometry("1280x760")
        self.minsize(1120, 700)

        self.state_obj = AppState()
        self.client = BridgeClient()
        self.preview_photo = None
        self.preview_fetching = False

        self.mode_labels = {
            "realtime": "Realtime Translation",
            "aid": "Emergency Hand",
            "eye": "Eye Assist",
            "record": "Quick Record",
            "manage": "Manage Signs",
        }
        self.mode_to_api = {
            "realtime": "translation",
            "aid": "aid",
            "eye": "eye",
            "record": "record",
        }

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("Card.TFrame", background="#1d2335")
        self.style.configure("Heading.TLabel", font=("Segoe UI", 15, "bold"))
        self.style.configure("Body.TLabel", font=("Segoe UI", 10))

        self._load_taught_sign_keys()
        self._build_layout()
        self._switch_mode("realtime", sync_backend=False)
        self._connect_bridge()
        self._refresh_all()
        self._poll_loop()
        self._preview_loop()

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        topbar = ttk.Frame(root, padding=(8, 4))
        topbar.pack(fill="x")
        ttk.Label(topbar, text="SignifyAI", style="Heading.TLabel").pack(side="left")
        self.bridge_badge = ttk.Label(topbar, text="Bridge: Offline", style="Body.TLabel")
        self.bridge_badge.pack(side="left", padx=14)
        self.session_badge = ttk.Label(topbar, text="Session Paused", style="Body.TLabel")
        self.session_badge.pack(side="left", padx=10)
        self.speech_badge = ttk.Label(topbar, text="Speech Enabled", style="Body.TLabel")
        self.speech_badge.pack(side="left")

        controls = ttk.Frame(topbar)
        controls.pack(side="right")
        self.refresh_btn = ttk.Button(controls, text="Reconnect", command=self._connect_bridge)
        self.refresh_btn.pack(side="left", padx=4)
        self.start_btn = ttk.Button(controls, text="Start Session", command=self._start_session)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(controls, text="Stop Session", command=self._stop_session)
        self.stop_btn.pack(side="left", padx=4)
        self.speech_btn = ttk.Button(controls, text="Toggle Speech", command=self._toggle_speech)
        self.speech_btn.pack(side="left", padx=4)

        content = ttk.Frame(root)
        content.pack(fill="both", expand=True, pady=(6, 0))

        sidebar = ttk.Frame(content, width=230, padding=(8, 10))
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="Modes", style="Heading.TLabel").pack(anchor="w", pady=(0, 8))
        for mode_key, label in self.mode_labels.items():
            ttk.Button(sidebar, text=label, command=lambda m=mode_key: self._switch_mode(m)).pack(fill="x", pady=4)

        self.main_area = ttk.Frame(content, padding=(10, 8))
        self.main_area.pack(side="left", fill="both", expand=True)

        self.preview_card = ttk.Frame(self.main_area, style="Card.TFrame", padding=14)
        self.preview_card.pack(fill="both", expand=True)
        self.mode_title = ttk.Label(self.preview_card, text="", style="Heading.TLabel")
        self.mode_title.pack(anchor="w")
        self.preview_image = ttk.Label(self.preview_card)
        self.preview_image.pack(fill="both", expand=True, pady=(8, 4))
        self.preview_text = ttk.Label(self.preview_card, justify="center", font=("Consolas", 11))
        self.preview_text.pack(fill="x")

        right_col = ttk.Frame(self.main_area, padding=(10, 0))
        right_col.pack(fill="x", pady=(10, 0))
        info = ttk.Frame(right_col)
        info.pack(fill="x")
        self.intent_label = ttk.Label(info, text="Intent: Waiting", font=("Segoe UI", 12, "bold"))
        self.intent_label.pack(side="left")
        self.confidence_label = ttk.Label(info, text="Confidence: 0.0%", font=("Segoe UI", 11))
        self.confidence_label.pack(side="left", padx=20)
        self.confidence_bar = ttk.Progressbar(right_col, orient="horizontal", mode="determinate", maximum=100)
        self.confidence_bar.pack(fill="x", pady=(8, 10))

        panel_wrap = ttk.Frame(self.main_area)
        panel_wrap.pack(fill="both", expand=False)
        self.mode_panel = ttk.LabelFrame(panel_wrap, text="Mode Actions", padding=12)
        self.mode_panel.pack(side="left", fill="both", expand=True)
        log_frame = ttk.LabelFrame(panel_wrap, text="Recent Activity", padding=8)
        log_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.log_list = tk.Listbox(log_frame, height=10)
        self.log_list.pack(fill="both", expand=True)

    def _switch_mode(self, mode: str, sync_backend: bool = True) -> None:
        self.state_obj.active_mode = mode
        self._render_mode_panel()
        self._refresh_all()
        self._log(f"Switched to {self.mode_labels.get(mode, mode)} mode.")
        if sync_backend and mode in self.mode_to_api and self.state_obj.connected:
            self._safe_api(lambda: self.client.set_mode(self.mode_to_api[mode]), f"Mode switched to {self.mode_to_api[mode]}")

    def _render_mode_panel(self) -> None:
        for child in self.mode_panel.winfo_children():
            child.destroy()
        mode = self.state_obj.active_mode
        if mode == "realtime":
            self._build_realtime_panel()
        elif mode == "aid":
            self._build_aid_panel()
        elif mode == "eye":
            self._build_eye_panel()
        elif mode == "record":
            self._build_record_panel()
        elif mode == "manage":
            self._build_manage_panel()

    def _build_realtime_panel(self) -> None:
        ttk.Label(self.mode_panel, text="Teach a new sign label:", style="Body.TLabel").pack(anchor="w")
        self.rt_teach_entry = ttk.Entry(self.mode_panel)
        self.rt_teach_entry.pack(fill="x", pady=6)
        ttk.Button(self.mode_panel, text="Save Sign", command=self._teach_sign_from_realtime).pack(anchor="w")

    def _build_aid_panel(self) -> None:
        ttk.Label(self.mode_panel, text="Quick emergency intents:", style="Body.TLabel").pack(anchor="w", pady=(0, 6))
        quick = ["Need Water", "Need Food", "Call Family", "Hospital Help", "Emergency", "Yes", "No"]
        grid = ttk.Frame(self.mode_panel)
        grid.pack(fill="x")
        for i, label in enumerate(quick):
            ttk.Button(grid, text=label, command=lambda x=label: self._aid_trigger(x)).grid(
                row=i // 2, column=i % 2, padx=4, pady=4, sticky="ew"
            )
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        ttk.Button(self.mode_panel, text="Acknowledge Alert", command=self._aid_ack).pack(anchor="w", pady=8)

    def _build_eye_panel(self) -> None:
        ttk.Label(self.mode_panel, text="Eye assist controls:", style="Body.TLabel").pack(anchor="w")
        row = ttk.Frame(self.mode_panel)
        row.pack(fill="x", pady=6)
        ttk.Button(row, text="Blink -> YES", command=lambda: self._eye_trigger("Yes")).pack(side="left", padx=4)
        ttk.Button(row, text="Blink -> NO", command=lambda: self._eye_trigger("No")).pack(side="left", padx=4)
        ttk.Button(row, text="Emergency", command=lambda: self._eye_trigger("Emergency")).pack(side="left", padx=4)

    def _build_record_panel(self) -> None:
        ttk.Label(self.mode_panel, text="Quick record label:", style="Body.TLabel").pack(anchor="w")
        self.record_entry = ttk.Entry(self.mode_panel)
        self.record_entry.pack(fill="x", pady=6)
        row = ttk.Frame(self.mode_panel)
        row.pack(fill="x")
        ttk.Button(row, text="Start Capture", command=self._start_capture).pack(side="left", padx=4)
        ttk.Button(row, text="Stop + Save", command=self._stop_capture_and_save).pack(side="left", padx=4)

    def _build_manage_panel(self) -> None:
        self.manage_list = tk.Listbox(self.mode_panel, height=10)
        self.manage_list.pack(fill="both", expand=True, pady=(0, 8))
        for key in self.state_obj.taught_sign_keys:
            self.manage_list.insert("end", key_to_label(key))
        row = ttk.Frame(self.mode_panel)
        row.pack(fill="x")
        ttk.Button(row, text="Add", command=self._manage_add).pack(side="left", padx=4)
        ttk.Button(row, text="Modify", command=self._manage_modify).pack(side="left", padx=4)
        ttk.Button(row, text="Delete", command=self._manage_delete).pack(side="left", padx=4)

    def _connect_bridge(self) -> None:
        ok = self.client.refresh_base()
        self.state_obj.connected = ok
        self.state_obj.base_url = self.client.base_url
        if ok:
            self._log(f"Bridge connected: {self.client.base_url}")
            self._safe_api(self.client.get_state, "Initial state synced")
        else:
            self._log("Bridge not found. Start src/web_bridge.py first.")
        self._refresh_all()

    def _poll_loop(self) -> None:
        if self.state_obj.connected and (not self.state_obj.poll_running):
            self.state_obj.poll_running = True
            thread = threading.Thread(target=self._poll_worker, daemon=True)
            thread.start()
        self.after(700, self._poll_loop)

    def _preview_loop(self) -> None:
        if (not PIL_AVAILABLE) or (not self.state_obj.connected) or (not self.state_obj.session_active):
            self.preview_fetching = False
            self.after(220, self._preview_loop)
            return
        if self.preview_fetching:
            self.after(220, self._preview_loop)
            return
        self.preview_fetching = True
        thread = threading.Thread(target=self._preview_worker, daemon=True)
        thread.start()
        self.after(220, self._preview_loop)

    def _preview_worker(self) -> None:
        try:
            payload = self.client.get_frame_bytes()
            if not payload:
                self.after(0, self._preview_no_frame)
                return
            arr = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.after(0, self._preview_no_frame)
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            max_w = max(320, self.preview_card.winfo_width() - 28)
            max_h = max(220, self.preview_card.winfo_height() - 72)
            scale = min(max_w / float(w), max_h / float(h), 1.0)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            if (new_w, new_h) != (w, h):
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            image = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=image)
            self.after(0, lambda p=photo: self._apply_preview_photo(p))
        except Exception:
            self.after(0, self._preview_no_frame)
        finally:
            self.after(0, self._preview_done)

    def _preview_done(self) -> None:
        self.preview_fetching = False

    def _preview_no_frame(self) -> None:
        self.preview_image.configure(image="")
        self.preview_photo = None

    def _apply_preview_photo(self, photo) -> None:
        self.preview_photo = photo
        self.preview_image.configure(image=self.preview_photo)

    def _poll_worker(self) -> None:
        try:
            payload = self.client.get_state()
            self.after(0, lambda p=payload: self._apply_snapshot(p, from_poll=True))
        except Exception:
            self.after(0, self._mark_disconnected)
        finally:
            self.state_obj.poll_running = False

    def _mark_disconnected(self) -> None:
        if self.state_obj.connected:
            self._log("Bridge disconnected.")
        self.state_obj.connected = False
        self.state_obj.base_url = ""
        self._refresh_all()

    def _safe_api(self, fn, success_log: str | None = None) -> None:
        try:
            payload = fn()
            if isinstance(payload, dict) and payload:
                self._apply_snapshot(payload)
            if success_log:
                self._log(success_log)
        except (HTTPError, URLError, TimeoutError, RuntimeError) as ex:
            self._log(f"API error: {ex}")
            self._mark_disconnected()
        except Exception as ex:
            self._log(f"Unexpected error: {ex}")

    def _apply_snapshot(self, snap: dict, from_poll: bool = False) -> None:
        self.state_obj.connected = True
        self.state_obj.base_url = self.client.base_url
        self.state_obj.session_active = bool(snap.get("session_active", False))
        self.state_obj.speech_enabled = bool(snap.get("voice_enabled", True))
        label = str(snap.get("display_label", "Waiting")).strip() or "Waiting"
        self.state_obj.current_intent = label
        self.state_obj.confidence = float(snap.get("confidence_pct", 0.0)) / 100.0
        ui_mode = str(snap.get("ui_mode", "")).strip().lower()
        api_to_local = {"translation": "realtime", "aid": "aid", "eye": "eye", "record": "record"}
        if ui_mode in api_to_local and self.state_obj.active_mode != api_to_local[ui_mode]:
            self.state_obj.active_mode = api_to_local[ui_mode]
            self._render_mode_panel()
        if (not from_poll) and snap.get("error"):
            self._log(f"Backend error: {snap.get('error')}")
        self._refresh_all()

    def _start_session(self) -> None:
        if not self.state_obj.connected:
            self._connect_bridge()
            if not self.state_obj.connected:
                return
        api_mode = self.mode_to_api.get(self.state_obj.active_mode, "translation")
        self._safe_api(lambda: self.client.start_session(api_mode), f"Session started in {api_mode} mode")

    def _stop_session(self) -> None:
        self._safe_api(self.client.stop_session, "Session stopped")

    def _toggle_speech(self) -> None:
        next_state = not self.state_obj.speech_enabled
        self._safe_api(lambda: self.client.set_voice(next_state), f"Speech {'enabled' if next_state else 'muted'}")

    def _teach_sign_from_realtime(self) -> None:
        label = self.rt_teach_entry.get().strip()
        if not label:
            messagebox.showwarning("SignifyAI", "Enter a label first.")
            return
        self._safe_api(lambda: self.client.teach(label), f"Taught sign: {label}")
        self.rt_teach_entry.delete(0, "end")
        self._load_taught_sign_keys()
        if self.state_obj.active_mode == "manage":
            self._render_mode_panel()

    def _start_capture(self) -> None:
        self._safe_api(self.client.capture, "Capture trigger sent")

    def _stop_capture_and_save(self) -> None:
        label = self.record_entry.get().strip() if hasattr(self, "record_entry") else ""
        if not label:
            messagebox.showwarning("SignifyAI", "Enter a label before saving capture.")
            return
        self._safe_api(lambda: self.client.teach(label), f"Captured and saved: {label}")
        self.record_entry.delete(0, "end")
        self._load_taught_sign_keys()

    def _aid_trigger(self, label: str) -> None:
        self._safe_api(lambda: self.client.aid_trigger(label), f"AID intent sent: {label}")

    def _eye_trigger(self, label: str) -> None:
        self._safe_api(lambda: self.client.eye_trigger(label), f"EYE intent sent: {label}")

    def _aid_ack(self) -> None:
        self._safe_api(self.client.aid_ack, "Alert acknowledged")

    def _load_taught_sign_keys(self) -> None:
        if not PROTOTYPES_FILE.exists():
            self.state_obj.taught_sign_keys = []
            return
        try:
            raw = json.loads(PROTOTYPES_FILE.read_text(encoding="utf-8"))
            self.state_obj.taught_sign_keys = sorted([str(k) for k in raw.keys()])
        except Exception:
            self.state_obj.taught_sign_keys = []

    def _write_prototypes(self, payload: dict) -> None:
        PROTOTYPES_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROTOTYPES_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _manage_add(self) -> None:
        value = simpledialog.askstring("Add Sign", "Enter new sign label:")
        if not value:
            return
        label = value.strip()
        if not label:
            return
        if not self.state_obj.connected or not self.state_obj.session_active:
            messagebox.showinfo("SignifyAI", "Start session in Realtime/Record mode and use Teach to add real sign vectors.")
            return
        self._switch_mode("realtime")
        if hasattr(self, "rt_teach_entry"):
            self.rt_teach_entry.delete(0, "end")
            self.rt_teach_entry.insert(0, label)
        self._teach_sign_from_realtime()

    def _manage_modify(self) -> None:
        if not hasattr(self, "manage_list"):
            return
        idx = self.manage_list.curselection()
        if not idx:
            messagebox.showinfo("SignifyAI", "Select one sign first.")
            return
        old_key = self.state_obj.taught_sign_keys[idx[0]]
        old_label = key_to_label(old_key)
        value = simpledialog.askstring("Modify Sign", f"Change label:\n{old_label}", initialvalue=old_label)
        if not value:
            return
        new_key = label_to_key(value)
        if not new_key:
            return
        if not PROTOTYPES_FILE.exists():
            return
        try:
            raw = json.loads(PROTOTYPES_FILE.read_text(encoding="utf-8"))
            if old_key not in raw:
                return
            item = raw.pop(old_key)
            raw[new_key] = item
            self._write_prototypes(raw)
            self._log(f"Renamed taught sign: {old_key} -> {new_key}")
            self._load_taught_sign_keys()
            self._render_mode_panel()
        except Exception as ex:
            self._log(f"Modify failed: {ex}")

    def _manage_delete(self) -> None:
        if not hasattr(self, "manage_list"):
            return
        idx = self.manage_list.curselection()
        if not idx:
            messagebox.showinfo("SignifyAI", "Select one sign first.")
            return
        target_key = self.state_obj.taught_sign_keys[idx[0]]
        if not messagebox.askyesno("Delete Sign", f"Delete taught sign '{key_to_label(target_key)}'?"):
            return
        if not PROTOTYPES_FILE.exists():
            return
        try:
            raw = json.loads(PROTOTYPES_FILE.read_text(encoding="utf-8"))
            raw.pop(target_key, None)
            self._write_prototypes(raw)
            self._log(f"Deleted taught sign: {target_key}")
            self._load_taught_sign_keys()
            self._render_mode_panel()
        except Exception as ex:
            self._log(f"Delete failed: {ex}")

    def _log(self, text: str) -> None:
        item = f"{now_time()}  {text}"
        self.state_obj.logs.insert(0, item)
        self.state_obj.logs = self.state_obj.logs[:120]
        self._refresh_logs()

    def _refresh_logs(self) -> None:
        self.log_list.delete(0, "end")
        for item in self.state_obj.logs:
            self.log_list.insert("end", item)

    def _refresh_all(self) -> None:
        active = self.state_obj.session_active
        self.bridge_badge.configure(text=f"Bridge: {'Online' if self.state_obj.connected else 'Offline'}")
        self.session_badge.configure(text="Live Active" if active else "Session Paused")
        self.speech_badge.configure(text="Speech Enabled" if self.state_obj.speech_enabled else "Speech Muted")
        self.start_btn.configure(state="disabled" if active else "normal")
        self.stop_btn.configure(state="normal" if active else "disabled")

        title = self.mode_labels.get(self.state_obj.active_mode, "Mode")
        self.mode_title.configure(text=title)
        if (not PIL_AVAILABLE) or (not self.state_obj.connected) or (not active):
            self.preview_image.configure(image="")
            self.preview_photo = None
        preview_lines: list[str] = []
        if not PIL_AVAILABLE:
            preview_lines.append("Live camera preview requires Pillow: pip install pillow")
        elif not self.state_obj.connected:
            preview_lines.append("Bridge not connected. Start: python -u src/web_bridge.py")
        elif not active:
            preview_lines.append("Session paused. Start session to show live camera preview.")
        else:
            preview_lines.append("Live camera preview active.")
        preview_lines.extend(
            [
                f"Bridge: {self.state_obj.base_url or 'not connected'}",
                f"Mode: {title}",
                f"Session: {'Active' if active else 'Paused'}",
            ]
        )
        self.preview_text.configure(text="\n".join(preview_lines))

        self.intent_label.configure(text=f"Intent: {self.state_obj.current_intent}")
        conf_pct = self.state_obj.confidence * 100.0
        self.confidence_label.configure(text=f"Confidence: {conf_pct:.1f}%")
        self.confidence_bar["value"] = conf_pct
        self._refresh_logs()


def main() -> None:
    app = SignifyWizard()
    app.mainloop()


if __name__ == "__main__":
    main()
