from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class SignifyGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SignifyAI Control Panel")
        self.root.geometry("1060x700")
        self.root.minsize(920, 620)

        self.proc: subprocess.Popen[str] | None = None
        self.out_q: queue.Queue[str] = queue.Queue()

        self._build_ui()
        self.root.after(120, self._drain_output)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Card.TFrame", relief="groove", borderwidth=1)

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="both", expand=True)

        ttk.Label(top, text="SignifyAI", style="Title.TLabel").pack(anchor="w")
        ttk.Label(top, text="Simple control panel for running all project actions").pack(anchor="w", pady=(0, 10))

        main = ttk.Frame(top)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(0, weight=1)

        left = ttk.Notebook(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        basic = ttk.Frame(left, padding=10)
        adv = ttk.Frame(left, padding=10)
        left.add(basic, text="Basic")
        left.add(adv, text="Advanced")

        log_card = ttk.Frame(main, style="Card.TFrame", padding=8)
        log_card.grid(row=0, column=1, sticky="nsew")
        ttk.Label(log_card, text="Console Output").pack(anchor="w")
        self.log = ScrolledText(log_card, wrap="word", font=("Consolas", 10), height=24)
        self.log.pack(fill="both", expand=True, pady=(4, 8))

        controls = ttk.Frame(log_card)
        controls.pack(fill="x")
        self.run_status = ttk.Label(controls, text="Idle")
        self.run_status.pack(side="left")
        self.pb = ttk.Progressbar(controls, mode="indeterminate", length=160)
        self.pb.pack(side="left", padx=8)
        ttk.Button(controls, text="Stop Running Command", command=self.stop_process).pack(side="right")

        self._build_basic_tab(basic)
        self._build_advanced_tab(adv)

    def _build_basic_tab(self, parent: ttk.Frame) -> None:
        block1 = ttk.LabelFrame(parent, text="Run", padding=8)
        block1.pack(fill="x", pady=(0, 8))
        ttk.Button(block1, text="Stage Demo", command=lambda: self.run_main(["stage_demo.py"])).pack(fill="x", pady=2)
        ttk.Button(
            block1,
            text="Normal Realtime",
            command=lambda: self.run_main(["main.py", "run", "--profile", "balanced", "--mode", "hybrid"]),
        ).pack(fill="x", pady=2)
        ttk.Button(
            block1,
            text="Rules Only Realtime",
            command=lambda: self.run_main(["main.py", "run", "--mode", "rules"]),
        ).pack(fill="x", pady=2)

        block2 = ttk.LabelFrame(parent, text="Collect + Train", padding=8)
        block2.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(block2)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Label:").pack(side="left")
        self.label_var = tk.StringVar(value="hello")
        ttk.Entry(row, textvariable=self.label_var, width=16).pack(side="left", padx=6)
        ttk.Label(row, text="Samples:").pack(side="left")
        self.samples_var = tk.StringVar(value="250")
        ttk.Entry(row, textvariable=self.samples_var, width=8).pack(side="left", padx=6)
        ttk.Button(block2, text="Collect Samples", command=self.collect_samples).pack(fill="x", pady=2)
        ttk.Button(block2, text="Train Model (AutoML)", command=lambda: self.run_main(["main.py", "train", "--automl"])).pack(fill="x", pady=2)
        ttk.Button(block2, text="Train Model (Deep TF)", command=lambda: self.run_main(["main.py", "train-deep"])).pack(fill="x", pady=2)
        ttk.Button(block2, text="Train All (AutoML + Deep + Temporal)", command=lambda: self.run_main(["main.py", "train-all"])).pack(fill="x", pady=2)

        block3 = ttk.LabelFrame(parent, text="Custom Sequence (Easy)", padding=8)
        block3.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(block3)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Sequence Label:").pack(side="left")
        self.seq_label_var = tk.StringVar(value="watching_you")
        ttk.Entry(row, textvariable=self.seq_label_var, width=20).pack(side="left", padx=6)
        row2 = ttk.Frame(block3)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Spoken Sentence:").pack(side="left")
        self.seq_text_var = tk.StringVar(value="I am watching you.")
        ttk.Entry(row2, textvariable=self.seq_text_var, width=28).pack(side="left", padx=6)
        row3 = ttk.Frame(block3)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="Clips:").pack(side="left")
        self.seq_clips_var = tk.StringVar(value="80")
        ttk.Entry(row3, textvariable=self.seq_clips_var, width=8).pack(side="left", padx=6)
        ttk.Button(block3, text="Record Custom Sequence", command=self.record_combo).pack(fill="x", pady=2)
        ttk.Button(block3, text="Run Temporal Mode", command=lambda: self.run_main(["main.py", "run", "--mode", "temporal"])).pack(fill="x", pady=2)

        block4 = ttk.LabelFrame(parent, text="Reports", padding=8)
        block4.pack(fill="x", pady=(0, 8))
        ttk.Button(block4, text="Build Session Report", command=lambda: self.run_main(["main.py", "report"])).pack(fill="x", pady=2)
        ttk.Button(block4, text="List Custom Phrases", command=lambda: self.run_main(["main.py", "list-phrases"])).pack(fill="x", pady=2)

    def _build_advanced_tab(self, parent: ttk.Frame) -> None:
        block = ttk.LabelFrame(parent, text="Tools", padding=8)
        block.pack(fill="x", pady=(0, 8))
        ttk.Button(block, text="Doctor Check", command=lambda: self.run_main(["main.py", "doctor"])).pack(fill="x", pady=2)
        ttk.Button(block, text="Calibrate Camera/User", command=lambda: self.run_main(["main.py", "calibrate"])).pack(fill="x", pady=2)
        ttk.Button(block, text="Check Dataset", command=lambda: self.run_main(["main.py", "check-dataset"])).pack(fill="x", pady=2)
        ttk.Button(block, text="Run Full QA Validation", command=lambda: self.run_main(["main.py", "validate-all"])).pack(fill="x", pady=2)
        ttk.Button(block, text="Run Final Test Gate", command=lambda: self.run_main(["main.py", "final-test"])).pack(fill="x", pady=2)
        ttk.Button(block, text="Benchmark", command=lambda: self.run_main(["main.py", "benchmark"])).pack(fill="x", pady=2)
        ttk.Button(block, text="Realtime Ultra Speed", command=lambda: self.run_main(["main.py", "run", "--profile", "ultra-speed"])).pack(fill="x", pady=2)
        ttk.Button(block, text="Realtime Ultra Accuracy", command=lambda: self.run_main(["main.py", "run", "--profile", "ultra-accuracy"])).pack(fill="x", pady=2)
        ttk.Button(block, text="Train Temporal", command=lambda: self.run_main(["main.py", "train-seq"])).pack(fill="x", pady=2)
        ttk.Button(block, text="Train Production", command=lambda: self.run_main(["main.py", "train-production"])).pack(fill="x", pady=2)
        ttk.Button(block, text="Build Release Bundle", command=lambda: self.run_main(["main.py", "release-bundle"])).pack(fill="x", pady=2)
        ttk.Button(block, text="Bootstrap ML (Kaggle)", command=lambda: self.run_main(["main.py", "bootstrap-ml"])).pack(fill="x", pady=2)

        block2 = ttk.LabelFrame(parent, text="Offline Video", padding=8)
        block2.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(block2)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Video path:").pack(side="left")
        self.video_path_var = tk.StringVar(value=str(ROOT / "data" / "raw" / "demo.mp4"))
        ttk.Entry(row, textvariable=self.video_path_var, width=40).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(block2, text="Run Video Inference", command=self.infer_video).pack(fill="x", pady=2)

        block3 = ttk.LabelFrame(parent, text="Image Adapt (Auto Learn)", padding=8)
        block3.pack(fill="x", pady=(0, 8))
        row_a = ttk.Frame(block3)
        row_a.pack(fill="x", pady=2)
        ttk.Label(row_a, text="Image/Folder:").pack(side="left")
        self.adapt_path_var = tk.StringVar(value=str(ROOT / "data" / "raw" / "images"))
        ttk.Entry(row_a, textvariable=self.adapt_path_var, width=40).pack(side="left", padx=6, fill="x", expand=True)
        row_b = ttk.Frame(block3)
        row_b.pack(fill="x", pady=2)
        ttk.Label(row_b, text="Label (single):").pack(side="left")
        self.adapt_label_var = tk.StringVar(value="custom_sign")
        ttk.Entry(row_b, textvariable=self.adapt_label_var, width=16).pack(side="left", padx=6)
        ttk.Button(block3, text="Read Image Points", command=self.image_points).pack(fill="x", pady=2)
        ttk.Button(block3, text="Adapt One Label", command=self.adapt_one).pack(fill="x", pady=2)
        ttk.Button(block3, text="Adapt Label Folders", command=self.adapt_folder).pack(fill="x", pady=2)

    def run_main(self, args: list[str]) -> None:
        target = SRC / args[0]
        cmd = [sys.executable, "-u", str(target), *args[1:]]
        self._run_process(cmd)

    def collect_samples(self) -> None:
        label = self.label_var.get().strip().lower().replace(" ", "_")
        samples = self.samples_var.get().strip() or "250"
        if not label:
            messagebox.showerror("Input error", "Label is required.")
            return
        self.run_main(["main.py", "collect", "--label", label, "--samples", samples])

    def record_combo(self) -> None:
        label = self.seq_label_var.get().strip().lower().replace(" ", "_")
        text = self.seq_text_var.get().strip()
        clips = self.seq_clips_var.get().strip() or "80"
        if not label or not text:
            messagebox.showerror("Input error", "Sequence label and sentence are required.")
            return
        self.run_main(
            [
                "main.py",
                "record-combo",
                "--label",
                label,
                "--text",
                text,
                "--clips",
                clips,
            ]
        )

    def infer_video(self) -> None:
        video = self.video_path_var.get().strip()
        if not video:
            messagebox.showerror("Input error", "Video path is required.")
            return
        self.run_main(["main.py", "infer-video", "--input", video])

    def image_points(self) -> None:
        path = self.adapt_path_var.get().strip()
        if not path:
            messagebox.showerror("Input error", "Image path is required.")
            return
        self.run_main(["main.py", "image-points", "--image", path])

    def adapt_one(self) -> None:
        path = self.adapt_path_var.get().strip()
        label = self.adapt_label_var.get().strip().lower().replace(" ", "_")
        if not path or not label:
            messagebox.showerror("Input error", "Path and label are required.")
            return
        self.run_main(["main.py", "adapt-sign", "--label", label, "--images", path])

    def adapt_folder(self) -> None:
        path = self.adapt_path_var.get().strip()
        if not path:
            messagebox.showerror("Input error", "Folder path is required.")
            return
        self.run_main(["main.py", "adapt-signs-folder", "--images-root", path, "--max-per-label", "120"])

    def _run_process(self, cmd: list[str]) -> None:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Running", "A command is already running. Stop it first.")
            return
        self.log.insert("end", "\nRunning: " + " ".join(cmd) + "\n" + "-" * 80 + "\n")
        self.log.see("end")
        self.run_status.configure(text="Running...")
        self.pb.start(10)

        def worker() -> None:
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    self.out_q.put(line)
                code = self.proc.wait()
                self.out_q.put(f"\n[Process finished with code {code}]\n")
            except Exception as ex:
                self.out_q.put(f"\n[Error] {ex}\n")
            finally:
                self.proc = None
                self.out_q.put("__PROC_DONE__")

        threading.Thread(target=worker, daemon=True).start()

    def _drain_output(self) -> None:
        try:
            while True:
                msg = self.out_q.get_nowait()
                if msg == "__PROC_DONE__":
                    self.run_status.configure(text="Idle")
                    self.pb.stop()
                    continue
                self.log.insert("end", msg)
                self.log.see("end")
        except queue.Empty:
            pass
        self.root.after(120, self._drain_output)

    def stop_process(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.log.insert("end", "\n[Stopped running process]\n")
            self.log.see("end")

    def _on_close(self) -> None:
        self.stop_process()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SignifyGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
