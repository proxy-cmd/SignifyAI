from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal


HardwareTier = Literal["low", "mid", "high"]


@dataclass
class HardwareInfo:
    cpu_cores: int
    ram_gb: float
    cpu_freq_ghz: float


@dataclass
class HardwarePreset:
    width: int
    height: int
    camera_fps: int
    infer_scale: float
    infer_interval: int
    smoothing_window: int
    target_fps: float
    enable_deep_runtime: bool
    tts_rate: int
    tts_min_gap_sec: float
    tts_dedup_sec: float


def detect_hardware_info() -> HardwareInfo:
    cores = int(os.cpu_count() or 4)
    ram_gb = 8.0
    cpu_freq_ghz = 2.4
    try:
        import psutil  # type: ignore

        ram_gb = float(psutil.virtual_memory().total) / (1024 ** 3)
        freq = psutil.cpu_freq()
        if freq is not None:
            if freq.max and freq.max > 0:
                cpu_freq_ghz = float(freq.max) / 1000.0
            elif freq.current and freq.current > 0:
                cpu_freq_ghz = float(freq.current) / 1000.0
    except Exception:
        pass
    return HardwareInfo(cpu_cores=cores, ram_gb=ram_gb, cpu_freq_ghz=cpu_freq_ghz)


def classify_hardware_tier(info: HardwareInfo) -> HardwareTier:
    low_signals = 0
    high_signals = 0

    if info.cpu_cores <= 4:
        low_signals += 1
    elif info.cpu_cores >= 10:
        high_signals += 1

    if info.ram_gb <= 8:
        low_signals += 1
    elif info.ram_gb >= 24:
        high_signals += 1

    if info.cpu_freq_ghz < 2.5:
        low_signals += 1
    elif info.cpu_freq_ghz >= 3.2:
        high_signals += 1

    if low_signals >= 2:
        return "low"
    if high_signals >= 2:
        return "high"
    return "mid"


def preset_for_tier(tier: HardwareTier) -> HardwarePreset:
    if tier == "low":
        return HardwarePreset(
            width=854,
            height=480,
            camera_fps=45,
            infer_scale=0.56,
            infer_interval=2,
            smoothing_window=5,
            target_fps=26.0,
            enable_deep_runtime=False,
            tts_rate=188,
            tts_min_gap_sec=0.12,
            tts_dedup_sec=0.28,
        )
    if tier == "high":
        return HardwarePreset(
            width=1280,
            height=720,
            camera_fps=60,
            infer_scale=0.78,
            infer_interval=1,
            smoothing_window=7,
            target_fps=42.0,
            enable_deep_runtime=True,
            tts_rate=180,
            tts_min_gap_sec=0.16,
            tts_dedup_sec=0.32,
        )
    return HardwarePreset(
        width=960,
        height=540,
        camera_fps=60,
        infer_scale=0.64,
        infer_interval=1,
        smoothing_window=6,
        target_fps=34.0,
        enable_deep_runtime=False,
        tts_rate=184,
        tts_min_gap_sec=0.14,
        tts_dedup_sec=0.30,
    )
