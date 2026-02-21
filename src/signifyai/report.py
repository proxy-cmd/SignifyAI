from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ReportConfig:
    log_path: Path
    out_path: Path


def build_session_report(cfg: ReportConfig) -> Path:
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cfg.log_path.exists():
        cfg.out_path.write_text(
            "# SignifyAI Session Report\n\nNo session log found yet.\n",
            encoding="utf-8",
        )
        return cfg.out_path

    df = pd.read_csv(cfg.log_path)
    if df.empty:
        cfg.out_path.write_text(
            "# SignifyAI Session Report\n\nSession log is empty.\n",
            encoding="utf-8",
        )
        return cfg.out_path

    total_events = int(len(df))
    avg_conf = float(df["confidence"].mean()) if "confidence" in df.columns else 0.0
    top_counts = df["label"].value_counts().head(10)

    lines: list[str] = []
    lines.append("# SignifyAI Session Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Total spoken events: **{total_events}**")
    lines.append(f"- Average confidence: **{avg_conf:.2f}**")
    lines.append("")
    lines.append("## Top detected labels")
    for label, count in top_counts.items():
        lines.append(f"- {label}: {int(count)}")
    lines.append("")

    # Last 10 timeline events for demo evidence.
    lines.append("## Last 10 events")
    tail = df.tail(10)
    lines.append("")
    lines.append("| Timestamp | Label | Confidence | Hands |")
    lines.append("|---|---:|---:|---:|")
    for _, row in tail.iterrows():
        ts = str(row.get("timestamp", ""))
        label = str(row.get("label", ""))
        conf = float(row.get("confidence", 0.0))
        hands = int(row.get("hand_count", 0))
        lines.append(f"| {ts} | {label} | {conf:.2f} | {hands} |")

    cfg.out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg.out_path

