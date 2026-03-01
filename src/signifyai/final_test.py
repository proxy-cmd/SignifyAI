from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from .config import DEFAULT_DATASET_PATH, PATHS
from .dataset_check import run_dataset_check
from .qa import QAConfig, run_validate_all


@dataclass
class FinalTestConfig:
    out_json: Path = PATHS.data_processed / "final_test_report.json"
    out_md: Path = PATHS.data_processed / "final_test_report.md"
    include_pytest: bool = True
    include_cli_help_checks: bool = True
    include_release_bundle_check: bool = True
    dataset_csv: Path = DEFAULT_DATASET_PATH
    min_samples_per_label: int = 5


def _build_report_markdown(payload: dict) -> str:
    summary = payload.get("summary", {})
    ds = payload.get("dataset_check", {})
    lines = [
        "# SignifyAI Final Test Report",
        "",
        f"- Timestamp: `{payload.get('timestamp', '-')}`",
        f"- Python: `{payload.get('python', '-')}`",
        "",
        "## Dataset Check",
        f"- Status: `{ds.get('status', 'SKIP')}`",
        f"- Detail: {ds.get('detail', '-')}",
        "",
        "## QA Summary",
        f"- Total checks: `{summary.get('total', 0)}`",
        f"- Passed: `{summary.get('passed', 0)}`",
        f"- Failed: `{summary.get('failed', 0)}`",
        f"- Skipped: `{summary.get('skipped', 0)}`",
        f"- Pass rate: `{summary.get('pass_rate', 0)}%`",
        "",
        "## Decision",
    ]
    if int(summary.get("failed", 0)) == 0:
        lines.append("- PASS: No failed checks.")
    else:
        lines.append("- FAIL: One or more checks failed.")
    return "\n".join(lines) + "\n"


def run_final_test(cfg: FinalTestConfig) -> tuple[Path, Path]:
    dataset_status = {"status": "SKIP", "detail": "dataset not found"}
    if cfg.dataset_csv.exists():
        ds = run_dataset_check(cfg.dataset_csv, min_samples_per_label=cfg.min_samples_per_label)
        dataset_status = {
            "status": "PASS" if ds.ok else "FAIL",
            "detail": ds.detail,
            "rows": ds.rows,
            "labels": ds.labels,
            "min_count": ds.min_count,
            "max_count": ds.max_count,
        }

    qa_json = run_validate_all(
        QAConfig(
            out_json=cfg.out_json,
            include_pytest=cfg.include_pytest,
            include_cli_help_checks=cfg.include_cli_help_checks,
            include_release_bundle_check=cfg.include_release_bundle_check,
        )
    )

    payload = json.loads(qa_json.read_text(encoding="utf-8"))
    payload["dataset_check"] = dataset_status
    payload["timestamp"] = datetime.now().isoformat(timespec="seconds")

    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_text = _build_report_markdown(payload)
    cfg.out_md.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_md.write_text(report_text, encoding="utf-8")
    return cfg.out_json, cfg.out_md

