from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import List

import cv2


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_doctor(camera_index: int = 0, check_camera: bool = True) -> List[CheckResult]:
    results: list[CheckResult] = []

    results.append(
        CheckResult(
            name="python",
            ok=True,
            detail=f"{platform.python_version()} ({sys.executable})",
        )
    )

    try:
        import mediapipe as mp  # type: ignore

        has_solutions = hasattr(mp, "solutions")
        results.append(
            CheckResult(
                name="mediapipe",
                ok=bool(has_solutions),
                detail="import ok" if has_solutions else "imported but missing 'solutions'",
            )
        )
    except Exception as ex:
        results.append(CheckResult(name="mediapipe", ok=False, detail=str(ex)))

    try:
        _ = cv2.__version__
        results.append(CheckResult(name="opencv", ok=True, detail=f"cv2 {cv2.__version__}"))
    except Exception as ex:
        results.append(CheckResult(name="opencv", ok=False, detail=str(ex)))

    if check_camera:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        opened = cap.isOpened()
        if opened:
            ok, _ = cap.read()
            results.append(
                CheckResult(
                    name="camera",
                    ok=bool(ok),
                    detail=f"index={camera_index}, frame_read={'ok' if ok else 'failed'}",
                )
            )
        else:
            results.append(CheckResult(name="camera", ok=False, detail=f"index={camera_index}, failed to open"))
        cap.release()

    return results


def print_results(results: List[CheckResult]) -> int:
    failed = 0
    print("SignifyAI Doctor")
    print("-" * 60)
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"[{status:4}] {r.name:<12} {r.detail}")
        if not r.ok:
            failed += 1
    print("-" * 60)
    if failed == 0:
        print("Environment looks good.")
    else:
        print(f"{failed} check(s) failed.")
    return 0 if failed == 0 else 1

