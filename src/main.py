from __future__ import annotations

import argparse

from signifyai.realtime import RealtimeConfig, run_realtime


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SignifyAI rebuild runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run realtime hardcoded sign recognition")
    p_run.add_argument("--camera", type=int, default=0)
    p_run.add_argument("--width", type=int, default=960)
    p_run.add_argument("--height", type=int, default=540)
    p_run.add_argument("--fps", type=int, default=30)
    p_run.add_argument("--voice", dest="voice", action="store_true")
    p_run.add_argument("--no-voice", dest="voice", action="store_false")
    p_run.set_defaults(voice=True)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        run_realtime(
            RealtimeConfig(
                camera_index=args.camera,
                width=args.width,
                height=args.height,
                camera_fps=args.fps,
                voice_enabled=args.voice,
            )
        )


if __name__ == "__main__":
    main()
