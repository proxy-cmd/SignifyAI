"""Backward-compatible wrapper for live sign-to-speech mode."""

from signifyai.realtime import RealtimeConfig, run_realtime


if __name__ == "__main__":
    run_realtime(RealtimeConfig())
