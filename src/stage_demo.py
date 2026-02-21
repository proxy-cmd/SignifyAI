"""One-click stage demo launcher."""

from signifyai.realtime import RealtimeConfig, run_realtime


if __name__ == "__main__":
    cfg = RealtimeConfig(
        mode="rules",            # robust for demos without trained model
        stage_mode=True,
        demo_script=True,
        show_sentence=False,
        inference_interval=1,
        inference_scale=0.75,
        confidence_threshold=0.62,
        smoothing_window=7,
    )
    run_realtime(cfg)
