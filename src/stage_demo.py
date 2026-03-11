"""One-click stage demo launcher."""

from signifyai.realtime import RealtimeConfig, run_realtime


if __name__ == "__main__":
    cfg = RealtimeConfig(
        mode="rules",            # robust for demos without trained model
        mini_runtime=True,
        stage_mode=False,
        demo_script=False,
        show_sentence=False,
        width=960,
        height=540,
        camera_fps=30,
        inference_interval=2,
        prediction_interval=2,
        inference_scale=0.60,
        confidence_threshold=0.58,
        smoothing_window=5,
        model_complexity=0,
        quality_gate=False,
        use_deep_model=False,
        use_prototypes=False,
        enhance_frame=False,
        auto_speak=True,
    )
    run_realtime(cfg)
