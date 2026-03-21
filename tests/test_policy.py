from core.output_policy import apply_uncertain, should_speak


def test_keep_unknown():
    label, source, uncertain = apply_uncertain("unknown", 0.20, "none", 0.58)
    assert label == "unknown"
    assert source == "none"
    assert uncertain is False


def test_mark_low_conf():
    label, source, uncertain = apply_uncertain("yes", 0.31, "adaptive", 0.58)
    assert label == "uncertain"
    assert source == "adaptive+uncertain"
    assert uncertain is True


def test_keep_confident():
    label, source, uncertain = apply_uncertain("yes", 0.91, "adaptive", 0.58)
    assert label == "yes"
    assert source == "adaptive"
    assert uncertain is False


def test_reject_uncertain():
    assert (
        should_speak(
            label="uncertain",
            raw_label="yes",
            has_signal=True,
            voice_on=True,
            now_ts=10.0,
            last_spoken_label="",
            last_spoken_ts=0.0,
        )
        is False
    )
    assert (
        should_speak(
            label="unknown",
            raw_label="unknown",
            has_signal=True,
            voice_on=True,
            now_ts=10.0,
            last_spoken_label="",
            last_spoken_ts=0.0,
        )
        is False
    )


def test_speak_cooldowns():
    assert (
        should_speak(
            label="yes",
            raw_label="yes",
            has_signal=True,
            voice_on=True,
            now_ts=2.0,
            last_spoken_label="yes",
            last_spoken_ts=1.0,
            repeat_cooldown_sec=1.8,
            global_cooldown_sec=0.35,
        )
        is False
    )
    assert (
        should_speak(
            label="yes",
            raw_label="yes",
            has_signal=True,
            voice_on=True,
            now_ts=4.0,
            last_spoken_label="no",
            last_spoken_ts=3.9,
            repeat_cooldown_sec=1.8,
            global_cooldown_sec=0.35,
        )
        is False
    )
    assert (
        should_speak(
            label="yes",
            raw_label="yes",
            has_signal=True,
            voice_on=True,
            now_ts=6.0,
            last_spoken_label="no",
            last_spoken_ts=4.0,
            repeat_cooldown_sec=1.8,
            global_cooldown_sec=0.35,
        )
        is True
    )
