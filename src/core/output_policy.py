def apply_uncertainty_policy(label, conf, source, min_conf):
    label_txt = str(label)
    if label_txt in {"unknown", "silence"}:
        return label_txt, source, False
    if float(conf) < float(min_conf):
        src = str(source) if source else "none"
        return "uncertain", f"{src}+uncertain", True
    return label_txt, source, False


def should_speak(
    label,
    raw_label,
    has_signal,
    voice_on,
    now_ts,
    last_spoken_label,
    last_spoken_ts,
    repeat_cooldown_sec=1.8,
    global_cooldown_sec=0.35,
):
    if not bool(voice_on):
        return False
    if not bool(has_signal):
        return False

    label_txt = str(label)
    if label_txt in {"unknown", "silence", "uncertain"}:
        return False
    if raw_label is None or str(raw_label) != label_txt:
        return False

    now = float(now_ts)
    last = float(last_spoken_ts)
    if label_txt == str(last_spoken_label) and (now - last) < float(repeat_cooldown_sec):
        return False
    if (now - last) < float(global_cooldown_sec):
        return False
    return True
