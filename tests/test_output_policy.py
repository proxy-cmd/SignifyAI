from core.output_policy import apply_uncertainty_policy


def test_uncertainty_policy_keeps_unknown_labels():
    label, source, uncertain = apply_uncertainty_policy("unknown", 0.20, "none", 0.58)
    assert label == "unknown"
    assert source == "none"
    assert uncertain is False


def test_uncertainty_policy_marks_low_confidence():
    label, source, uncertain = apply_uncertainty_policy("yes", 0.31, "adaptive", 0.58)
    assert label == "uncertain"
    assert source == "adaptive+uncertain"
    assert uncertain is True


def test_uncertainty_policy_keeps_confident_prediction():
    label, source, uncertain = apply_uncertainty_policy("yes", 0.91, "adaptive", 0.58)
    assert label == "yes"
    assert source == "adaptive"
    assert uncertain is False
