from model.model_manager import ModelHub


def test_model_hub_rollback(tmp_path):
    reg = tmp_path / "registry.json"
    hub = ModelHub(path=reg)

    hub.promote("m1", notes="first")
    hub.promote("m2", notes="second")
    out = hub.rollback(notes="go back")

    assert out["ok"] is True
    assert out["active_model"] == "m1"
    assert hub.active() == "m1"


def test_model_hub_rollback_without_history(tmp_path):
    reg = tmp_path / "registry.json"
    hub = ModelHub(path=reg)
    hub.promote("m1", notes="first")

    out = hub.rollback(notes="go back")
    assert out["ok"] is False
