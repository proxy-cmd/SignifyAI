import json

from core.telemetry import SessionTelemetry


def test_writes_jsonl(tmp_path):
    t = SessionTelemetry(out_dir=tmp_path, session_id="abc123")
    try:
        t.log({"mode": "default", "final_label": "yes", "final_conf": 0.9})
        t.log({"mode": "default", "final_label": "no", "final_conf": 0.8})
    finally:
        t.close()

    p = tmp_path / "abc123.jsonl"
    assert p.exists()
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    row = json.loads(lines[0])
    assert row["session_id"] == "abc123"
    assert row["mode"] == "default"
    assert row["final_label"] == "yes"
    assert "ts_ms" in row
