"""tools/cpu_speed.py on the tiny fake Datasets tree: it writes its report and nothing else."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
pytest.importorskip("torch")
import cpu_speed  # noqa: E402
from tests.test_dataset_configs import source, snapshot  # noqa: E402,F401


def test_cpu_speed_report(source):  # noqa: F811
    before = snapshot(source)
    cpu_speed.main(["--source", str(source), "--fake-embedder", "--excerpt-minutes", "0.5"])
    d = source / "Configurations" / "_cpu_speed"
    r = json.load(open(d / "cpu_speed.json"))
    assert r["threads"] and r["estimate"]["days_for_target"] > 0
    assert "Estimate" in (d / "cpu_speed.md").read_text()
    assert not (d / "excerpt.npy").exists()
    after = snapshot(source)
    out = str(source / "Configurations")
    assert not [p for p in after if p not in before and not p.startswith(out)]
