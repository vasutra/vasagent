import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vasagent.ktv import extract_lab_values, predict_ktv


def test_predict_ktv():
    assert predict_ktv(10, 2) == 20


def test_extract_lab_values():
    text = "BUN: 8 Creatinine: 1.2"
    assert extract_lab_values(text) == (8.0, 1.2)


def test_main_predict(tmp_path: Path):
    report = {"BUN": 5, "Creatinine": 3}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

    result = subprocess.run(
        [sys.executable, "-m", "vasagent.main", str(report_path), "--predict"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert "Predicted Kt/V: 15" in result.stdout
