import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vasagent.ktv import (
    extract_lab_values,
    predict_ktv,
    predict_ktv_from_text,
)
from vasagent.qa import answer_query


def test_predict_ktv():
    assert predict_ktv(10, 2) == 20


def test_extract_lab_values():
    text = "BUN: 8 Creatinine: 1.2"
    assert extract_lab_values(text) == (8.0, 1.2)


def test_predict_ktv_from_text_direct():
    text = "BUN: 3 Creatinine: 2"
    assert predict_ktv_from_text(text) == 6


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

from vasagent import mcp_agent
from vasagent.mcp_agent import MCPAgent

class DummyCompletions:
    @staticmethod
    def create(model=None, messages=None):
        class _Msg:
            def __init__(self, content):
                self.content = content
        class _Choice:
            def __init__(self, content):
                self.message = _Msg(content)
        class _Resp:
            def __init__(self, content):
                self.choices = [_Choice(content)]
        return _Resp("BUN: 4 Creatinine: 1")

class DummyChat:
    completions = DummyCompletions()

class DummyOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = DummyChat()


def test_mcp_agent(monkeypatch):
    monkeypatch.setattr(mcp_agent, "OpenAI", DummyOpenAI)
    agent = MCPAgent(api_key="test")
    result = agent.predict("Sample report")
    assert result == 4.0


def test_predict_ktv_from_text_agent(monkeypatch):
    monkeypatch.setattr(mcp_agent, "OpenAI", DummyOpenAI)
    monkeypatch.setattr("vasagent.mcp_agent.MCPAgent", MCPAgent)
    result = predict_ktv_from_text("no labs", api_key="key")
    assert result == 4.0

class DummyChain:
    def __init__(self):
        self.invocations = []
    def invoke(self, args):
        self.invocations.append(args)
        return {"answer": "A"}

def test_answer_query_predict(monkeypatch):
    chain = DummyChain()
    def dummy_predict(text, api_key=None):
        assert text == "BUN: 2 Creatinine: 2"
        assert api_key == "key"
        return 4.0
    monkeypatch.setattr("vasagent.qa.predict_ktv_from_text", dummy_predict)
    result = answer_query("Predict KtV", "BUN: 2 Creatinine: 2", chain, api_key="key")
    assert result == "Predicted Kt/V: 4.0"
    assert chain.invocations == []


def test_answer_query_other(monkeypatch):
    chain = DummyChain()
    def dummy_predict(*args, **kwargs):
        raise AssertionError("should not call")
    monkeypatch.setattr("vasagent.qa.predict_ktv_from_text", dummy_predict)
    result = answer_query("What is BUN", "text", chain)
    assert result == "A"
    assert chain.invocations == [{"input": "What is BUN"}]
