import importlib

validate_agent_graph = importlib.import_module("validate_agent_graph")


def _registry(tmp_path):
    p = tmp_path / "langgraph.json"
    p.write_text('{"graphs": {"bare": "x", "dcode": "y", "tau3": "z"}}')
    return str(p)


def test_valid_impl_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_IMPL", "bare")
    assert validate_agent_graph.main([_registry(tmp_path)]) == 0


def test_unknown_impl_returns_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_IMPL", "nope")
    assert validate_agent_graph.main([_registry(tmp_path)]) == 1
    assert "::error::" in capsys.readouterr().out


def test_missing_registry_returns_two(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_IMPL", "bare")
    assert validate_agent_graph.main([str(tmp_path / "absent.json")]) == 2


def test_missing_graphs_object_returns_two(tmp_path, monkeypatch, capsys):
    # A registry with no (or an empty/non-dict) "graphs" object is a broken
    # registry, not a bad impl choice; it must not degrade to the "unknown
    # impl ... (have: [])" exit-1 path (see test_unknown_impl_returns_one).
    p = tmp_path / "langgraph.json"
    p.write_text('{"nograph": {}}')
    monkeypatch.setenv("AGENT_IMPL", "bare")
    assert validate_agent_graph.main([str(p)]) == 2
    assert "has no 'graphs' object" in capsys.readouterr().out
