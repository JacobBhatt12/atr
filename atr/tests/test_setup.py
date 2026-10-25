"""atr setup on a JSON-config app (Claude Desktop shape), in a temporary home. No real app is touched."""
from __future__ import annotations

import json
import tomllib

from atr import setup


def test_setup_moves_local_servers_and_undo_restores(tmp_path, monkeypatch):
    cfg_file = tmp_path / "claude_desktop_config.json"
    before = {"mcpServers": {
        "files": {"command": "npx", "args": ["-y", "server-filesystem", "/tmp"], "env": {"KEY": "x"}},
        "remote": {"url": "https://example.com/mcp"},
        "atr": {"command": "python", "args": ["-m", "atr", "serve"]},
        "off": {"command": "x", "disabled": True}}, "theme": "dark"}
    cfg_file.write_text(json.dumps(before))
    app = setup.JsonApp("claude-desktop", "Claude Desktop", cfg_file)
    monkeypatch.setattr(setup, "apps", lambda: [app])
    monkeypatch.setattr(setup, "DEFAULT_PATH", tmp_path / ".atr" / "config.json")
    monkeypatch.setattr(setup, "state_path", lambda: tmp_path / ".atr" / "setup.json")

    move, keep = setup.plan(app)
    assert list(move) == ["files"] and set(keep) == {"remote", "off"}  # ATR itself is neither

    lines = []
    assert setup.run(yes=True, out=lines.append) == 0
    after = json.loads(cfg_file.read_text())
    assert set(after["mcpServers"]) == {"atr", "remote", "off"} and after["theme"] == "dark"
    atr_cfg = json.loads((tmp_path / ".atr" / "config.json").read_text())
    assert atr_cfg["servers"]["files"]["env"] == {"KEY": "x"}
    assert oct((tmp_path / ".atr" / "config.json").stat().st_mode & 0o777) == "0o600"  # env can hold API keys
    assert (tmp_path / "claude_desktop_config.json.atr-backup").exists()

    assert setup.undo(out=lines.append) == 0
    restored = json.loads(cfg_file.read_text())
    assert set(restored["mcpServers"]) == {"files", "remote", "off"}  # the old atr entry was ATR anyway
    assert restored["mcpServers"]["files"] == before["mcpServers"]["files"]
    assert json.loads((tmp_path / ".atr" / "config.json").read_text())["servers"] == {}


def test_toml_tables_round_trip():
    spec = {"command": "node", "args": ["a b", 'q"uote'], "cwd": "/x", "startup_timeout_sec": 30,
            "enabled": True, "env": {"K": "v\\n"}}
    text = "[mcp_servers.\"my-server\"]\n" + "\n".join(f"{k} = {setup.toml_value(v)}" for k, v in spec.items())
    assert tomllib.loads(text)["mcp_servers"]["my-server"] == spec
