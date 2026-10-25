"""atr setup: plug ATR into the AI apps on this machine, with their MCP servers moved behind it.

    atr setup            find Claude Code, Codex, Claude Desktop, Cursor and Windsurf, show the plan, ask, do it
    atr setup --yes      ... without asking
    atr setup --undo     put every app back the way it was

Only servers that run on this machine (a command) move behind ATR. Remote servers (a URL) stay
in the app, because they often need a sign-in that ATR can't do yet. ATR itself is never moved.
Every changed file gets a backup next to it (*.atr-backup), and ~/.atr/setup.json records what
moved where so --undo can reverse it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from atr.config import DEFAULT_PATH


def state_path() -> Path:
    return DEFAULT_PATH.parent / "setup.json"


def atr_entry() -> dict:
    # PYTHONPATH as well: macOS can flag an install's .pth files hidden, and Python then skips them
    root = str(Path(__file__).resolve().parents[1])
    return {"command": sys.executable, "args": ["-m", "atr", "serve"], "env": {"PYTHONPATH": root}}


def is_atr(name: str, spec: dict) -> bool:
    args = [str(a) for a in spec.get("args") or []]
    cmd = str(spec.get("command") or "")
    return name == "atr" or ("-m" in args and "atr" in args) or Path(cmd).name == "atr"


def backup(path: Path) -> None:
    b = path.with_name(path.name + ".atr-backup")
    if path.exists() and not b.exists():
        shutil.copy2(path, b)


def write_private(path: Path, text: str) -> None:
    """Write a file only this user can read: moved servers can carry API keys in their env."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)


def local_spec(spec: dict) -> dict | None:
    """The parts ATR needs to run a local server, or None if it isn't one ATR can run."""
    if not spec.get("command") or spec.get("url"):
        return None
    out = {"command": spec["command"], "args": list(spec.get("args") or [])}
    for k in ("env", "cwd"):
        if spec.get(k):
            out[k] = spec[k]
    return out


# ------------------------------------------------------------------ apps


# ATR tools that only read; Claude Code may run these without asking (atr_run and atr_call still ask).
READ_TOOLS = ["atr_find_tools", "atr_query", "atr_read_file", "atr_repo_map", "atr_search_code", "atr_diff", "atr_stats"]


class App:
    key = label = ""
    def allow(self, rules: list[str]) -> list[str]: return []  # permission rules added, for --undo
    def disallow(self, rules: list[str]) -> None: ...

    def present(self) -> bool: ...
    def servers(self) -> dict[str, dict]: ...      # name -> the app's own entry
    def remove(self, name: str) -> None: ...
    def add(self, name: str, spec: dict) -> None: ...  # spec in the app's own shape
    def to_app(self, spec: dict) -> dict: return spec  # ATR/stdio spec -> the app's shape


class JsonApp(App):
    """Apps whose MCP servers live under "mcpServers" in one JSON file."""

    def __init__(self, key: str, label: str, path: Path):
        self.key, self.label, self.path = key, label, path

    def present(self) -> bool:
        return self.path.exists()

    def _load(self) -> dict:
        return json.loads(self.path.read_text() or "{}")

    def _save(self, d: dict) -> None:
        backup(self.path)
        self.path.write_text(json.dumps(d, indent=2) + "\n")

    def servers(self) -> dict[str, dict]:
        return dict(self._load().get("mcpServers") or {})

    def remove(self, name: str) -> None:
        d = self._load()
        (d.get("mcpServers") or {}).pop(name, None)
        self._save(d)

    def add(self, name: str, spec: dict) -> None:
        d = self._load()
        d.setdefault("mcpServers", {})[name] = spec
        self._save(d)


class ClaudeCode(App):
    """Claude Code's user-scope servers. Changed through the `claude` CLI, which owns ~/.claude.json."""

    key, label = "claude-code", "Claude Code"

    def _file(self) -> Path:
        base = os.environ.get("CLAUDE_CONFIG_DIR")
        return Path(base) / ".claude.json" if base else Path.home() / ".claude.json"

    def present(self) -> bool:
        return bool(shutil.which("claude")) and self._file().exists()

    def servers(self) -> dict[str, dict]:
        try:
            return dict(json.loads(self._file().read_text()).get("mcpServers") or {})
        except (OSError, ValueError):
            return {}

    def remove(self, name: str) -> None:
        subprocess.run(["claude", "mcp", "remove", "-s", "user", name], check=True, capture_output=True)

    def add(self, name: str, spec: dict) -> None:
        subprocess.run(["claude", "mcp", "add-json", "-s", "user", name, json.dumps(spec)],
                       check=True, capture_output=True)

    def to_app(self, spec: dict) -> dict:
        return {"type": "stdio", **spec}

    def _settings(self) -> Path:
        base = os.environ.get("CLAUDE_CONFIG_DIR")
        return (Path(base) if base else Path.home() / ".claude") / "settings.json"

    def allow(self, rules: list[str]) -> list[str]:
        f = self._settings()
        d = json.loads(f.read_text() or "{}") if f.exists() else {}
        have = d.setdefault("permissions", {}).setdefault("allow", [])
        added = [r for r in rules if r not in have]
        if added:
            backup(f)
            have.extend(added)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(d, indent=2) + "\n")
        return added

    def disallow(self, rules: list[str]) -> None:
        f = self._settings()
        if not rules or not f.exists():
            return
        d = json.loads(f.read_text() or "{}")
        allow = (d.get("permissions") or {}).get("allow") or []
        d.setdefault("permissions", {})["allow"] = [r for r in allow if r not in rules]
        f.write_text(json.dumps(d, indent=2) + "\n")


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{json.dumps(str(k))} = {toml_value(x)}" for k, x in v.items()) + " }"
    return json.dumps(str(v))  # a TOML basic string uses the same escapes as JSON


class Codex(App):
    """Codex CLI. Servers are [mcp_servers.<name>] tables in config.toml. Removed through the
    `codex` CLI; added by appending the table, which keeps every setting (cwd, timeouts) for --undo."""

    key, label = "codex", "Codex"

    def _file(self) -> Path:
        return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"

    def present(self) -> bool:
        return bool(shutil.which("codex")) and self._file().exists()

    def servers(self) -> dict[str, dict]:
        try:
            return dict(tomllib.loads(self._file().read_text()).get("mcp_servers") or {})
        except (OSError, tomllib.TOMLDecodeError):
            return {}

    def remove(self, name: str) -> None:
        backup(self._file())
        subprocess.run(["codex", "mcp", "remove", name], check=True, capture_output=True)

    def add(self, name: str, spec: dict) -> None:
        backup(self._file())
        f = self._file()
        text = f.read_text()
        lines = [f"[mcp_servers.{json.dumps(name)}]"] + [f"{k} = {toml_value(v)}" for k, v in spec.items()]
        new = text.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n"
        tomllib.loads(new)  # never leave Codex with a file it can't read
        f.write_text(new)


def apps() -> list[App]:
    home = Path.home()
    desktop = (home / "Library/Application Support/Claude/claude_desktop_config.json" if sys.platform == "darwin"
               else Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / "Claude/claude_desktop_config.json"
               if sys.platform == "win32" else home / ".config/Claude/claude_desktop_config.json")
    return [ClaudeCode(), Codex(), JsonApp("claude-desktop", "Claude Desktop", desktop),
            JsonApp("cursor", "Cursor", home / ".cursor/mcp.json"),
            JsonApp("windsurf", "Windsurf", home / ".codeium/windsurf/mcp_config.json")]


# ------------------------------------------------------------------ plan and apply


def why_kept(name: str, spec: dict) -> str | None:
    if is_atr(name, spec):
        return "is ATR"
    if spec.get("enabled") is False or spec.get("disabled"):
        return "turned off"
    if spec.get("cwd") and isinstance(spec.get("cwd"), str) and not Path(spec["cwd"]).expanduser().is_dir():
        return "its folder is missing"
    if local_spec(spec) is None:
        return "remote (sign-in not supported behind ATR yet)"
    return None


def plan(app: App) -> tuple[dict[str, dict], dict[str, str]]:
    move, keep = {}, {}
    for name, spec in app.servers().items():
        reason = why_kept(name, spec)
        if reason:
            if reason != "is ATR":
                keep[name] = reason
        else:
            move[name] = spec
    return move, keep


def load_atr_config() -> dict:
    if not DEFAULT_PATH.exists():
        return {"servers": {}, "expose": "search", "budget": 1500, "level": 2, "read_roots": ["~"]}
    text = re.sub(r"^\s*//.*$", "", DEFAULT_PATH.read_text(), flags=re.M)
    return json.loads(text)


def load_state() -> dict:
    try:
        return json.loads(state_path().read_text())
    except (OSError, ValueError):
        return {}


def run(yes: bool = False, only: list[str] | None = None, dry_run: bool = False, out=print) -> int:
    found = [a for a in apps() if a.present() and (not only or a.key in only)]
    if not found:
        out("No supported AI app found (Claude Code, Codex, Claude Desktop, Cursor, Windsurf).")
        out("Add ATR by hand instead:  " + json.dumps({"atr": atr_entry()}))
        return 1
    plans = {a.key: plan(a) for a in found}
    out("ATR will be added to:")
    for a in found:
        move, keep = plans[a.key]
        out(f"  {a.label}")
        out(f"    servers moved behind ATR: {', '.join(move) or 'none'}")
        for n, r in keep.items():
            out(f"    stays in {a.label}: {n} ({r})")
        if isinstance(a, ClaudeCode):
            out("    ATR's read-only tools run without asking (searching, reading, diffs); atr_run still asks")
    out("Backups are written next to each changed file, and `atr setup --undo` puts everything back.")
    if dry_run:
        return 0
    if not yes:
        if not sys.stdin.isatty():
            out("Run again with --yes to go ahead.")
            return 1
        if input("Go ahead? [y/N] ").strip().lower() not in ("y", "yes"):
            out("Nothing changed.")
            return 1

    cfg = load_atr_config()
    servers = cfg.setdefault("servers", {})
    state = load_state()
    backup(DEFAULT_PATH)
    for a in found:
        move, _ = plans[a.key]
        record = state.setdefault(a.key, {"moved": {}})
        for name, spec in move.items():
            local = local_spec(spec)
            atr_name = name
            if atr_name in servers and servers[atr_name] != local:
                atr_name = f"{name}-{a.key}"
            servers[atr_name] = local
            record["moved"][name] = {"as": atr_name, "spec": spec}
        write_private(DEFAULT_PATH, json.dumps(cfg, indent=2) + "\n")  # save before touching the app
        write_private(state_path(), json.dumps(state, indent=2) + "\n")
        for name in move:
            a.remove(name)
        if "atr" in a.servers():
            a.remove("atr")
        a.add("atr", a.to_app(atr_entry()))
        record["allowed"] = sorted(set(record.get("allowed", [])) | set(a.allow([f"mcp__atr__{t}" for t in READ_TOOLS])))
        write_private(state_path(), json.dumps(state, indent=2) + "\n")
        out(f"✓ {a.label}: ATR added, {len(move)} server(s) moved behind it")
    out("Restart the apps (or start a new chat) to pick up the change.")
    return 0


def undo(only: list[str] | None = None, out=print) -> int:
    state = load_state()
    if not state:
        out("Nothing to undo: `atr setup` hasn't changed any app.")
        return 1
    cfg = load_atr_config()
    servers = cfg.get("servers") or {}
    for a in apps():
        if a.key not in state or (only and a.key not in only) or not a.present():
            continue
        current = a.servers()
        if "atr" in current:
            a.remove("atr")
        for name, info in state[a.key]["moved"].items():
            if name not in current:
                a.add(name, info["spec"])
            servers.pop(info["as"], None)
        a.disallow(state[a.key].get("allowed", []))
        del state[a.key]
        out(f"✓ {a.label}: back to how it was")
    write_private(DEFAULT_PATH, json.dumps(cfg, indent=2) + "\n")
    write_private(state_path(), json.dumps(state, indent=2) + "\n")
    return 0
