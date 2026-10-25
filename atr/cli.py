"""atr command line.

    atr serve [--config PATH]           run the MCP server (what your AI client launches)
    atr serve --demo                    ... with a built-in fake 50-tool server, to try ATR out
    atr serve --http [--port 8789]      ... over HTTP, for ChatGPT / claude.ai connectors (needs .[http])
    atr compress FILE|- [--budget N]    shrink a file/paste before you paste it into any chat
                       [--copy]         ... and put the result on the clipboard
    atr init [--from CLIENT_CONFIG]     write ~/.atr/config.json (moving servers over from e.g.
                                        claude_desktop_config.json) and print what to put in the client
    atr snippet                         print the client config entry for ATR
    atr setup [--yes] [--undo]          add ATR to Claude Code / Codex / Claude Desktop / Cursor / Windsurf,
                                        moving their MCP servers behind it
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

from atr.compress import compress
from atr.config import DEFAULT_PATH, Config


def snippet() -> dict:
    # PYTHONPATH as well: macOS can flag the install's .pth files hidden, and Python then skips them
    root = str(Path(__file__).resolve().parents[1])
    return {"atr": {"command": sys.executable, "args": ["-m", "atr", "serve"], "env": {"PYTHONPATH": root}}}


def saved_token() -> str:
    """One secret per machine, kept in ~/.atr/http_token, so the connector URL survives restarts."""
    import os
    import secrets

    p = DEFAULT_PATH.parent / "http_token"
    if p.exists() and p.read_text().strip():
        return p.read_text().strip()
    p.parent.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(24)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok + "\n")
    return tok


DEMO = {"command": sys.executable, "args": ["-m", "atr.tests.fake_server"]}


def load_config(a: argparse.Namespace) -> Config:
    cfg = Config.load(a.config)
    if a.demo:
        cfg.servers = {**cfg.servers, "demo": DEMO}
    return cfg


def cmd_serve(a: argparse.Namespace) -> None:
    if a.http:
        try:
            from atr.remote import run
        except ImportError as e:
            sys.exit(f"HTTP mode needs extra packages: pip install -e '.[http]'  ({e})")
        token = None if a.no_auth else (a.token or saved_token())
        if a.no_auth and a.host not in ("127.0.0.1", "localhost"):
            sys.exit("--no-auth is only allowed on 127.0.0.1")
        run(load_config(a), a.host, a.port, token, a.read_files, a.allow_run, a.project)
        return
    from atr.server import serve

    asyncio.run(serve(load_config(a)))


def cmd_compress(a: argparse.Namespace) -> None:
    raw = sys.stdin.read() if a.file == "-" else Path(a.file).expanduser().read_text(errors="replace")
    r = compress(raw, budget=a.budget, level=a.level)
    print(r.text)
    note = f"{r.before} → {r.after} tokens ({r.saved_pct:.0f}% smaller)" + (" · lossy: " + "; ".join(r.notes) if r.notes else "")
    print(f"[atr] {note}", file=sys.stderr)
    if a.copy:
        tool = shutil.which("pbcopy") or shutil.which("wl-copy") or shutil.which("xclip")
        if not tool:
            print("[atr] no clipboard tool found (pbcopy / wl-copy / xclip)", file=sys.stderr)
            return
        args = [tool, "-selection", "clipboard"] if tool.endswith("xclip") else [tool]
        subprocess.run(args, input=r.text.encode(), check=True)
        print("[atr] copied to clipboard", file=sys.stderr)


def cmd_init(a: argparse.Namespace) -> None:
    servers = {}
    if a.source:
        src = json.loads(Path(a.source).expanduser().read_text())
        servers = {k: v for k, v in (src.get("mcpServers") or src.get("servers") or {}).items() if k != "atr"}
    if DEFAULT_PATH.exists() and not a.force:
        sys.exit(f"{DEFAULT_PATH} already exists (use --force to overwrite)")
    DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_PATH.write_text(json.dumps({"servers": servers, "expose": "search", "pin": [], "budget": 1500,
                                        "level": 2, "read_roots": ["~"]}, indent=2) + "\n")
    print(f"wrote {DEFAULT_PATH} with {len(servers)} server(s): {', '.join(servers) or 'none'}")
    print("\nNow make ATR the only entry under \"mcpServers\" in your client config:\n")
    print(json.dumps({"mcpServers": snippet()}, indent=2))


def cmd_setup(a: argparse.Namespace) -> None:
    from atr import setup

    only = [x.strip() for x in a.only.split(",")] if a.only else None
    sys.exit(setup.undo(only) if a.undo else setup.run(a.yes, only, a.dry_run))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="atr", description="ATR: Automation for Token Reduction")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the MCP server over stdio")
    s.add_argument("--config")
    s.add_argument("--demo", action="store_true", help="add a built-in fake 50-tool server, to try ATR out")
    s.add_argument("--http", action="store_true", help="Streamable HTTP instead of stdio")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8789)
    s.add_argument("--token", help="secret for the URL (default: saved in ~/.atr/http_token)")
    s.add_argument("--no-auth", action="store_true", help="no token (only on 127.0.0.1)")
    s.add_argument("--read-files", action="store_true", help="allow file and code reading over HTTP")
    s.add_argument("--allow-run", action="store_true", help="allow atr_run (shell commands) over HTTP")
    s.add_argument("--project", help="over HTTP: the only folder file/code tools may see")
    s.set_defaults(fn=cmd_serve)
    c = sub.add_parser("compress", help="compress a file or stdin for pasting into a chat")
    c.add_argument("file", nargs="?", default="-")
    c.add_argument("--budget", type=int)
    c.add_argument("--level", type=int, default=2, choices=[1, 2, 3])
    c.add_argument("--copy", action="store_true")
    c.set_defaults(fn=cmd_compress)
    i = sub.add_parser("init", help="create ~/.atr/config.json")
    i.add_argument("--from", dest="source", help="client config to take servers from")
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_init)
    su = sub.add_parser("setup", help="add ATR to your AI apps, with their MCP servers moved behind it")
    su.add_argument("--yes", action="store_true", help="don't ask first")
    su.add_argument("--undo", action="store_true", help="put every app back the way it was")
    su.add_argument("--dry-run", action="store_true", help="only show the plan")
    su.add_argument("--only", help="comma-separated: claude-code,codex,claude-desktop,cursor,windsurf")
    su.set_defaults(fn=cmd_setup)
    sn = sub.add_parser("snippet", help="print the client config entry for ATR")
    sn.set_defaults(fn=lambda a: print(json.dumps({"mcpServers": snippet()}, indent=2)))
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
