"""ATR config: which MCP servers to sit in front of, and how hard to compress.

~/.atr/config.json (or ATR_CONFIG, or --config). The "servers" block uses the same shape as
Claude Desktop's "mcpServers", so you can move your existing servers over as they are:

{
  "servers": {
    "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
               "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "..."}}
  },
  "expose": "search",        // "search": only ATR's meta-tools (+ pinned); "all": every tool, slimmed
  "pin": ["github__create_issue"],   // always visible, even in search mode
  "read_only": ["github__list_*"],    // run through atr_read (no approval) even if the server doesn't say so
  "write": [],                        // always through atr_call, whatever the server says
  "budget": 1500,            // soft cap, in tokens, for any single tool result
  "level": 2,                // 1 lossless-ish, 2 default, 3 aggressive
  "read_roots": ["~"],       // where the file and coding tools may look
  "project": "~/code/app",   // where coding tools start (default: the folder ATR was started in)
  "coding": true,            // atr_repo_map, atr_search_code, atr_run
  "run": true                // atr_run (asks for approval in clients that ask)
}
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(os.environ.get("ATR_CONFIG", "~/.atr/config.json")).expanduser()


@dataclass
class Config:
    servers: dict[str, dict] = field(default_factory=dict)
    expose: str = "search"
    pin: list[str] = field(default_factory=list)
    budget: int = 1500
    level: int = 2
    k: int = 5
    read_only: list[str] = field(default_factory=list)  # tools to treat as read-only (globs: github__list_*)
    write: list[str] = field(default_factory=list)      # tools to always treat as changing things
    read_roots: list[str] = field(default_factory=lambda: ["~"])
    project: str | None = None   # folder the coding tools start in (default: where ATR was started)
    coding: bool = True          # atr_repo_map, atr_search_code, atr_run
    run: bool = True             # atr_run (marked as changing things, so clients ask first)
    run_timeout: int = 300

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path).expanduser() if path else DEFAULT_PATH
        if not p.exists():
            if path:
                raise FileNotFoundError(f"no config at {p}")
            return cls()
        text = re.sub(r"^\s*//.*$", "", p.read_text(), flags=re.M)  # allow // comment lines
        d = json.loads(text)
        servers = d.get("servers") or d.get("mcpServers") or {}
        servers = {k: v for k, v in servers.items() if k != "atr" and not v.get("disabled")}  # never proxy ourselves
        if d.get("expose", "search") not in ("search", "all"):
            raise ValueError("expose must be 'search' or 'all'")
        return cls(servers=servers, expose=d.get("expose", "search"), pin=list(d.get("pin") or []),
                   budget=int(d.get("budget", 1500)), level=int(d.get("level", 2)), k=int(d.get("k", 5)),
                   read_only=list(d.get("read_only") or []), write=list(d.get("write") or []),
                   read_roots=list(d.get("read_roots") or ["~"]),
                   project=d.get("project"), coding=bool(d.get("coding", True)), run=bool(d.get("run", True)),
                   run_timeout=int(d.get("run_timeout", 300)))
