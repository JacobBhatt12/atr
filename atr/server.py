"""The ATR MCP server (stdio). Your AI client connects to this one server instead of many.

    client (Claude Desktop / Claude Code / Cursor / Zed)
        │  sees ~5 small ATR tools instead of every tool's full schema
        ▼
    ATR ── atr_find_tools ── BM25 over all downstream tool definitions
        ├─ atr_call ──────── forwards to the real server, compresses the result, stashes the rest
        ├─ atr_query ─────── slices a stashed result (filter, fields, search, group-by, paging)
        ├─ atr_read_file ─── reads a local file as compact text (outline for big code files)
        └─ atr_stats ─────── tokens saved so far
        │
        ▼
    github, postgres, filesystem, ... (spawned by ATR, unchanged)
"""
from __future__ import annotations

import asyncio
import difflib
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from atr import __version__, code, tokens
from atr.compress import compress, compress_output, head_tail
from atr.config import Config
from atr.downstream import Downstream, log, make_downstream
from atr.router import ToolIndex, signature, slim_schema, topics
from atr.stash import Stash, query

# argument names that hold a file or folder path (checked against the secret-file rules)
PATH_KEYS = re.compile(r"^(path|paths|file|files|filename|filenames|file_?path|file_?paths|filepath|uri|url|dir|directory|"
                       r"cwd|source|target|src|dest|destination|from|to|module|location|root)$", re.I)


def secret_path_arg(args: Any, key: str = "") -> str | None:
    """The first path-like argument that names a secret file (e.g. file=".env"), else None."""
    if isinstance(args, dict):
        for k, v in args.items():
            hit = secret_path_arg(v, k)
            if hit:
                return hit
    elif isinstance(args, list):
        for v in args:
            hit = secret_path_arg(v, key)
            if hit:
                return hit
    elif isinstance(args, str) and PATH_KEYS.match(key) and args.strip():
        p = args.strip().removeprefix("file://")
        if code.is_secret(Path(p).expanduser()):
            return p
    return None

SUPPORTED = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]
SEP = "__"  # exposed name = <server>__<tool>
CODE_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".swift", ".c", ".cc", ".cpp",
            ".h", ".hpp", ".cs", ".rb", ".php", ".scala", ".lua", ".sh"}
OUTLINE = re.compile(r"^\s*(def |async def |class |function |export |public |private |protected |func |fn |"
                     r"pub fn |impl |interface |type |struct |enum |const \w+ = \(|module |package )")
RICH_DOCS = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".doc", ".epub"}

INSTRUCTIONS = (
    "ATR saves tokens. Coding: run tests, builds, installs and linters with atr_run (failures, errors and the "
    "summary are kept; passing tests and progress lines are dropped), see changes with atr_diff instead of "
    "`git diff`, find code with atr_search_code (context=3 often answers without opening the file), and read "
    "with atr_read_file using lines= or search= rather than whole files. Re-reading a file you already read "
    "returns 'unchanged' or only the changed lines; pass fresh=true if you no longer have it. "
    "Tools from connected servers are hidden: atr_find_tools with a short task description, then atr_read "
    "for tools marked [r] (read-only) or atr_call for [w] (they change something and may need approval). "
    "Large results come back summarized with a ref; use atr_query on the ref instead of calling the tool again.")

READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
CALL_SCHEMA = {"type": "object", "properties": {
    "tool": {"type": "string"},
    "arguments": {"type": "object"},
    "budget": {"type": "integer", "description": "Max tokens for the result (default from config)."}},
    "required": ["tool"]}

META_TOOLS = [
    {"name": "atr_find_tools",
     "description": "Find tools on the connected servers for a task. Returns compact signatures (* = required).",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "What you want to do, in a few words."},
         "k": {"type": "integer", "description": "How many tools (default 5)."},
         "detail": {"type": "boolean", "description": "Also return the parameter schemas."}},
         "required": ["query"]},
     "annotations": READ_ONLY},
    {"name": "atr_read",
     "description": "Run a read-only tool ([r] in atr_find_tools). Result is compressed; big ones are stashed.",
     "inputSchema": CALL_SCHEMA,
     "annotations": READ_ONLY},
    {"name": "atr_call",
     "description": "Run a tool that changes something ([w] in atr_find_tools). Result is compressed.",
     "inputSchema": CALL_SCHEMA,
     "annotations": {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True}},
    {"name": "atr_query",
     "description": "Read part of a stashed result by ref. Data: path/where/fields/group_by. Text: search (regex).",
     "inputSchema": {"type": "object", "properties": {
         "ref": {"type": "string"},
         "path": {"type": "string", "description": "Dotted path, e.g. data.items or items.*.user"},
         "where": {"type": "array", "items": {"type": "string"},
                   "description": "Filters: key=v, key!=v, key>n, key<n, key~regex"},
         "fields": {"type": "array", "items": {"type": "string"}},
         "search": {"type": "string"},
         "group_by": {"type": "string"},
         "offset": {"type": "integer"},
         "limit": {"type": "integer"}},
         "required": ["ref"]},
     "annotations": READ_ONLY},
    {"name": "atr_read_file",
     "description": "Read part of a local file. Pass lines= (e.g. from a search hit) or search= to get only what you need; "
                    "with just a path, big code files return an outline instead of the whole file.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string"},
         "lines": {"type": "string", "description": "Line range, e.g. 120-180."},
         "search": {"type": "string", "description": "Regex: only matching lines, with line numbers."},
         "context": {"type": "integer", "description": "Lines around each search match (default 0)."},
         "fresh": {"type": "boolean", "description": "Send the full text even if you read it before."},
         "budget": {"type": "integer"}},
         "required": ["path"]},
     "annotations": READ_ONLY},
    {"name": "atr_repo_map",
     "description": "Project layout with each code file's top-level functions/classes. Start here in a codebase.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Folder (default: the project)."},
         "budget": {"type": "integer"}}},
     "annotations": READ_ONLY},
    {"name": "atr_search_code",
     "description": "Find where something is defined or used, before reading any file. Regex, grouped by file "
                    "(path:line: text); context=3 shows the lines around each hit, often enough to answer.",
     "inputSchema": {"type": "object", "properties": {
         "pattern": {"type": "string"},
         "path": {"type": "string", "description": "Folder (default: the project)."},
         "glob": {"type": "string", "description": "Only these files, e.g. *.py or src/**/*.ts"},
         "context": {"type": "integer", "description": "Lines to show around each match (default 0, try 3)."},
         "budget": {"type": "integer"}},
         "required": ["pattern"]},
     "annotations": READ_ONLY},
    {"name": "atr_diff",
     "description": "What changed, compactly: git diff with 1 line of context, lockfiles/generated/binary files "
                    "listed but not shown. Default: uncommitted changes. Use instead of `git diff`.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Folder inside the repo (default: the project)."},
         "ref": {"type": "string", "description": "Compare with this commit/branch, e.g. main or HEAD~1."},
         "staged": {"type": "boolean", "description": "Only staged changes."},
         "file": {"type": "string", "description": "Only this file (path from the repo root)."},
         "budget": {"type": "integer"}}},
     "annotations": READ_ONLY},
    {"name": "atr_run",
     "description": "Run a shell command (tests, build, lint) and get compressed output: failures kept, noise cut.",
     "inputSchema": {"type": "object", "properties": {
         "command": {"type": "string"},
         "cwd": {"type": "string", "description": "Folder (default: the project)."},
         "timeout": {"type": "integer", "description": "Seconds (default 300)."},
         "budget": {"type": "integer"}},
         "required": ["command"]},
     "annotations": {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True}},
    {"name": "atr_stats", "description": "Tokens ATR has saved in this session.",
     "inputSchema": {"type": "object", "properties": {}},
     "annotations": READ_ONLY},
]


FILE_TOOLS = {"atr_read_file", "atr_repo_map", "atr_search_code", "atr_diff"}
CODE_TOOLS = {"atr_repo_map", "atr_search_code", "atr_diff", "atr_run"}


def text_result(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class ATRServer:
    def __init__(self, config: Config):
        self.cfg = config
        self.downstreams: dict[str, Downstream] = {}
        self.tools: dict[str, tuple[Downstream, dict]] = {}  # exposed name -> (server, tool definition)
        self.index = ToolIndex({})
        self.stash = Stash()
        self.reads: dict[tuple, str] = {}  # what atr_read_file already showed this session
        self.ready = asyncio.Event()
        self.out_lock = asyncio.Lock()
        self.stats = {"results_before": 0, "results_after": 0, "calls": 0, "stashed": 0,
                      "schema_full": 0, "schema_exposed": 0}
        self.notify_client = None  # set by the stdio loop
        self.allow_read_files = True  # file + code reading; off by default over HTTP (atr/remote.py)
        self.allow_run = config.run   # atr_run; off by default over HTTP

    # ------------------------------------------------------------ downstream servers

    async def start(self) -> None:
        async def one(name: str, spec: dict) -> None:
            d = make_downstream(name, spec, on_tools_changed=self._tools_changed)
            self.downstreams[name] = d
            try:
                await d.start()
                log(f"{name}: {len(d.tools)} tools")
            except Exception as e:  # noqa: BLE001
                d.error = str(e)
                log(f"{name}: failed to start: {e}")

        await asyncio.gather(*(one(n, s) for n, s in self.cfg.servers.items()))
        self._rebuild()
        self.ready.set()

    def _rebuild(self) -> None:
        self.tools = {f"{n}{SEP}{t['name']}": (d, t) for n, d in self.downstreams.items() for t in d.tools}
        self.index = ToolIndex({name: t for name, (_, t) in self.tools.items()})
        full = [dict(t, name=name) for name, (_, t) in self.tools.items()]
        self.stats["schema_full"] = tokens.count(json.dumps(full)) if full else 0
        self.stats["schema_exposed"] = tokens.count(json.dumps(self.list_tools()))

    async def _tools_changed(self, _name: str) -> None:
        self._rebuild()
        if self.notify_client and (self.cfg.expose == "all" or self.cfg.pin):
            await self.notify_client("notifications/tools/list_changed")

    async def close(self) -> None:
        await asyncio.gather(*(d.close() for d in self.downstreams.values()), return_exceptions=True)

    # ------------------------------------------------------------ what the client sees

    def is_read_only(self, name: str) -> bool:
        """Safe to run without asking: the server marks it read-only (and not destructive), or the
        config lists it under read_only. `write` in the config wins over both. Unmarked = write."""
        if any(fnmatch.fnmatchcase(name, p) for p in self.cfg.write):
            return False
        if any(fnmatch.fnmatchcase(name, p) for p in self.cfg.read_only):
            return True
        ann = (self.tools[name][1].get("annotations") or {}) if name in self.tools else {}
        return ann.get("readOnlyHint") is True and ann.get("destructiveHint") is not True

    def _exposed(self, name: str, t: dict) -> dict:
        out = {"name": name, "description": slim_schema({"description": t.get("description", "")}, 160).get("description", ""),
               "inputSchema": slim_schema(t.get("inputSchema") or {"type": "object"})}
        if t.get("annotations"):
            out["annotations"] = t["annotations"]
        return out

    def connected_summary(self) -> str:
        """One line telling the model what is behind ATR, so it uses ATR without being told to."""
        parts = []
        for name in sorted(self.downstreams):
            d = self.downstreams[name]
            if d.tools:
                parts.append(f"{name} ({', '.join(topics([t['name'] for t in d.tools]))})")
        return f" Connected: {'; '.join(parts)}." if parts else ""

    def list_tools(self) -> list[dict]:
        summary = self.connected_summary()
        tools = []
        for t in META_TOOLS:
            if t["name"] in FILE_TOOLS and not self.allow_read_files:
                continue
            if t["name"] in CODE_TOOLS and not self.cfg.coding:
                continue
            if t["name"] == "atr_run" and not (self.allow_run and self.cfg.coding):
                continue
            if t["name"] == "atr_find_tools" and summary:
                t = dict(t, description=t["description"] + summary)
            tools.append(t)
        names = self.tools if self.cfg.expose == "all" else [p for p in self.cfg.pin if p in self.tools]
        tools += [self._exposed(n, self.tools[n][1]) for n in names]
        return tools

    # ------------------------------------------------------------ tool calls

    def shrink(self, label: str, result: dict, budget: int | None) -> dict:
        """Compress every text block of a downstream tool result; stash what had to be cut."""
        budget = budget or self.cfg.budget
        content = list(result.get("content") or [])
        texts = [c for c in content if c.get("type") == "text"]
        if not texts and result.get("structuredContent") is not None:
            texts = [{"type": "text", "text": json.dumps(result["structuredContent"])}]
            content = texts
        out, before, after = [], 0, 0
        per = max(200, budget // max(1, len(texts)))
        for c in content:
            if c.get("type") != "text":
                out.append(c)  # images, audio, resource links pass through
                continue
            r = compress(c.get("text", ""), budget=per, level=self.cfg.level)
            text = r.text
            if r.lossy:
                ref = self.stash.put(label, c.get("text", ""))
                self.stats["stashed"] += 1
                text += (f"\n[atr: cut {r.before}→{r.after} tokens. Full result is {ref}: "
                         f"atr_query(ref=\"{ref}\", path=…, where=[…], fields=[…], search=…) for the rest]")
            before += r.before
            after += tokens.count(text)
            out.append({"type": "text", "text": text})
        # structuredContent repeats the text blocks; sending both doubles the cost
        self.stats["results_before"] += before + (tokens.count(json.dumps(result["structuredContent"]))
                                                  if result.get("structuredContent") is not None and len(texts) else 0)
        self.stats["results_after"] += after
        self.stats["calls"] += 1
        return {"content": out, "isError": bool(result.get("isError"))}

    async def call_downstream(self, name: str, args: dict, budget: int | None = None) -> dict:
        await self.ready.wait()
        if name not in self.tools:
            hits = self.index.search(name.replace(SEP, " "), 3)
            hint = f" Did you mean: {', '.join(h for h, _ in hits)}?" if hits else ""
            return text_result(f"No tool named {name!r}.{hint} Use atr_find_tools.", True)
        d, t = self.tools[name]
        secret = secret_path_arg(args)
        if secret:  # tools behind ATR get the same secret-file rule as atr_read_file
            return text_result(f"{secret} looks like a secret (keys, tokens, .env); ATR never passes it to a tool.", True)
        try:
            res = await d.call_tool(t["name"], args)
        except Exception as e:  # noqa: BLE001
            return text_result(f"{name} failed: {e}", True)
        return self.shrink(name, res, budget)

    async def find_tools(self, q: str, k: int | None, detail: bool) -> dict:
        await self.ready.wait()
        if not self.tools:
            down = [f"{n}: {d.error}" for n, d in self.downstreams.items() if d.error]
            return text_result("No downstream tools are connected." + (" Errors: " + "; ".join(down) if down else
                               " Add servers to ~/.atr/config.json."), True)
        hits = self.index.search(q, k or self.cfg.k)
        if not hits:
            servers = ", ".join(f"{n} ({len(d.tools)})" for n, d in self.downstreams.items())
            return text_result(f"No tools match {q!r}. Servers: {servers}. Try other words.")
        lines = [f"[{'r' if self.is_read_only(n) else 'w'}] {signature(n, self.tools[n][1])}" for n, _ in hits]
        lines.append("[r]: run with atr_read. [w]: run with atr_call.")
        if detail:
            lines += ["", "schemas:"] + [f"{n}: {json.dumps(slim_schema(self.tools[n][1].get('inputSchema') or {}), separators=(',', ':'))}"
                                         for n, _ in hits]
        return text_result("\n".join(lines))

    def resolve(self, path: str | None) -> Path:
        """A path the model gave, relative to the project; must sit inside read_roots."""
        base = Path(self.cfg.project).expanduser() if self.cfg.project else Path.cwd()
        p = (base / Path(path or ".").expanduser()).resolve()
        roots = [Path(r).expanduser().resolve() for r in self.cfg.read_roots]
        if not any(p == r or r in p.parents for r in roots):
            raise PermissionError(f"{p} is outside read_roots ({', '.join(map(str, roots))}).")
        if code.is_secret(p):
            raise PermissionError(f"{p.name} looks like a secret (keys, tokens, .env); ATR never reads those.")
        return p

    def read_file(self, path: str, search: str | None, budget: int | None, lines: str | None = None,
                  context: int = 0, fresh: bool = False) -> dict:
        """atr_read_file, remembering what this session already read: the same read again returns
        "unchanged", and a file that changed since returns only the changed lines."""
        stats = dict(self.stats)
        res = self._read_file(path, search, budget, lines, context)
        if res.get("isError"):
            return res
        key = (str(self.resolve(path)), lines or "", search or "", context)
        text = res["content"][0]["text"]
        shown = re.sub(r"\br\d+\b", "r#", text)  # stash refs differ between reads of the same thing
        prev = self.reads.get(key)
        self.reads[key] = shown
        if fresh or prev is None:
            return res
        if prev == shown:
            note = (f"[atr: {Path(key[0]).name} is unchanged since you read it earlier in this session (same "
                    f"{'lines' if lines else 'search' if search else 'read'}). Pass fresh=true if you no longer have it.]")
        else:
            diff = [ln for ln in difflib.unified_diff(prev.splitlines(), shown.splitlines(), lineterm="", n=1)
                    if not ln.startswith(("---", "+++"))]
            note = (f"[atr: {Path(key[0]).name} changed since you last read it. Only the changes ('-' old, '+' new); "
                    f"pass fresh=true for the full text.]\n" + "\n".join(diff))
            if tokens.count(note) * 2 > tokens.count(text):  # most of it changed: just send it
                return res
        # count it as the read it replaced, not a second full read
        self.stats.update({k: stats[k] for k in ("results_before", "results_after", "calls")})
        self._count(tokens.count(text), tokens.count(note))
        return text_result(note)

    def _read_file(self, path: str, search: str | None, budget: int | None, lines: str | None = None,
                   context: int = 0) -> dict:
        budget = budget or self.cfg.budget
        try:
            p = self.resolve(path)
        except PermissionError as e:
            return text_result(str(e), True)
        if not p.is_file():
            return text_result(f"No file at {p}.", True)
        if p.suffix.lower() in RICH_DOCS:
            try:
                from markitdown import MarkItDown  # optional: pip install 'markitdown[all]'
                raw = MarkItDown().convert(str(p)).text_content
            except ImportError:
                return text_result(f"{p.suffix} needs the optional converter: pip install 'markitdown[all]'", True)
        else:
            raw = p.read_text(errors="replace")
        all_lines = raw.splitlines()
        if lines:
            m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", str(lines))
            if not m:
                return text_result("lines must look like 120-180 or 120", True)
            lo = max(1, int(m.group(1)))
            hi = min(len(all_lines), int(m.group(2) or lo))
            body = "\n".join(f"{i}: {all_lines[i - 1]}" for i in range(lo, hi + 1))
            text = f"{p.name} lines {lo}-{hi} of {len(all_lines)}\n{body}"
            if tokens.count(text) > budget:  # code: trim by length only, never touch indentation
                text = head_tail(text, budget) + "\n[atr: range too big for the budget; ask for fewer lines]"
            self._count(tokens.count(raw), tokens.count(text))  # instead of reading the whole file
            return text_result(text)
        if search:
            try:
                rx = re.compile(search, re.I)
            except re.error as e:
                return text_result(f"bad regex: {e}", True)
            match = [i for i, ln in enumerate(all_lines) if rx.search(ln)]
            keep: list[int] = []  # line indexes to show; -1 marks a gap
            last = -1
            for i in match:
                for j in range(max(0, i - context), min(len(all_lines), i + context + 1)):
                    if j > last:
                        if last >= 0 and j > last + 1 and context:
                            keep.append(-1)
                        keep.append(j)
                        last = j
            body = "\n".join("--" if j < 0 else f"{j + 1}: {all_lines[j]}" for j in keep) or "(no matches)"
            if tokens.count(body) > budget:
                body = head_tail(body, budget)
            text = f"{p.name}: {len(match)} matching lines of {len(all_lines)}\n{body}"
            self._count(tokens.count(raw), tokens.count(text))  # instead of reading the whole file
            return text_result(text)
        before = tokens.count(raw)
        if p.suffix.lower() in CODE_EXT and before > budget:
            ref = self.stash.put(str(p), raw)
            lines = raw.splitlines()
            outline = [f"{i + 1}: {ln.rstrip()}" for i, ln in enumerate(lines) if OUTLINE.match(ln)]
            body = compress("\n".join(outline), budget=budget, level=self.cfg.level).text
            text = (f"{p.name}: {len(lines)} lines, ~{before} tokens. Outline:\n{body}\n"
                    f"[atr: read the parts you need with atr_read_file(path, lines=\"120-180\") or search=…]")
            self._count(before, tokens.count(text))
            return text_result(text)
        r = compress(raw, budget=budget, level=self.cfg.level)
        text = r.text
        if r.lossy:
            ref = self.stash.put(str(p), raw)
            text += f"\n[atr: cut {r.before}→{r.after} tokens. Full file is {ref}; use atr_query]"
        self._count(r.before, tokens.count(text))
        return text_result(text)

    async def code_tool(self, name: str, a: dict) -> dict:
        if not self.cfg.coding:
            return text_result("coding tools are off in the config (coding: false).", True)
        budget = int(a.get("budget") or (max(3000, self.cfg.budget) if name == "atr_repo_map" else self.cfg.budget))
        try:
            root = self.resolve(a.get("cwd") if name == "atr_run" else a.get("path"))
        except PermissionError as e:
            return text_result(str(e), True)
        if not root.is_dir():
            return text_result(f"No folder at {root}.", True)
        if name == "atr_repo_map":
            text = await asyncio.to_thread(code.repo_map, root, budget)
            self._count(tokens.count(text), tokens.count(text))  # no fair baseline: counted as neutral
            return text_result(text)
        if name == "atr_search_code":
            try:
                re.compile(a.get("pattern", ""))
            except re.error as e:
                return text_result(f"bad regex: {e}", True)
            text = await asyncio.to_thread(code.search_code, root, a.get("pattern", ""), a.get("glob"), budget,
                                     context=min(int(a.get("context") or 0), 10))
            self._count(tokens.count(text), tokens.count(text))  # no fair baseline: counted as neutral
            return text_result(text)
        if name == "atr_diff":
            try:
                text, plain = await asyncio.to_thread(code.git_diff, root, a.get("ref"), bool(a.get("staged")),
                                                      a.get("file"), budget)
            except (ValueError, OSError, subprocess.TimeoutExpired) as e:
                return text_result(f"atr_diff: {e}", True)
            self._count(max(plain, tokens.count(text)), tokens.count(text))  # vs. a plain `git diff`
            return text_result(text)
        if not self.allow_run:
            return text_result("atr_run is off on this server (start it with --allow-run to allow).", True)
        timeout = float(a.get("timeout") or self.cfg.run_timeout)
        rc, out, secs = await code.run(a.get("command", ""), root, timeout)
        r = compress_output(out or "(no output)", budget=budget, level=self.cfg.level)
        status = f"timed out after {timeout:.0f}s" if rc is None else f"exit {rc}"
        text = f"{status} · {secs:.1f}s · {len(out.splitlines())} lines of output\n{r.text}"
        if r.lossy:
            ref = self.stash.put(f"$ {a.get('command')}", out)
            self.stats["stashed"] += 1
            text += f"\n[atr: full output is {ref}; atr_query(ref=\"{ref}\", search=\"FAIL|Error\") for more]"
        self._count(r.before, tokens.count(text))
        return text_result(text, is_error=rc != 0)

    def _count(self, before: int, after: int) -> None:
        self.stats["results_before"] += before
        self.stats["results_after"] += after
        self.stats["calls"] += 1

    def query_ref(self, a: dict) -> dict:
        try:
            label, raw = self.stash.get(a["ref"])
            res = query(raw, path=a.get("path"), where=a.get("where"), fields=a.get("fields"),
                        search=a.get("search"), group_by=a.get("group_by"),
                        offset=int(a.get("offset") or 0), limit=int(a.get("limit") or 20))
        except (KeyError, ValueError, IndexError, re.error) as e:
            return text_result(f"atr_query: {e}", True)
        r = compress(res, budget=self.cfg.budget, level=self.cfg.level)
        return text_result(f"{label}:\n{r.text}")

    def stats_text(self) -> str:
        s = self.stats
        saved = s["results_before"] - s["results_after"]
        pct = 100 * saved / s["results_before"] if s["results_before"] else 0
        schema_saved = s["schema_full"] - s["schema_exposed"]
        servers = ", ".join(f"{n}: {len(d.tools)} tools" + (f" (error: {d.error})" if d.error and not d.tools else "")
                            for n, d in self.downstreams.items()) or "none"
        return (f"tool results: {s['results_before']}→{s['results_after']} tokens over {s['calls']} calls "
                f"(saved {saved}, {pct:.0f}%), {s['stashed']} stashed\n"
                f"tool schemas: all downstream tools would cost {s['schema_full']} tokens per turn; "
                f"ATR exposes {s['schema_exposed']} (saves {max(0, schema_saved)} every turn)\n"
                f"servers: {servers}")

    async def call(self, name: str, a: dict) -> dict:
        if name == "atr_find_tools":
            return await self.find_tools(a.get("query", ""), a.get("k"), bool(a.get("detail")))
        if name in ("atr_read", "atr_call"):
            tool = a.get("tool", "")
            if name == "atr_read" and tool in self.tools and not self.is_read_only(tool):
                return text_result(f"{tool} can change things, so it isn't allowed through atr_read. "
                                   f"Use atr_call (the user may be asked to approve it).", True)
            return await self.call_downstream(tool, a.get("arguments") or {}, a.get("budget"))
        if name == "atr_query":
            return self.query_ref(a)
        if name in FILE_TOOLS and not self.allow_read_files:
            return text_result(f"{name} is off on this server (start it with --read-files to allow).", True)
        if name == "atr_read_file":
            return self.read_file(a.get("path", ""), a.get("search"), a.get("budget"), a.get("lines"),
                                  int(a.get("context") or 0), bool(a.get("fresh")))
        if name in ("atr_repo_map", "atr_search_code", "atr_diff", "atr_run"):
            return await self.code_tool(name, a)
        if name == "atr_stats":
            await self.ready.wait()
            return text_result(self.stats_text())
        return await self.call_downstream(name, a)  # pinned or expose=all tools

    # ------------------------------------------------------------ JSON-RPC

    async def handle(self, msg: dict) -> dict | None:
        method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if mid is None:  # notification
            return None
        try:
            if method == "initialize":
                want = params.get("protocolVersion")
                result = {"protocolVersion": want if want in SUPPORTED else SUPPORTED[1],
                          "capabilities": {"tools": {"listChanged": True}},
                          "serverInfo": {"name": "atr", "title": "ATR: Automation for Token Reduction",
                                         "version": __version__},
                          "instructions": INSTRUCTIONS}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                await self.ready.wait()  # the connected-servers line needs the downstream tool lists
                result = {"tools": self.list_tools()}
            elif method == "tools/call":
                result = await self.call(params.get("name", ""), params.get("arguments") or {})
            elif method in ("resources/list", "prompts/list"):
                result = {method.split("/")[0]: []}
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        except Exception as e:  # noqa: BLE001
            log(f"error handling {method}: {e!r}")
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}


async def serve(config: Config) -> None:
    """Run over stdin/stdout until the client hangs up."""
    srv = ATRServer(config)
    loop = asyncio.get_running_loop()
    out = sys.stdout.buffer

    async def write(obj: Any) -> None:
        data = (json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        async with srv.out_lock:
            out.write(data)
            out.flush()

    async def notify(method: str) -> None:
        await write({"jsonrpc": "2.0", "method": method})

    srv.notify_client = notify
    starting = asyncio.create_task(srv.start())

    async def run(msg: Any) -> None:
        if isinstance(msg, list):
            res = [r for r in await asyncio.gather(*(srv.handle(m) for m in msg)) if r]
            if res:
                await write(res)
        else:
            r = await srv.handle(msg)
            if r:
                await write(r)

    tasks: set[asyncio.Task] = set()
    try:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.buffer.readline)
            if not line:
                break
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                await write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                continue
            t = asyncio.create_task(run(msg))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
    finally:
        if tasks:
            await asyncio.wait(tasks, timeout=5)
        starting.cancel()
        await srv.close()
