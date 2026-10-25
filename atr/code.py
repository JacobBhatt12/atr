"""Coding tools: the parts of a coding session that burn the most tokens, made small.

  repo_map     the project's layout and the main functions/classes in each file
  search_code  regex search across the project, grouped by file and capped
  run          a shell command (tests, builds, linters) with its output compressed
  git_diff     the working-tree (or staged, or vs. a ref) diff without lockfiles, generated files or secrets

Every result goes through the same compression and budget as other tool results; anything
cut is stashed, so atr_query can page through the rest.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from atr import tokens

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build", ".next", ".nuxt",
             "target", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".idea", ".vscode", "coverage",
             ".gradle", ".cache", "vendor", "Pods", ".turbo", ".parcel-cache", ".svelte-kit", "out"}
CODE_EXT = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".swift", ".c",
            ".cc", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".scala", ".lua", ".sh", ".vue", ".svelte", ".dart"}
TEXT_EXT = CODE_EXT | {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".html", ".css", ".scss",
                       ".sql", ".graphql", ".proto", ".env.example", ".xml", ".gradle", ".dockerfile", ""}
# top-level definitions per language (no indentation = top level)
SYMBOL = [
    re.compile(r"^(?:async\s+def|def|class)\s+(\w+)"),                                        # Python
    re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?|class|interface|type|enum)\s+(\w+)"),  # JS/TS
    re.compile(r"^(?:export\s+)(?:const|let)\s+(\w+)"),                                       # exported consts
    re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)"), re.compile(r"^type\s+(\w+)"),              # Go
    re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|mod)\s+(\w+)"),  # Rust
    re.compile(r"^(?:public\s+|internal\s+|abstract\s+|final\s+|sealed\s+|data\s+|open\s+)*(?:class|interface|object|enum|struct)\s+(\w+)"),
]


SECRET_DIRS = {".ssh", ".aws", ".gnupg", ".atr", ".thrift-ai", ".kube", ".docker", ".azure", ".config/gcloud", ".password-store"}
SECRET_FILE = re.compile(r"^(\.env(\..*)?|.+\.env|\.netrc|\.git-credentials|\.npmrc|\.pypirc|id_(rsa|dsa|ecdsa|ed25519)(\.pub)?"
                         r"|.*\.(pem|key|p12|pfx|keystore|jks)|credentials(\.json)?|secrets?\.(json|ya?ml|toml)"
                         r"|http_token|.*\.sqlite3?)$", re.I)


SECRET_WORDS = re.compile(r"secret|access[-_ ]?keys?|api[-_ ]?keys?|apikey|passw(or)?d|credential|private[-_ ]?key"
                          r"|auth[-_ ]?token|(tokens?|keys?)\.(txt|json|ini|ya?ml|csv|toml|cfg)$|keychain|vault", re.I)


def is_secret(path: Path) -> bool:
    """Never shown by ATR: keys, tokens, .env files, cloud credentials (example files are fine)."""
    parts = path.parts
    if any(p in SECRET_DIRS for p in parts) or ".config/gcloud" in path.as_posix():
        return True
    # name rules are for files; a folder called "secrets-demo" is just a folder. Only ask the disk about
    # full paths: a relative path would be looked up from wherever ATR was started, not the project.
    if path.is_absolute() and path.is_dir():
        return False
    name = path.name
    if name.endswith((".example", ".sample", ".template")):
        return False
    if SECRET_FILE.match(name):
        return True
    # data/config files whose name says what they hold (code files like tokens.py are fine)
    return path.suffix.lower() not in CODE_EXT and bool(SECRET_WORDS.search(name))


def _ignored(rel: Path) -> bool:
    return any(part in SKIP_DIRS or part.startswith(".venv") for part in rel.parts[:-1])


def list_files(root: Path, limit: int = 20000) -> list[Path]:
    """Project files, relative to root. Uses git (respects .gitignore) when root is in a repo."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            files: list[Path] = []
            for f in out.stdout.splitlines():
                p = Path(f)
                if not f or is_secret(root / p):
                    continue
                if f.endswith("/") and (root / p).is_dir():
                    if _ignored(p / "x"):
                        continue
                    # git shows a nested repo as one folder entry; list that repo's own files
                    files += [p / sub for sub in list_files(root / p, limit - len(files))]
                elif not _ignored(p) and (root / p).is_file():
                    files.append(p)
                if len(files) >= limit:
                    break
            return files[:limit]
    except (OSError, subprocess.TimeoutExpired):
        pass
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".venv") and not d.startswith("."))
        rel_dir = Path(dirpath).relative_to(root)
        for f in sorted(filenames):
            if is_secret(rel_dir / f):
                continue
            files.append(rel_dir / f)
            if len(files) >= limit:
                return files
    return files


def _is_text(p: Path) -> bool:
    if (p.suffix.lower() in TEXT_EXT or p.suffix.lower() in (".example", ".sample", ".template")
            or p.name in ("Dockerfile", "Makefile", "Procfile", ".gitignore")):
        try:
            with open(p, "rb") as f:
                return b"\0" not in f.read(2048)
        except OSError:
            return False
    return False


def symbols(path: Path, limit: int = 12) -> tuple[int, list[str]]:
    """(line count, top-level names) for a code file."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return 0, []
    names: list[str] = []
    for ln in text.splitlines():
        if ln[:1] in (" ", "\t", "#", "/", "*", "@") or not ln.strip():
            continue
        for rx in SYMBOL:
            m = rx.match(ln)
            if m and m.group(1) not in names:
                names.append(m.group(1))
                break
    more = len(names) - limit
    return len(text.splitlines()), names[:limit] + ([f"+{more}"] if more > 0 else [])


def repo_map(root: Path, budget: int) -> str:
    files = list_files(root)
    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for f in files:
        by_dir[f.parent].append(f)

    def render(with_symbols: bool, max_depth: int | None, symbol_depth: int = 99) -> str:
        out = [f"{root.name}/ ({len(files)} files)"]
        for d in sorted(by_dir):
            depth = len(d.parts)
            if max_depth is not None and depth > max_depth:
                continue
            fs = by_dir[d]
            if max_depth is not None and depth == max_depth:
                nested = sum(len(v) for k, v in by_dir.items() if k == d or d in k.parents)
                out.append(f"{'  ' * depth}{d.as_posix()}/ ({nested} files)")
                continue
            if d.parts:
                out.append(f"{'  ' * (depth - 1)}{d.as_posix()}/")
            pad = "  " * depth
            plain = []
            for f in sorted(fs):
                if with_symbols and depth <= symbol_depth and f.suffix.lower() in CODE_EXT:
                    n, names = symbols(root / f)
                    out.append(f"{pad}{f.name} {n}L" + (f": {', '.join(names)}" if names else ""))
                else:
                    plain.append(f.name)
            if plain:
                shown = plain[:15]
                out.append(pad + ", ".join(shown) + (f", +{len(plain) - 15} more" if len(plain) > 15 else ""))
        return "\n".join(out)

    # shrink step by step: functions/classes only near the top, then names only, then fewer levels
    steps = [(True, None, 99), (True, None, 1), (True, None, 0), (False, None, 0), (False, 3, 0), (False, 2, 0), (False, 1, 0)]
    for with_symbols, depth, sym_depth in steps:
        text = render(with_symbols, depth, sym_depth)
        if tokens.count(text) <= budget:
            if (with_symbols, sym_depth) != (True, 99):
                text += "\n[atr: map shortened to fit; pass path= a subfolder to see its functions and classes]"
            return text
    return text + "\n[atr: still large; pass path= a subfolder]"


def search_code(root: Path, pattern: str, glob: str | None, budget: int, per_file: int = 6, context: int = 0) -> str:
    """file -> [(line, text)] for a regex, via ripgrep when installed. context=N adds N lines around each match."""
    hits: dict[str, list[tuple[int, str]]] = defaultdict(list)
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--line-number", "--no-heading", "--color", "never", "--max-columns", "300", "-i", "-e", pattern,
               "-g", "!.env*", "-g", "!*.pem", "-g", "!*.key", "-g", "!id_*"]
        if glob:
            cmd += ["-g", glob]
        out = subprocess.run(cmd + ["."], cwd=root, capture_output=True, text=True, timeout=60)
        for ln in out.stdout.splitlines():
            parts = ln.split(":", 2)
            if len(parts) == 3 and parts[1].isdigit() and not is_secret(Path(parts[0])):
                hits[parts[0].removeprefix("./")].append((int(parts[1]), parts[2]))
    else:
        rx = re.compile(pattern, re.I)
        for f in list_files(root):
            if glob and not (f.match(glob) or Path(f.name).match(glob)):
                continue
            p = root / f
            if not _is_text(p):
                continue
            try:
                for i, ln in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if rx.search(ln):
                        hits[f.as_posix()].append((i, ln))
            except OSError:
                continue
    total = sum(len(v) for v in hits.values())
    if not total:
        return f"no matches for {pattern!r}" + (f" in {glob}" if glob else "")
    lines = [f"{total} matches in {len(hits)} files"]
    shown_files = 0
    for f in sorted(hits, key=lambda k: (-len(hits[k]), k)):
        block = [f] + (_with_context(root / f, [n for n, _ in hits[f][:per_file]], context) if context > 0
                       else [f"  {n}: {t.strip()[:160]}" for n, t in hits[f][:per_file]])
        if len(hits[f]) > per_file:
            block.append(f"  … +{len(hits[f]) - per_file} more in this file")
        if tokens.count("\n".join(lines + block)) > budget:
            break
        lines += block
        shown_files += 1
    if shown_files < len(hits):
        lines.append(f"… +{len(hits) - shown_files} more files; narrow it with glob= or a more specific pattern")
    return "\n".join(lines)


def _with_context(path: Path, line_numbers: list[int], context: int) -> list[str]:
    """Matching lines plus `context` lines around each, indentation kept, "--" between separate chunks."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return [f"  {n}: (unreadable)" for n in line_numbers]
    keep: list[int] = []
    for n in line_numbers:
        for j in range(max(1, n - context), min(len(lines), n + context) + 1):
            if not keep or j > keep[-1]:
                if keep and j > keep[-1] + 1:
                    keep.append(-1)
                keep.append(j)
    return ["  --" if j < 0 else f"  {j}{':' if j in line_numbers else '-'} {lines[j - 1].rstrip()[:200]}" for j in keep]


async def run(command: str, cwd: Path, timeout: float) -> tuple[int | None, str, float]:
    """(exit code or None on timeout, combined output, seconds)."""
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_shell(command, cwd=str(cwd), stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.STDOUT, stdin=asyncio.subprocess.DEVNULL,
                                                 start_new_session=True)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        code = proc.returncode
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, _ = await proc.communicate()
        code = None
    return code, out.decode(errors="replace"), time.monotonic() - t0


# ------------------------------------------------------------------ git diff

GENERATED = re.compile(
    r"(^|/)(package-lock\.json|npm-shrinkwrap\.json|yarn\.lock|pnpm-lock\.yaml|bun\.lockb?|poetry\.lock|Pipfile\.lock|"
    r"uv\.lock|pdm\.lock|Cargo\.lock|Gemfile\.lock|composer\.lock|go\.sum|flake\.lock|Podfile\.lock|packages\.lock\.json)$|"
    r"\.min\.(js|css)$|\.map$|\.snap$|_pb2\.pyi?$|\.pb\.go$|\.generated\.\w+$|"
    r"(^|/)(dist|build|out|\.next|coverage|__snapshots__|vendor)/")
REF = re.compile(r"[\w./~^@{}:-]+")


def _git(args: list[str], cwd: Path) -> str:
    r = subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=str(cwd), capture_output=True, text=True,
                       errors="replace", timeout=60)
    if r.returncode != 0:
        raise ValueError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"git {args[0]} failed")
    return r.stdout


def git_diff(root: Path, ref: str | None = None, staged: bool = False, file: str | None = None,
             budget: int = 1500) -> tuple[str, int]:
    """(compact diff, tokens a plain `git diff` would have cost). Lockfiles, generated and binary files
    are listed but not shown; secret-looking files never show their content."""
    top = Path(_git(["rev-parse", "--show-toplevel"], root).strip())
    if ref and (ref.startswith("-") or not REF.fullmatch(ref)):
        raise ValueError(f"not a git ref: {ref!r}")
    if file and (file.startswith("-") or is_secret(top / file)):
        raise ValueError(f"{file} looks like a secret (keys, tokens, .env); ATR never shows those.")
    base = ["diff", "--no-color", "--no-ext-diff"] + (["--staged"] if staged else []) + ([ref] if ref else [])
    scope = ["--", file] if file else []
    plain = _git(base + scope, top)
    numstat = _git(base + ["--numstat"] + scope, top)
    compact = _git(base + ["-U1"] + scope, top)
    stat = {}
    for ln in numstat.splitlines():
        a, d, path = ln.split("\t", 2)
        stat[path.split(" => ")[-1].strip("{}")] = (a, d)
    hunks: dict[str, list[str]] = {}
    cur = None
    for ln in compact.splitlines():
        if ln.startswith("diff --git "):
            cur = ln.split(" b/", 1)[-1]
            hunks[cur] = []
        elif cur is not None and not ln.startswith(("index ", "--- ", "+++ ", "similarity index", "new file mode",
                                                    "deleted file mode", "old mode", "new mode")):
            hunks[cur].append(ln)
    plus = sum(int(a) for a, _ in stat.values() if a.isdigit())
    minus = sum(int(d) for _, d in stat.values() if d.isdigit())
    what = f"staged changes" if staged else (f"changes since {ref}" if ref else "uncommitted changes")
    out = [f"{what}: {len(hunks)} file(s), +{plus} -{minus}"]
    used = tokens.count(out[0])
    hidden = []
    for path, body in hunks.items():
        a, d = stat.get(path, ("?", "?"))
        head = f"== {path} (+{a} -{d})"
        if is_secret(top / path):
            out.append(f"{head} [secret-looking file: content not shown]")
            continue
        if a == "-" or any("Binary files" in ln for ln in body):
            out.append(f"{head} [binary]")
            continue
        if GENERATED.search(path):
            out.append(f"{head} [lockfile/generated: not shown]")
            continue
        text = "\n".join([head] + [ln[:300] for ln in body])
        cost = tokens.count(text)
        if used + cost > budget and not file:
            hidden.append(f"{head} [not shown: atr_diff(file=\"{path}\")]")
            continue
        if used + cost > budget:  # one file asked for, still too big: keep the start
            keep, total = [], used
            for ln in text.splitlines():
                total += tokens.count(ln) + 1
                if total > budget:
                    keep.append(f"… {len(text.splitlines()) - len(keep)} more lines (narrow it with ref= or a smaller change)")
                    break
                keep.append(ln)
            text = "\n".join(keep)
        out.append(text)
        used += tokens.count(text)
    out += hidden
    if not staged and not ref and not file:
        untracked = [u for u in _git(["ls-files", "--others", "--exclude-standard"], top).splitlines()]
        if untracked:
            names = ", ".join(untracked[:15]) + (f", … {len(untracked) - 15} more" if len(untracked) > 15 else "")
            out.append(f"new files not in git yet ({len(untracked)}): {names}")
    if len(out) == 1:
        out.append("(no changes)")
    return "\n".join(out), tokens.count(plain)
