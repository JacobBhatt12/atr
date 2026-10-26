"""Coding tools: repo map, code search, line ranges, compressed command output."""
from __future__ import annotations

import asyncio
import sys

from atr import tokens
from atr.config import Config
from atr.server import ATRServer


def make_project(root):
    (root / "src").mkdir()
    (root / "node_modules" / "junk").mkdir(parents=True)
    (root / "node_modules" / "junk" / "index.js").write_text("function shouldNotAppear() {}\n")
    (root / "src" / "app.py").write_text(
        "import os\n\n\nclass Store:\n    def get(self, key):\n        return key\n\n\n"
        "def handle_login(user):\n    # TODO: rate limit\n    return Store().get(user)\n\n\n"
        "async def main():\n    pass\n")
    (root / "src" / "ui.ts").write_text(
        "export function renderLogin() {}\nexport class LoginForm {}\nexport const API_URL = 'x'\n"
        "interface Props { a: string }\n")
    (root / "README.md").write_text("# Demo\n")
    (root / "big.py").write_text("\n".join(f"def f{i}(x):\n    return x + {i}" for i in range(400)))
    return root


def srv_for(root, **kw):
    return ATRServer(Config(read_roots=[str(root)], project=str(root), **kw))


def call(srv, name, args):
    return asyncio.run(srv.call(name, args))


def test_repo_map_shows_layout_and_symbols(tmp_path):
    srv = srv_for(make_project(tmp_path))
    text = call(srv, "atr_repo_map", {})["content"][0]["text"]
    assert "app.py 15L: Store, handle_login, main" in text
    assert "ui.ts 4L: renderLogin, LoginForm, API_URL, Props" in text
    assert "shouldNotAppear" not in text and "node_modules" not in text
    assert "README.md" in text


def test_repo_map_shrinks_to_budget(tmp_path):
    for d in range(30):
        sub = tmp_path / f"pkg{d}" / "inner"
        sub.mkdir(parents=True)
        for f in range(10):
            (sub / f"mod{f}.py").write_text("\n".join(f"def func_{f}_{i}(): pass" for i in range(20)))
    text = call(srv_for(tmp_path), "atr_repo_map", {"budget": 400})["content"][0]["text"]
    assert tokens.count(text) < 450 and "pass path=" in text


def test_search_code_groups_by_file(tmp_path):
    srv = srv_for(make_project(tmp_path))
    text = call(srv, "atr_search_code", {"pattern": "login"})["content"][0]["text"]
    assert text.startswith("3 matches in 2 files")
    assert "src/app.py" in text and "  9: def handle_login(user):" in text
    only_ts = call(srv, "atr_search_code", {"pattern": "login", "glob": "*.ts"})["content"][0]["text"]
    assert "app.py" not in only_ts and "ui.ts" in only_ts
    assert call(srv, "atr_search_code", {"pattern": "shouldNotAppear"})["content"][0]["text"].startswith("no matches")


def test_read_file_lines_and_context(tmp_path):
    srv = srv_for(make_project(tmp_path))
    text = call(srv, "atr_read_file", {"path": "src/app.py", "lines": "9-11"})["content"][0]["text"]
    assert text.splitlines()[1:] == ["9: def handle_login(user):", "10:     # TODO: rate limit", "11:     return Store().get(user)"]
    ctx = call(srv, "atr_read_file", {"path": "src/app.py", "search": "TODO", "context": 1})["content"][0]["text"]
    assert "9: def handle_login" in ctx and "11:     return" in ctx
    outline = call(srv, "atr_read_file", {"path": "big.py", "budget": 600})["content"][0]["text"]
    assert "Outline" in outline and 'lines="120-180"' in outline


def test_run_compresses_and_reports_exit(tmp_path):
    srv = srv_for(make_project(tmp_path))
    script = tmp_path / "noisy.py"
    script.write_text("import sys\nfor i in range(3000): print(f'test_{i} PASSED')\n"
                      "print('FAILED test_login - AssertionError: expected 200, got 500')\nsys.exit(1)\n")
    res = call(srv, "atr_run", {"command": f"{sys.executable} noisy.py", "budget": 300})
    text = res["content"][0]["text"]
    assert res["isError"] and text.startswith("exit 1")
    assert "FAILED test_login" in text and "similar lines" in text and tokens.count(text) < 400
    slow = call(srv, "atr_run", {"command": f"{sys.executable} -c 'import time; time.sleep(5)'", "timeout": 1})
    assert slow["content"][0]["text"].startswith("timed out")


def test_paths_stay_inside_read_roots(tmp_path):
    srv = srv_for(make_project(tmp_path))
    for name, args in [("atr_repo_map", {"path": "/etc"}), ("atr_search_code", {"pattern": "x", "path": "../.."}),
                       ("atr_read_file", {"path": "/etc/hosts"}), ("atr_run", {"command": "ls", "cwd": "/"})]:
        res = call(srv, name, args)
        assert res["isError"] and "outside read_roots" in res["content"][0]["text"], name


def test_switches(tmp_path):
    srv = srv_for(make_project(tmp_path), coding=False)
    assert not {"atr_repo_map", "atr_search_code", "atr_run"} & {t["name"] for t in srv.list_tools()}
    srv = srv_for(tmp_path, run=False)
    names = {t["name"] for t in srv.list_tools()}
    assert "atr_repo_map" in names and "atr_run" not in names
    assert call(srv, "atr_run", {"command": "ls"})["isError"]


def test_secrets_are_never_read(tmp_path):
    root = make_project(tmp_path)
    (root / ".env").write_text("API_KEY=sk-live-123\n")
    (root / ".env.example").write_text("API_KEY=\n")
    (root / "src" / "server.pem").write_text("-----BEGIN PRIVATE KEY-----\n")
    srv = srv_for(root)
    for args in [{"path": ".env"}, {"path": "src/server.pem"}]:
        res = call(srv, "atr_read_file", args)
        assert res["isError"] and "secret" in res["content"][0]["text"]
    assert not call(srv, "atr_read_file", {"path": ".env.example"})["isError"]
    found = call(srv, "atr_search_code", {"pattern": "API_KEY|PRIVATE"})["content"][0]["text"]
    assert "sk-live" not in found and "BEGIN PRIVATE" not in found and ".env.example" in found
    assert ".env\n" not in call(srv, "atr_repo_map", {})["content"][0]["text"]


def test_secret_names_in_data_files():
    from pathlib import Path
    from atr.code import is_secret
    for name in ["Forms-Access-Keys.ini", "forms_keys.json", "api_keys.json", "passwords.txt", "client_secret.json", "prod.env",
                 ".env.local", "id_ed25519", "aws-credentials", "server.pem"]:
        assert is_secret(Path(name)), name
    for name in ["tokens.py", "secret_santa.py", "keyboard.ts", "README.md", ".env.example", "auth.py"]:
        assert not is_secret(Path(name)), name


def test_nested_git_repo_is_listed(tmp_path):
    """A project that is its own repo inside another repo (e.g. a Desktop that is a repo)."""
    import subprocess
    from atr.code import list_files, search_code
    outer, inner = tmp_path, tmp_path / "My Project"
    (inner / "pkg").mkdir(parents=True)
    (outer / "notes.md").write_text("hi")
    (inner / "pkg" / "compress.py").write_text("def compress(x):\n    return x\n")
    (inner / ".env").write_text("KEY=1")
    for d in (outer, inner):
        subprocess.run(["git", "init", "-q", str(d)], check=True)
    files = {f.as_posix() for f in list_files(outer)}
    assert "My Project/pkg/compress.py" in files and "notes.md" in files
    assert not any(f.endswith(".env") for f in files)
    assert "My Project/pkg/compress.py" in search_code(outer, "def compress", None, 2000)


def test_stats_never_count_map_or_search_as_a_loss(tmp_path):
    import asyncio
    from atr.config import Config
    from atr.server import ATRServer
    (tmp_path / "a.py").write_text("\n".join(f"def f{i}():\n    return {i}" for i in range(400)))
    srv = ATRServer(Config(read_roots=[str(tmp_path)], project=str(tmp_path)))
    asyncio.run(srv.code_tool("atr_repo_map", {"path": str(tmp_path)}))
    asyncio.run(srv.code_tool("atr_search_code", {"path": str(tmp_path), "pattern": "def f1"}))
    srv.read_file(str(tmp_path / "a.py"), None, None, lines="10-20")
    srv.read_file(str(tmp_path / "a.py"), "def f399", None)
    s = srv.stats
    assert s["calls"] == 4 and s["results_after"] < s["results_before"]


def test_tools_behind_atr_cannot_be_pointed_at_secrets():
    from atr.server import secret_path_arg
    assert secret_path_arg({"file": ".env", "start": 1}) == ".env"
    assert secret_path_arg({"path": "app/prod.env"}) == "app/prod.env"
    assert secret_path_arg({"files": ["a.py", "~/.ssh/id_ed25519"]}) == "~/.ssh/id_ed25519"
    assert secret_path_arg({"uri": "file:///Users/x/.aws/credentials"})
    assert secret_path_arg({"options": {"path": "data/forms_keys.json"}})
    # questions and ordinary files are fine
    assert secret_path_arg({"query": "where are secret files like .env blocked"}) is None
    assert secret_path_arg({"file": "atr/server.py", "path": ".env.example"}) is None


def test_downstream_call_refuses_secret_paths():
    import asyncio, sys
    from atr.config import Config
    from atr.server import ATRServer
    async def go():
        srv = ATRServer(Config(servers={"fake": {"command": sys.executable, "args": ["-m", "atr.demo_server"]}}))
        await srv.start()
        try:
            bad = await srv.call("atr_call", {"tool": "fake__github_list_issues", "arguments": {"path": "../.env"}})
            assert bad["isError"] and "secret" in bad["content"][0]["text"]
            ok = await srv.call("atr_call", {"tool": "fake__github_list_issues", "arguments": {"owner": "a", "repo": "b"}})
            assert not ok["isError"]
        finally:
            await srv.close()
    asyncio.run(go())


def test_is_secret_does_not_depend_on_where_atr_started(tmp_path, monkeypatch):
    """A relative path must get the same answer whatever folder the server was started in."""
    from atr.code import is_secret
    from pathlib import Path
    (tmp_path / "api_keys.json").mkdir()          # a folder that happens to share a secret file's name
    monkeypatch.chdir(tmp_path)
    assert is_secret(Path("api_keys.json"))       # relative: judged by name, not by the stray folder
    assert not is_secret(tmp_path / "api_keys.json")  # absolute and really a folder: allowed
    (tmp_path / "secrets-demo").mkdir()
    assert not is_secret(tmp_path / "secrets-demo")


def test_search_code_context_shows_surrounding_lines(tmp_path):
    from atr.code import search_code
    (tmp_path / "m.py").write_text("import os\n\n\ndef target(x):\n    if x:\n        return 1\n    return 2\n\n\ndef other():\n    pass\n")
    out = search_code(tmp_path, "def target", None, 2000, context=2)
    assert "4: def target(x):" in out and "6-         return 1" in out and "2- " in out
    assert "def other" not in out
    plain = search_code(tmp_path, "def target", None, 2000)
    assert "return 1" not in plain


def test_command_output_keeps_failures_in_the_middle():
    from atr.compress import compress_output
    out = ["collected 2003 items"] + [f"tests/test_a.py::test_{i} PASSED [ {i // 20}%]" for i in range(1000)]
    out += ["tests/test_a.py::test_login FAILED", "    def test_login():", ">       assert status == 200",
            "E       assert 500 == 200", "tests/test_a.py:41: AssertionError"]
    out += [f"tests/test_b.py::test_{i} PASSED [ {50 + i // 20}%]" for i in range(1000)]
    out += ["Downloading foo-1.0.tar.gz (12 kB)"] * 0 + ["=== 1 failed, 2000 passed in 3.1s ==="]
    r = compress_output("\n".join(out), budget=300)
    assert "assert 500 == 200" in r.text and "tests/test_a.py:41" in r.text and "1 failed, 2000 passed" in r.text
    assert r.after < 300


def test_command_output_folds_repeated_error_blocks():
    from atr.compress import compress_output
    block = ("FAIL src/m{0}.test.js\n  ● Test suite failed to run\n\n    Cannot find module 'ts-jest' from 'jest.config.js'\n\n"
             "      at Resolver.resolveModule (node_modules/jest-resolve/build/index.js:{0}:11)\n")
    r = compress_output("\n".join(block.format(i) for i in range(40)) + "\nTest Suites: 40 failed, 40 total\n", budget=5000)
    assert r.text.count("Cannot find module") == 1 and "39 more block(s)" in r.text and "40 failed" in r.text
    assert r.lossy  # folded, so the full output is worth stashing


def test_rereading_a_file_returns_unchanged_or_only_the_changes(tmp_path):
    srv = srv_for(make_project(tmp_path))
    args = {"path": "big.py", "lines": "1-60"}
    first = call(srv, "atr_read_file", args)["content"][0]["text"]
    again = call(srv, "atr_read_file", args)["content"][0]["text"]
    assert "unchanged since you read it" in again and tokens.count(again) < tokens.count(first) / 5
    text = (tmp_path / "big.py").read_text().replace("return x + 7\n", "return x * 7\n", 1)
    (tmp_path / "big.py").write_text(text)
    changed = call(srv, "atr_read_file", args)["content"][0]["text"]
    assert "changed since you last read it" in changed and "+16:     return x * 7" in changed and "-16:     return x + 7" in changed
    assert tokens.count(changed) < tokens.count(first) / 3
    full = call(srv, "atr_read_file", {**args, "fresh": True})["content"][0]["text"]
    assert full.startswith("big.py lines 1-60")


def test_diff_hides_lockfiles_and_secrets(tmp_path):
    import subprocess
    git = lambda *a: subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=tmp_path,
                                    check=True, capture_output=True)
    make_project(tmp_path)
    (tmp_path / "package-lock.json").write_text("{}\n")
    git("init", "-q"); git("add", "src", "package-lock.json", "README.md"); git("commit", "-qm", "init")
    (tmp_path / "package-lock.json").write_text("\n".join(f'"pkg{i}": "1.0.{i}",' for i in range(2000)))
    (tmp_path / "src" / "app.py").write_text((tmp_path / "src" / "app.py").read_text().replace("# TODO: rate limit", "limit(user)"))
    (tmp_path / ".env").write_text("API_KEY=sk-live-123\n")
    git("add", "-N", ".env")
    srv = srv_for(tmp_path)
    text = call(srv, "atr_diff", {})["content"][0]["text"]
    assert "+    limit(user)" in text and "-    # TODO: rate limit" in text
    assert "package-lock.json (+2000 -1) [lockfile/generated: not shown]" in text and "pkg5" not in text
    assert ".env" in text and "sk-live" not in text
    assert srv.stats["results_before"] > 10 * srv.stats["results_after"]  # vs. a plain git diff
    assert call(srv, "atr_diff", {"file": ".env"})["isError"]
    assert call(srv, "atr_diff", {"ref": "--output=/tmp/x"})["isError"]


def test_setup_allows_atr_read_tools_in_claude_code(tmp_path, monkeypatch):
    import json
    from atr import setup
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "settings.json").write_text(json.dumps({"permissions": {"allow": ["Bash(ls:*)"]}, "theme": "dark"}))
    cc = setup.ClaudeCode()
    added = cc.allow(["mcp__atr__atr_read_file", "Bash(ls:*)"])
    assert added == ["mcp__atr__atr_read_file"]
    cc.disallow(added)
    assert json.loads((tmp_path / "settings.json").read_text()) == {"permissions": {"allow": ["Bash(ls:*)"]}, "theme": "dark"}


SNAPSHOT = '''### Page
- Page URL: https://news.ycombinator.com/
### Snapshot
```yaml
- table [ref=e3]:
  - rowgroup [ref=e4]:
    - row [ref=e5]:
      - cell "1." [ref=e33]
      - cell [ref=e38]:
        - generic [ref=e39]:
          - link "Xbox Turns Down Netflix $500M Offer" [ref=e40] [cursor=pointer]:
            - /url: https://www.darkhorizons.com/xbox-turns-down-netflix-500-million-offer/
          - text: "|"
          - generic [ref=e46]:
            - text: 57 points by
            - link "xoxxala" [ref=e47] [cursor=pointer]:
              - /url: user?id=xoxxala
      - cell [ref=e60]:
        - textbox "Search" [ref=e61]:
          - /placeholder: Search stories
        - generic [aria-hidden] [ref=e62]: "1"
```
'''


def test_browser_snapshot_keeps_text_and_refs_drops_layout():
    from atr.compress import compact_snapshot, compress
    c = compact_snapshot(SNAPSHOT)
    assert '- Page URL: https://news.ycombinator.com/' in c  # the page header is left alone
    assert 'link "Xbox Turns Down Netflix $500M Offer" [ref=e40] → darkhorizons.com/xbox-turns-down' in c
    assert '57 points by' in c and 'link "xoxxala" [ref=e47] → user?id=xoxxala' in c and 'cell "1." [ref=e33]' in c
    assert 'textbox "Search" [ref=e61] (placeholder: Search stories)' in c
    assert "rowgroup" not in c and "cursor=pointer" not in c and '"|"' not in c and "aria-hidden" not in c
    assert tokens.count(c) < tokens.count(SNAPSHOT) * 0.6
    big = SNAPSHOT.replace("```\n", "".join(f'  - link "Story {i}" [ref=x{i}] [cursor=pointer]:\n    - /url: item?id={i}\n'
                                             for i in range(200)) + "```\n")
    r = compress(big, budget=1500)
    assert not r.lossy and "Story 199" in r.text  # a compacted page isn't cut to the normal budget


def test_atr_call_accepts_arguments_as_a_json_string():
    srv = ATRServer(Config(servers={"fake": {"command": sys.executable, "args": ["-m", "atr.demo_server"]}}))

    async def go():
        await srv.start()
        try:
            res = await srv.call("atr_read", {"tool": "fake__github_get_issue", "args": '{"owner": "a", "repo": "b"}'})
            assert not res["isError"]
        finally:
            await srv.close()
    asyncio.run(go())
