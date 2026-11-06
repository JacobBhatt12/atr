# GitHub's official MCP server, with and without ATR in front

`measure.py` connects to `https://api.githubcopilot.com/mcp/` with your `gh` login (the token goes
only to GitHub), then runs the same four read tasks directly and through ATR's proxy
(`atr_read` with `question=`). Tokens are characters / 4.

    PYTHONPATH=. uv run --with-editable ".[http]" python benchmarks/github_mcp/measure.py

## Results (November 6, 2026, 46 tools)

| What | Direct | Through ATR |
|---|---:|---:|
| Tool definitions, if the app loads them all | 32,563 | 917 (6 tools) + 394 per `atr_find_tools` search |
| Tool names only (apps that load on demand, like claude.ai) | ~300 | ~40 |
| `list_issues`, 30 open issues: "which are about OAuth?" | 26,235 | 420 |
| `list_pull_requests`, 30 open PRs: "which touch streamable HTTP?" | 28,284 | 392 |
| `search_repositories`: "which have the most stars?" | 4,232 | 369 |
| `get_file_contents` README: "install command and minimum Python?" | 1,160 | 213 |

The full result of each call is kept on the server; `atr_query` on its ref returns anything that
was cut (the README answer had the install command but not the Python version).

This run found that ATR passed GitHub's file contents through untrimmed: they arrive as an
embedded `resource` block, not text. Fixed in `ATRServer.shrink`.
