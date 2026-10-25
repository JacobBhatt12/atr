# ATR: Automation for Token Reduction

**Make any AI do the same work with far fewer tokens, so a free plan goes much further and a paid plan rarely hits its limits.**

ATR is an MCP server that sits between your AI app and your other tools. The AI connects to ATR only, and ATR runs your real MCP servers behind it. Two things eat most of the tokens in tool-using chats, and ATR goes after both:

| Bloat | What happens without ATR | What ATR does |
|---|---|---|
| **Schema bloat** | Every connected tool's full definition is sent on *every* turn, whether it gets used or not | Shows the AI 5 small tools. It finds the one it needs with `atr_find_tools` |
| **Response bloat** | A tool returns 400k tokens of raw JSON and all of it goes into the chat | Compresses every result to fit a budget, keeps the full result on the server, and lets the AI query it |

Owner: Jacob Bhatt ([@JacobBhatt12](https://github.com/JacobBhatt12)) · Started: October 2026 · Budget: $0

---

## Measured savings

From `python -m benchmarks.bench_atr`, using the tiktoken `cl100k` tokenizer. The test server has 50 tools and returns results shaped like real ones (GitHub REST payloads, service logs, web pages). The raw numbers are in `benchmarks/results/atr_*.json`.

| Scenario | Before | After | Saved |
|---|---:|---:|---:|
| 50 tool schemas, paid on every turn | 11,256 | 541 | **95.2%** (+134 per `atr_find_tools` call) |
| 1,000 GitHub issues (JSON), 1,500-token budget | 429,203 | 879 | **99.8%** (the rest is kept for `atr_query`) |
| 30 GitHub issues, nothing cut | 12,878 | 3,448 | **73.2%** |
| 2,000-line service log, 600-token budget | 61,072 | 206 | **99.7%** (the error and traceback are kept) |
| Fetched web page (HTML) | 930 | 350 | **62.4%** |

Compression runs locally in 1–270 ms. It doesn't call a model or the network.

---

## How it works

```
 Claude Desktop / Claude Code / Cursor / Zed / VS Code
            │   sees 5 ATR tools (~540 tokens) instead of every schema (~11,000+)
            ▼
 ┌────────────────────────────── ATR ──────────────────────────────┐
 │ atr_find_tools  BM25 search over all tool names/descriptions →   │
 │                 3–5 one-line signatures: create_issue(owner*:str…)│
 │ atr_read        run a read-only tool → compress → stash          │
 │ atr_call        run a tool that changes things (same pipeline)   │
 │ atr_query       filter / pick fields / search / group / page a   │
 │                 stashed result: only the slice enters the chat   │
 │ atr_read_file   local file as compact text; outline for big code │
 │ atr_stats       tokens saved so far                              │
 └──────────────────────────────────────────────────────────────────┘
            │   spawns and talks to them exactly like your client would
            ▼
   github · postgres · slack · filesystem · … (unchanged)
```

**What compression does** (`atr/compress.py`), from safest to most aggressive:

1. Parses JSON (or JSON lines) that tools return as text.
2. Drops `null`, empty values and noise keys (`_links`, `etag`, `node_id`, …).
3. Writes data as `key: value` lines instead of quoted, braced JSON.
4. Turns lists of similar objects into one CSV table. Nested fields are flattened (`user.login`). A column that has the same value in every row is written once in the header (`all rows: state=open`).
5. Converts HTML to text, strips ANSI colors, and collapses repeated log lines: `… 1959 similar lines …`.
6. Drops `structuredContent` when it repeats the text content, which many servers send both ways.
7. When there's a budget, it shortens long strings, samples big tables, and keeps the head and tail of long text (that's where errors usually are). Anything cut is marked in the output, and the full result is stashed under a ref like `r3`.

The model can then ask for exactly what it needs:

```
atr_query(ref="r3", where=["state=open", "comments>5"], fields=["number","title","user.login"])
atr_query(ref="r3", group_by="state")            → {"open": 685, "closed": 315}
atr_query(ref="r4", search="ERROR|Traceback")     → matching lines only, with line numbers
```

This is the "code execution" pattern: the full data stays out of the context window, and the model reads only a summary or a slice.

---

## Try it in one minute

You need Python 3.11+. Nothing else.

```bash
pip install "git+https://github.com/JacobBhatt12/atr"
atr setup
```

`atr setup` finds Claude Code, Codex, Claude Desktop, Cursor and Windsurf on your machine, shows you the plan, and asks before it changes anything. Then for each app it:

1. moves the app's MCP servers behind ATR (only the ones that run on your machine; remote servers that need a sign-in stay where they are), and
2. adds ATR to the app.

Restart the app and it works: tool lists and tool results go through ATR and get smaller. `atr setup --undo` puts every app back exactly as it was, and each changed file also gets an `.atr-backup` copy. `atr setup --dry-run` only shows the plan.

**No MCP servers yet?** Use the demo instead. It's a built-in fake server with 50 tools (GitHub, Slack, logs, a database, a calendar), so you can watch the savings right away:

```bash
claude mcp add atr -- atr serve --demo        # Claude Code
codex mcp add atr -- atr serve --demo         # Codex CLI
```

Ask something like *"list the open GitHub issues"*, then *"show ATR stats"*. The demo data is fake and nothing leaves your machine.

**What to expect.** ATR saves the most when you have several MCP servers or tools that return big results (issue lists, logs, database rows, web pages). Claude Code already loads MCP tool descriptions only when it needs them, so there most of the savings come from smaller results. Without any MCP servers, ATR adds its coding tools (`atr_repo_map`, `atr_search_code`, `atr_read_file`). They work on your real files with no setup and never open secrets such as `.env` files and keys, but on code alone the savings are small.

## Install and connect

```bash
git clone https://github.com/JacobBhatt12/atr && cd atr
pip install -e .                       # no dependencies; add .[exact,docs] for tiktoken + PDF/Word support
atr init --from ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

`atr init` copies your existing servers into `~/.atr/config.json` and prints the one entry your client needs:

```json
{ "mcpServers": { "atr": { "command": "/path/to/python", "args": ["-m", "atr", "serve"] } } }
```

Then **replace** your client's other `mcpServers` with that entry. If you keep both, the old servers still send their full schemas and you save nothing.

| Client | Where the config goes |
|---|---|
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Code | `claude mcp add atr -- python -m atr serve` (and remove the servers ATR now runs) |
| Cursor | `~/.cursor/mcp.json` |
| Zed / VS Code / Windsurf | their MCP settings, same command and args |

If ATR says `No module named 'atr'` on macOS, the install's `.pth` files were flagged hidden, and recent Python versions skip hidden `.pth` files. Run `chflags nohidden .venv*/lib/python*/site-packages/*.pth`. The config entry from `atr snippet` sets `PYTHONPATH` so it isn't affected.

### Coding tools

For coding, ATR adds four tools that cut the usual token sinks: opening whole files, wide searches, and long test or build output.

| Tool | Instead of | Example on this repo |
|---|---|---|
| `atr_repo_map` | opening files one by one to learn the layout | whole project, with each file's functions and classes: **~3,000 tokens** (reading every `.py` file: ~159,000) |
| `atr_search_code` | grep output that runs for pages | results grouped by file, capped per file, `glob=` to narrow it down |
| `atr_read_file` | reading a 500-line file to see one function | an outline first, then `lines="120-180"`, or `search=` with `context=` |
| `atr_run` | 3,000 lines of test output | exit code + the failures; repeated lines collapsed; full output kept for `atr_query` |

- **Read-only tools:** `atr_repo_map`, `atr_search_code` and `atr_read_file`. Code output is only ever trimmed by length, so indentation is never touched.
- **`atr_run`** is marked as changing things, so clients ask before running it.
- **Where they work:** all four stay inside `read_roots` and start in `project` (default: the folder ATR was started from).
- **Local apps:** in Claude Code, Cursor and similar apps, they're on by default.
- **Over `--http`:** they're off unless you start with `--read-files`, plus `--allow-run` for commands. Add `--project <folder>` to limit them to one folder: anyone with the link gets what those flags allow.
- **Secrets are never read, listed or searched, in any mode:** `.env` files, SSH and other private keys, cloud credentials, ATR's own token, and data files whose names say they hold keys, passwords or tokens (for example `Access-Keys.ini`). Template files like `.env.example` are fine.

### Read vs. write: approve only what changes things

ATR runs downstream tools through two tools, so your client can treat them differently:

- **`atr_read`** runs tools that only read. A tool counts as read-only when its server marks it that way (`readOnlyHint: true`, not destructive) or your config lists it under `read_only`. ATR marks `atr_read` read-only too, so you can set it to **Always allow**.
- **`atr_call`** runs everything else. ATR marks it as changing things, so you keep it on **Needs approval**.
- **Unmarked tools count as changing things.** A tool sent to `atr_read` that isn't read-only is refused, and the model is told to use `atr_call`.
- **You can see it before anything runs:** `atr_find_tools` tags every result `[r]` or `[w]`.
- **Overrides:** many servers don't mark their tools yet. Use `read_only` in the config to allow them (globs work: `github__list_*`), or `write` to force approval. `write` wins over everything.

In claude.ai, set `atr_find_tools`, `atr_read`, `atr_query` and `atr_stats` to Always allow and `atr_call` to Needs approval. Reads then run on their own, and changes ask first.

### Config (`~/.atr/config.json`)

```jsonc
{
  "servers": { "github": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"], "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "…"} } },
  "expose": "search",                 // "search": only ATR's tools; "all": every tool, slimmed, results compressed
  "pin": ["github__create_issue"],    // tools you use constantly, always shown directly
  "budget": 1500,                     // max tokens for any single tool result
  "level": 2,                         // 1 = nearly lossless, 2 = default, 3 = aggressive
  "read_roots": ["~"]                 // folders atr_read_file is allowed to read
}
```

### For claude.ai and ChatGPT: `atr serve --http`

claude.ai and ChatGPT connect to *remote* MCP servers. That means a public https URL, because their servers have to reach yours.

```bash
pip install -e '.[http]'
atr serve --http                                   # prints http://127.0.0.1:8789/mcp/<secret>
cloudflared tunnel --url http://127.0.0.1:8789     # free; prints https://<random>.trycloudflare.com
```

Then add `https://<random>.trycloudflare.com/mcp/<secret>` as a connector:

- **claude.ai:** Settings → Connectors → Add custom connector
- **ChatGPT:** Settings → Apps & Connectors → developer mode → create a connector, with no authentication. The secret in the URL is the auth.

These menus move around, and which plans allow custom connectors changes often, so check each app's help page.

**Security.** Anyone with that URL can run your downstream tools with your credentials.

- The secret is created once and saved in `~/.atr/http_token`. Delete that file to rotate it.
- `atr_read_file` is off over HTTP unless you pass `--read-files`.
- Browser requests from origins other than localhost, claude.ai and chatgpt.com are refused.
- Stop the tunnel when you're not using it.

For a link that never changes, use Tailscale Funnel instead of a quick tunnel: `tailscale funnel --bg 8789` gives `https://<machine>.<tailnet>.ts.net/mcp/<secret>`.

ATR can also sit in front of **remote** MCP servers. Give a server a `url` (and optional `headers`, for example an `Authorization` bearer token) instead of a `command`. OAuth sign-in for remote servers isn't supported yet.

### Without MCP: `atr compress`

```bash
atr compress dump.json --budget 800 --copy     # compress, copy to clipboard, paste into any chat
pbpaste | atr compress --copy                  # shrink whatever is on the clipboard
```

---

## What works where

| App / plan | MCP server (schemas + results) | `atr compress` |
|---|---|---|
| Claude Desktop (Free, Pro, Max) | ✅ local MCP servers | ✅ |
| Claude Code, Codex CLI | ✅ | ✅ |
| Cursor, Zed, VS Code, Windsurf | ✅ | ✅ |
| claude.ai web (plans with custom connectors) | ✅ `atr serve --http` + a tunnel | ✅ |
| ChatGPT paid (developer mode) | ✅ `atr serve --http` + a tunnel | ✅ |
| ChatGPT Free, Gemini app | ❌ no custom MCP | ✅ |
| Your own app / agent | ✅ (if it uses MCP) | ✅ |

Plan features change often. Check your app's current MCP support.

### Limits

- **A free plan won't become a Pro plan.** Usage limits are set by the provider, and paid plans also include better models and features. ATR makes each message cost less, so the same limit covers a lot more work, especially in tool-heavy chats. Measuring this on real usage is part of month 3.
- **The MCP server can't see or edit your chat history.** MCP servers only see the tool calls sent to them, never the conversation. Some guides say an MCP server can summarize your chat automatically; it can't. For long chats, use the client's own `/compact`, or start a new chat with a short summary.
- **Searching for tools adds one step.** Finding a tool is an extra call (about 130 tokens). Pin the tools you use constantly to skip it.

---

## Repo layout

```
atr/                ATR (stdlib only; HTTP mode needs .[http])
  server.py         the MCP server your client connects to (stdio)
  remote.py         the same server over Streamable HTTP (atr serve --http)
  downstream.py     MCP client for the servers ATR runs behind it (stdio and HTTP)
  code.py           repo map, code search, command runner (coding tools)
  compress.py       the compression pipeline
  stash.py          full results kept by ref + atr_query
  router.py         BM25 tool search, schema slimming, one-line signatures
  config.py · cli.py · tokens.py
  tests/            34 tests and a fake 50-tool server (python -m pytest atr/tests -c /dev/null)
benchmarks/bench_atr.py       the savings table above
gen-context.config.json       settings for SigMap, which runs behind ATR (only the tools ATR doesn't have)
```

---

## Build plan

| Month | Goal | Done when |
|---|---|---|
| **1** ✅ | Stdio MCP proxy, compression, stash + query, BM25 tool search, CLI | 15 tests pass; 95% schema and 99%+ big-result savings measured |
| **2** ✅ v1 | **Streamable HTTP**, both directions: `atr serve --http` (token-protected) and remote downstream servers by `url`. **Still to do:** OAuth (sign-in) for remote servers, and a real test from claude.ai and ChatGPT through a tunnel | v1: transport tests + an end-to-end ATR → remote ATR → tools chain. Done when: both apps call ATR tools |
| **3** | **Session memory for tool calls**: return a short "same as r2" instead of repeating a result, send only what changed when the same tool is called again, and let the model pin which fields it cares about | Repeated calls cost less than 10% of the first |
| **4** | **Local summarizer**: a small local model summarizes long text results when cutting them would lose meaning | Summaries beat head/tail on a QA benchmark |
| **5** | Embedding search for tools (optional), per-tool compression profiles, a dashboard of tokens saved per day | Published package + write-up with before/after numbers |

