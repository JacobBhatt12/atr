"""GitHub's official MCP server: tokens with and without ATR in front. Token stays local (sent only to GitHub)."""
import asyncio, json, subprocess, sys
from atr.hosted import HostedServer

URL = "https://api.githubcopilot.com/mcp/"
TOKEN = subprocess.run(["/opt/homebrew/bin/gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
H = {"Authorization": f"Bearer {TOKEN}"}
tok = lambda s: len(s) // 4
REPO = {"owner": "modelcontextprotocol", "repo": "python-sdk"}
TASKS = [  # (tool, args, question)
    ("list_issues", {**REPO, "state": "OPEN", "perPage": 30}, "Which open issues are about OAuth or authentication?"),
    ("get_file_contents", {**REPO, "path": "README.md"}, "What is the install command and minimum Python version?"),
    ("list_pull_requests", {**REPO, "state": "open", "perPage": 30}, "Which open pull requests touch streamable HTTP?"),
    ("search_repositories", {"query": "mcp server language:python stars:>500"}, "Which repositories have the most stars?"),
]

def text_of(res):
    return "\n".join(getattr(c, "text", "") or json.dumps(c.model_dump()) for c in res.content)

async def raw():
    srv = HostedServer({"github": {"url": URL, "headers": H}})
    await srv.start()
    try:
        d = srv.downstreams["github"]
        if d.error: sys.exit("github: " + d.error)
        defs = [dict(t) for t in d.tools]
        out = {"tools": len(defs), "defs": tok(json.dumps(defs)), "names": sum(len(t["name"]) // 4 + 3 for t in defs), "results": {}}
        for name, args, _ in TASKS:
            res = await d.call_tool(name, args)
            out["results"][name] = tok("\n".join(c.get("text", "") or (c.get("resource") or {}).get("text", "") for c in res.get("content", [])))
        return out
    finally:
        await srv.close()

async def via_atr():
    srv = HostedServer({"github": {"url": URL, "headers": H}})
    await srv.start()
    try:
        tools = srv.list_tools()
        out = {"tools": len(tools), "defs": tok(json.dumps(tools)), "names": [t["name"] for t in tools], "results": {}, "samples": {}}
        find = await srv.call("atr_find_tools", {"query": "list open issues in a repository"})
        ft = find["content"][0]["text"]; out["find_tokens"] = tok(ft); out["samples"]["find"] = ft[:600]
        for name, args, q in TASKS:
            r = await srv.call("atr_read", {"tool": f"github__{name}", "arguments": args, "question": q})
            t = "\n".join(c.get("text", "") for c in r["content"])
            out["results"][name] = tok(t); out["samples"][name] = t[:900]
        return out
    finally:
        await srv.close()

async def main():
    a = await raw(); b = await via_atr()
    print(json.dumps({"raw": a, "atr": b}, indent=1))
asyncio.run(main())
