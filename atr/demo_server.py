"""A stand-in MCP server with 50 tools and bloated results, for `atr serve --demo`, tests and the benchmark.

Run directly: python -m atr.demo_server   (stdio, newline-delimited JSON-RPC)
"""
from __future__ import annotations

import json
import random
import sys

AREAS = {
    "github": ["create_issue", "list_issues", "get_issue", "update_issue", "add_issue_comment", "list_pull_requests",
               "create_pull_request", "merge_pull_request", "get_file_contents", "search_code", "list_commits",
               "create_branch", "list_repositories"],
    "db": ["query", "list_tables", "describe_table", "insert_rows", "update_rows", "delete_rows", "export_csv"],
    "slack": ["send_message", "list_channels", "read_channel_history", "add_reaction", "search_messages",
              "get_user_profile"],
    "calendar": ["list_events", "create_event", "delete_event", "find_free_time", "list_calendars"],
    "files": ["read_file", "write_file", "list_directory", "move_file", "search_files", "get_file_info"],
    "web": ["fetch_url", "search_web", "screenshot_page"],
    "logs": ["tail_logs", "search_logs", "list_services", "get_metrics", "list_alerts", "ack_alert",
             "get_deploy_history", "rollback_deploy", "restart_service", "get_service_health"],
}
VERB_DESC = {"create": "Create a new", "list": "List all", "get": "Get one", "update": "Update an existing",
             "delete": "Delete a", "search": "Search for", "add": "Add a", "send": "Send a", "read": "Read the",
             "merge": "Merge a", "find": "Find", "query": "Run a SQL query against", "insert": "Insert rows into",
             "export": "Export data as", "describe": "Describe the columns of", "tail": "Stream the latest",
             "ack": "Acknowledge an", "rollback": "Roll back a", "restart": "Restart a", "fetch": "Fetch the content of",
             "screenshot": "Take a screenshot of", "move": "Move or rename a", "write": "Write content to"}


def tools() -> list[dict]:
    out = []
    for area, names in AREAS.items():
        for n in names:
            verb, _, noun = n.partition("_")
            props = {
                "owner": {"type": "string", "title": "Owner", "description": "The account owner of the repository. The name is not case sensitive. Example: 'octocat'."},
                "repo": {"type": "string", "title": "Repo", "description": "The name of the repository without the .git extension. The name is not case sensitive."},
                "limit": {"type": "integer", "title": "Limit", "default": 30, "description": "Maximum number of results to return per page. Must be between 1 and 100. Defaults to 30 when omitted."},
                "filter": {"type": "string", "title": "Filter", "description": "Optional filter expression applied server side before results are returned to the caller."},
            } if area == "github" else {
                "target": {"type": "string", "title": "Target", "description": f"The {noun.replace('_', ' ')} to operate on. Accepts an id or a name; ids are faster."},
                "options": {"type": "object", "title": "Options", "description": "Additional options. See the server documentation for the full list of accepted keys and their meaning.",
                            "additionalProperties": True},
            }
            reads = verb in ("list", "get", "search", "read", "describe", "find", "tail", "fetch", "query", "screenshot")
            out.append({
                "annotations": {"readOnlyHint": reads, "destructiveHint": verb in ("delete", "rollback", "merge")},
                "name": f"{area}_{n}",
                "description": f"{VERB_DESC.get(verb, verb.title())} {noun.replace('_', ' ')} in {area}. "
                               f"This tool talks to the {area} API on behalf of the user and returns the raw API "
                               f"response as JSON. Rate limits apply; on HTTP 429 wait and retry. "
                               f"Results may be paginated, in which case pass the cursor from the previous call.",
                "inputSchema": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
                                "title": f"{area}_{n}_args", "properties": props,
                                "required": ["owner", "repo"] if area == "github" else ["target"],
                                "additionalProperties": False},
            })
    return out


def issues(n: int = 1000) -> list[dict]:
    rnd = random.Random(7)
    users = ["octocat", "hubot", "monalisa", "defunkt", "mojombo"]
    out = []
    for i in range(1, n + 1):
        u = rnd.choice(users)
        out.append({
            "id": 100000 + i, "node_id": f"MDU6SXNzdWU{i:08d}", "number": i,
            "url": f"https://api.github.com/repos/acme/app/issues/{i}",
            "html_url": f"https://github.com/acme/app/issues/{i}",
            "comments_url": f"https://api.github.com/repos/acme/app/issues/{i}/comments",
            "title": f"Bug {i}: {rnd.choice(['crash on login', 'slow dashboard', 'typo in docs', 'memory leak', 'flaky test'])}",
            "state": rnd.choice(["open", "open", "closed"]), "locked": False,
            "user": {"login": u, "id": hash(u) % 10000, "node_id": "MDQ6VXNlcjE=", "avatar_url": f"https://avatars.githubusercontent.com/{u}",
                     "gravatar_id": "", "type": "User", "site_admin": False},
            "labels": [{"name": rnd.choice(["bug", "ui", "perf"]), "color": "d73a4a", "default": False, "description": None}],
            "assignee": None, "assignees": [], "milestone": None, "comments": rnd.randint(0, 12),
            "created_at": f"2026-0{rnd.randint(1, 9)}-1{rnd.randint(0, 9)}T10:00:00Z", "closed_at": None,
            "author_association": "CONTRIBUTOR", "active_lock_reason": None, "body": None,
            "reactions": {"url": f"https://api.github.com/repos/acme/app/issues/{i}/reactions", "total_count": 0,
                          "+1": 0, "-1": 0, "laugh": 0, "hooray": 0, "confused": 0, "heart": 0, "rocket": 0, "eyes": 0},
            "performed_via_github_app": None, "state_reason": None,
        })
    return out


def logs(n: int = 2000) -> str:
    lines = []
    for i in range(n):
        lines.append(f"2026-10-21T16:{i // 60 % 60:02d}:{i % 60:02d}.{i % 1000:03d}Z INFO  worker-3 heartbeat ok seq={i} latency_ms={i % 17}")
        if i == n - 40:
            lines += ["2026-10-21T16:59:20.000Z ERROR worker-3 request failed: connection reset by peer",
                      "Traceback (most recent call last):", '  File "app/handler.py", line 88, in handle',
                      "    resp = await upstream.send(req)", "ConnectionResetError: [Errno 54] Connection reset by peer"]
    return "\n".join(lines)


def call(name: str, args: dict) -> dict:
    if name == "github_list_issues":
        data = issues()
        return {"content": [{"type": "text", "text": json.dumps(data, indent=2)}], "structuredContent": {"items": data}}
    if name == "logs_tail_logs":
        return {"content": [{"type": "text", "text": logs()}]}
    if name == "github_create_issue":
        return {"content": [{"type": "text", "text": json.dumps({"number": 1001, "state": "open", "title": args.get("filter", "new")})}]}
    if name == "web_fetch_url":
        body = "".join(f"<div class='c'><p>Paragraph {i} of the article.</p></div>" for i in range(50))
        return {"content": [{"type": "text", "text": f"<html><head><style>.c{{color:red}}</style><script>var x=1;</script></head><body>{body}</body></html>"}]}
    return {"content": [{"type": "text", "text": json.dumps({"ok": True, "tool": name, "args": args})}]}


def main() -> None:
    all_tools = tools()
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:
            continue
        if method == "initialize":
            res = {"protocolVersion": params.get("protocolVersion"), "capabilities": {"tools": {}},
                   "serverInfo": {"name": "fake", "version": "1"}}
        elif method == "tools/list":  # two pages, to exercise cursors
            start = int(params.get("cursor") or 0)
            res = {"tools": all_tools[start:start + 25]}
            if start + 25 < len(all_tools):
                res["nextCursor"] = str(start + 25)
        elif method == "tools/call":
            res = call(params["name"], params.get("arguments") or {})
        else:
            res = {}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": res}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
