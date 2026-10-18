"""Cluster state: deployments, builds, nodes and replicas in SQLite.

Every write goes through `ClusterState.apply(command)`, where a command is a plain dict
like {"op": "replica.patch", "id": ..., "fields": {...}}. Commands carry their own ids
and timestamps, so applying the same list of commands to an empty database always gives
the same state. That is the property Raft needs in Month 6: the leader appends commands
to a replicated log and every node applies them in order. Reads go straight to SQLite.
"""
from __future__ import annotations

import json
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS deployments (
    name TEXT PRIMARY KEY, type TEXT NOT NULL, spec TEXT NOT NULL,
    artifact TEXT, version INTEGER NOT NULL DEFAULT 0, build_id TEXT,
    failures INTEGER NOT NULL DEFAULT 0, next_attempt_at REAL NOT NULL DEFAULT 0,
    last_error TEXT, deleting INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS builds (
    id TEXT PRIMARY KEY, deployment TEXT NOT NULL, status TEXT NOT NULL,
    source_hash TEXT, artifact TEXT, error TEXT, cached INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL, finished_at REAL
);
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY, url TEXT NOT NULL, runtime TEXT NOT NULL,
    capacity TEXT NOT NULL, usage TEXT NOT NULL DEFAULT '{}',
    registered_at REAL NOT NULL, last_seen REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS replicas (
    id TEXT PRIMARY KEY, deployment TEXT NOT NULL, version INTEGER NOT NULL,
    node TEXT NOT NULL, state TEXT NOT NULL, port INTEGER, url TEXT,
    error TEXT, exit_code INTEGER, resources TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL, updated_at REAL NOT NULL, running_since REAL
);
"""

JSON_COLUMNS = {"spec", "artifact", "capacity", "usage", "resources"}
PATCHABLE = {
    "deployments": {"artifact", "version", "build_id", "failures", "next_attempt_at", "last_error", "deleting"},
    "builds": {"status", "artifact", "error", "cached", "finished_at"},
    "replicas": {"state", "port", "url", "error", "exit_code", "running_since"},
}


class ClusterState:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()

    # ---------------------------------------------------------------- writes

    def apply(self, cmd: dict) -> None:
        """Apply one command. The only way state changes."""
        op = cmd["op"]
        with self.lock:
            self.db.execute("BEGIN")
            try:
                getattr(self, "_" + op.replace(".", "_"))(cmd)
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def _deployment_put(self, c):
        spec, ts = c["spec"], c["ts"]
        self.db.execute(
            """INSERT INTO deployments (name, type, spec, created_at, updated_at) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET spec=excluded.spec, type=excluded.type, updated_at=excluded.updated_at,
               failures=0, next_attempt_at=0, deleting=0""",
            (spec["name"], spec["type"], json.dumps(spec), ts, ts),
        )

    def _deployment_patch(self, c):
        fields = dict(c.get("fields", {}))
        if "spec" in c:  # partial spec update, e.g. {"replicas": 3}
            row = self.db.execute("SELECT spec FROM deployments WHERE name=?", (c["name"],)).fetchone()
            fields["spec"] = json.loads(row["spec"]) | c["spec"]
        self._patch("deployments", "name", c["name"], fields, c["ts"], extra={"spec"})

    def _deployment_remove(self, c):
        self.db.execute("DELETE FROM deployments WHERE name=?", (c["name"],))

    def _build_put(self, c):
        b = c["build"]
        self.db.execute(
            "INSERT INTO builds (id, deployment, status, source_hash, created_at) VALUES (?, ?, 'building', ?, ?)",
            (b["id"], b["deployment"], b.get("source_hash"), c["ts"]),
        )

    def _build_patch(self, c):
        self._patch("builds", "id", c["id"], c["fields"], None)

    def _node_register(self, c):
        n = c["node"]
        self.db.execute(
            """INSERT INTO nodes (id, url, runtime, capacity, registered_at, last_seen) VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET url=excluded.url, runtime=excluded.runtime,
               capacity=excluded.capacity, last_seen=excluded.last_seen""",
            (n["id"], n["url"], n["runtime"], json.dumps(n["capacity"]), c["ts"], c["ts"]),
        )

    def _node_seen(self, c):
        self.db.execute("UPDATE nodes SET usage=?, last_seen=? WHERE id=?", (json.dumps(c["usage"]), c["ts"], c["id"]))

    def _replica_put(self, c):
        r = c["replica"]
        self.db.execute(
            """INSERT INTO replicas (id, deployment, version, node, state, resources, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'scheduled', ?, ?, ?)""",
            (r["id"], r["deployment"], r["version"], r["node"], json.dumps(r.get("resources", {})), c["ts"], c["ts"]),
        )

    def _replica_patch(self, c):
        self._patch("replicas", "id", c["id"], c["fields"], c["ts"])

    def _replica_remove(self, c):
        self.db.execute("DELETE FROM replicas WHERE id=?", (c["id"],))

    def _patch(self, table, key, value, fields, ts, extra=frozenset()):
        bad = set(fields) - PATCHABLE[table] - set(extra)
        if bad:
            raise ValueError(f"cannot patch {table}.{', '.join(sorted(bad))}")
        cols = {k: json.dumps(v) if k in JSON_COLUMNS and v is not None else v for k, v in fields.items()}
        if ts is not None:
            cols["updated_at"] = ts
        if not cols:
            return
        sets = ", ".join(f"{k}=?" for k in cols)
        self.db.execute(f"UPDATE {table} SET {sets} WHERE {key}=?", (*cols.values(), value))

    # ----------------------------------------------------------------- reads

    def _rows(self, sql, args=()):
        with self.lock:
            rows = self.db.execute(sql, args).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            for k in JSON_COLUMNS & d.keys():
                d[k] = json.loads(d[k]) if d[k] is not None else None
            out.append(d)
        return out

    def deployments(self) -> list[dict]:
        return self._rows("SELECT * FROM deployments ORDER BY name")

    def deployment(self, name: str) -> dict | None:
        rows = self._rows("SELECT * FROM deployments WHERE name=?", (name,))
        return rows[0] if rows else None

    def build(self, build_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM builds WHERE id=?", (build_id,))
        return rows[0] if rows else None

    def builds(self, deployment: str) -> list[dict]:
        return self._rows("SELECT * FROM builds WHERE deployment=? ORDER BY created_at DESC", (deployment,))

    def nodes(self) -> list[dict]:
        return self._rows("SELECT * FROM nodes ORDER BY id")

    def node(self, node_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM nodes WHERE id=?", (node_id,))
        return rows[0] if rows else None

    def replicas(self, deployment: str | None = None, node: str | None = None) -> list[dict]:
        if deployment:
            return self._rows("SELECT * FROM replicas WHERE deployment=? ORDER BY created_at", (deployment,))
        if node:
            return self._rows("SELECT * FROM replicas WHERE node=? ORDER BY created_at", (node,))
        return self._rows("SELECT * FROM replicas ORDER BY created_at")

    def replica(self, replica_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM replicas WHERE id=?", (replica_id,))
        return rows[0] if rows else None
