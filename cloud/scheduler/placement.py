"""Where a new replica goes.

Month 5 is one good-enough rule: filter the nodes that can run the artifact and have room,
then prefer spreading replicas of one deployment, then the most free GPU memory.
Month 6 replaces `place` with First-Fit Decreasing / Best-Fit / a scored heuristic,
measured against an ILP optimum.

Room is counted from reservations (the resources each replica's spec asks for),
not from live usage, so a replica that is still loading weights already holds its share.
"""
from __future__ import annotations

from cloud.common import NODE_TIMEOUT_S

RUNTIME_FOR_ARTIFACT = {"image": "docker", "dir": "process", "engine": "process"}


def node_alive(node: dict, now: float) -> bool:
    return now - node["last_seen"] <= NODE_TIMEOUT_S


def reserved(node_id: str, replicas: list[dict]) -> dict:
    total = {"gpu_memory_gb": 0.0, "cpu": 0.0, "memory_mb": 0}
    for r in replicas:
        if r["node"] == node_id:
            for k in total:
                total[k] += r["resources"].get(k, 0)
    return total


def free(node: dict, replicas: list[dict]) -> dict:
    cap, used = node["capacity"], reserved(node["id"], replicas)
    return {
        "gpu_memory_gb": cap.get("gpu_memory_mb", 0) / 1024 - used["gpu_memory_gb"],
        "cpu": cap.get("cpus", 0) - used["cpu"],
        "memory_mb": cap.get("memory_mb", 0) - used["memory_mb"],
    }


def place(spec: dict, artifact: dict, nodes: list[dict], replicas: list[dict], now: float) -> tuple[dict | None, str]:
    """Return (node, "") or (None, reason it could not be placed)."""
    want = spec["resources"]
    runtime = RUNTIME_FOR_ARTIFACT[artifact["kind"]]
    alive = [n for n in nodes if node_alive(n, now)]
    if not alive:
        return None, "no live worker nodes (start one with `vrtx up` or `python -m cloud.agent`)"
    compatible = [n for n in alive if n["runtime"] == runtime]
    if not compatible:
        return None, f"no live node runs {runtime} artifacts"

    fits = []
    for n in compatible:
        room = free(n, replicas)
        if want["gpu_memory_gb"] > room["gpu_memory_gb"] + 1e-9:
            continue
        if want["cpu"] and want["cpu"] > room["cpu"]:
            continue
        if want["memory_mb"] and want["memory_mb"] > room["memory_mb"]:
            continue
        fits.append((n, room))
    if not fits:
        return None, (
            f"no node has room for gpu_memory_gb={want['gpu_memory_gb']:g}, cpu={want['cpu']:g}, "
            f"memory_mb={want['memory_mb']}"
        )

    def score(item):
        n, room = item
        same = sum(1 for r in replicas if r["node"] == n["id"] and r["deployment"] == spec["name"])
        total = sum(1 for r in replicas if r["node"] == n["id"])
        return (same, -room["gpu_memory_gb"], total, n["id"])

    return min(fits, key=score)[0], ""
