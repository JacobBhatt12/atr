"""Small helpers shared by the control plane, the agent and the CLI."""
from __future__ import annotations

import os
import re
import secrets
import socket
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAIN = os.getenv("VRTX_DOMAIN", "vrtx.local")
DEFAULT_API = "http://127.0.0.1:7070"

# How long a node may go without a heartbeat before its replicas are rescheduled.
NODE_TIMEOUT_S = float(os.getenv("VRTX_NODE_TIMEOUT_S", "10"))
HEARTBEAT_S = float(os.getenv("VRTX_HEARTBEAT_S", "2"))

_NAME = re.compile(r"^[a-z0-9]([a-z0-9-]{0,40}[a-z0-9])?$")


def vrtx_home() -> Path:
    """Where state, build contexts and logs live (default ~/.vrtx)."""
    path = Path(os.getenv("VRTX_HOME", Path.home() / ".vrtx"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def now() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(4)}"


def valid_name(name: str) -> bool:
    """Deployment names become DNS labels: name.vrtx.local."""
    return bool(_NAME.match(name))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def app_url(name: str, gateway_port: int | None = None) -> str:
    port = "" if gateway_port in (None, 80) else f":{gateway_port}"
    return f"http://{name}.{DOMAIN}{port}"
