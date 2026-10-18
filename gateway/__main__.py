"""python -m gateway --control-plane http://cp:7070 [--port 8080]: the *.vrtx.local proxy.

(The OpenAI-compatible model server is gateway/server.py: `uvicorn gateway.server:app`.)
"""
import argparse

import uvicorn

from gateway.proxy import create_proxy

p = argparse.ArgumentParser(description="VRTX gateway proxy")
p.add_argument("--control-plane", default="http://127.0.0.1:7070")
p.add_argument("--host", default="0.0.0.0")
p.add_argument("--port", type=int, default=8080)
args = p.parse_args()
uvicorn.run(create_proxy(args.control_plane), host=args.host, port=args.port)
