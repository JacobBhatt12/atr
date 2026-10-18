"""python -m cloud.agent --control-plane http://cp:7070 [--advertise http://this-host:7071]"""
import argparse

import uvicorn

from cloud.agent.agent import create_agent

p = argparse.ArgumentParser(description="VRTX worker agent")
p.add_argument("--control-plane", default="http://127.0.0.1:7070")
p.add_argument("--host", default="0.0.0.0")
p.add_argument("--port", type=int, default=7071)
p.add_argument("--advertise", help="URL the control plane and gateway use to reach this node (default http://127.0.0.1:PORT)")
p.add_argument("--name", help="node id (default: hostname)")
p.add_argument("--runtime", default="auto", choices=["auto", "docker", "process"])
args = p.parse_args()
app = create_agent(args.control_plane, args.name, args.runtime, args.advertise or f"http://127.0.0.1:{args.port}")
uvicorn.run(app, host=args.host, port=args.port)
