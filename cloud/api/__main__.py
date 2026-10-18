"""python -m cloud.api [--port 7070] [--runtime auto|docker|process]"""
import argparse

import uvicorn

from cloud.api.server import create_control_plane

p = argparse.ArgumentParser(description="VRTX control plane")
p.add_argument("--host", default="0.0.0.0")
p.add_argument("--port", type=int, default=7070)
p.add_argument("--runtime", default="auto", choices=["auto", "docker", "process"], help="what the builder produces")
p.add_argument("--gateway-port", type=int, default=8080, help="used to print app URLs")
args = p.parse_args()
uvicorn.run(create_control_plane(runtime=args.runtime, gateway_port=args.gateway_port), host=args.host, port=args.port)
