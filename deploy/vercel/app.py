"""Vercel entrypoint for ATR's online connector (deploy.py copies this next to the atr package)."""
from atr.hosted import create_hosted

app = create_hosted()
