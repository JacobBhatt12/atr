"""ATR: Automation for Token Reduction.

An MCP proxy that sits between an AI client (Claude Desktop, Claude Code, Cursor, Zed, ChatGPT
connectors) and your other MCP servers. It hides tool schemas until they are needed and
compresses every tool result before the model sees it.
"""
__version__ = "0.2.1"
