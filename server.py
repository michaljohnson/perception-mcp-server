"""Entry point for Perception MCP Server.

Usage:
    python server.py
    python server.py --transport streamable-http --host 0.0.0.0 --port 8003
"""

from perception_mcp.main import main

if __name__ == "__main__":
    main()
