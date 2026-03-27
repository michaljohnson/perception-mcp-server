"""Perception MCP Server - MCP instance and main entry point.

Supports two vision backends:
- "anthropic": Claude Vision API (requires ANTHROPIC_API_KEY)
- "openai": OpenAI-compatible API like ZHAW Qwen3-VL (requires OPENAI_BASE_URL)
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

# Load .env from the server root directory
load_dotenv(Path(__file__).parent.parent / ".env")

from perception_mcp.tools import register_all_tools
from perception_mcp.utils.websocket import WebSocketManager

# Connection settings
ROSBRIDGE_IP = os.environ.get("ROSBRIDGE_IP", "127.0.0.1")
ROSBRIDGE_PORT = int(os.environ.get("ROSBRIDGE_PORT", "9090"))

# Vision backend settings
VISION_BACKEND = os.environ.get("VISION_BACKEND", "openai")  # "anthropic" or "openai"
VISION_API_KEY = os.environ.get("VISION_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "dummy"))
VISION_MODEL = os.environ.get("VISION_MODEL", "")  # Empty = use default per backend
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "http://127.0.0.1:8000/v1")

# Camera topics per camera
CAMERA_TOPICS = {
    "front": {
        "rgb": "/front_rgbd_camera/color/image_raw/compressed",
        "depth": "/front_rgbd_camera/depth/image_raw",
        "camera_info": "/front_rgbd_camera/depth/camera_info",
    },
    "arm": {
        "rgb": "/arm_camera/color/image_raw/compressed",
        "depth": "/arm_camera/depth/image_raw",
        "camera_info": "/arm_camera/depth/camera_info",
    },
}

# Initialize MCP server
mcp = FastMCP("perception-mcp-server")

# Initialize WebSocket manager for rosbridge
ws_manager = WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT, default_timeout=10.0)

# Register all tools
register_all_tools(
    mcp,
    ws_manager,
    vision_backend=VISION_BACKEND,
    vision_api_key=VISION_API_KEY,
    vision_model=VISION_MODEL,
    vision_base_url=VISION_BASE_URL,
    camera_topics=CAMERA_TOPICS,
)


def parse_arguments():
    """Parse command line arguments for MCP server configuration."""
    parser = argparse.ArgumentParser(
        description="Perception MCP Server - Object detection and grasp planning for robots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using ZHAW Qwen3-VL (default, free)
  python server.py --transport streamable-http --port 8003

  # Using Claude Vision API
  VISION_BACKEND=anthropic VISION_API_KEY=sk-ant-... python server.py

  # Custom OpenAI-compatible endpoint
  VISION_BACKEND=openai VISION_BASE_URL=http://localhost:11434/v1 python server.py
        """,
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for HTTP transports (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8003,
        help="Port for HTTP transports (default: 8003)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the MCP server."""
    args = parse_arguments()

    print(f"Vision backend: {VISION_BACKEND}", file=sys.stderr)
    if VISION_BACKEND == "openai":
        print(f"Vision base URL: {VISION_BASE_URL}", file=sys.stderr)
        print(f"Vision model: {VISION_MODEL or '(default)'}", file=sys.stderr)
    print("LangSAM: available via segmentation_node (icclab_summit_xl)", file=sys.stderr)
    print(f"Rosbridge: {ROSBRIDGE_IP}:{ROSBRIDGE_PORT}", file=sys.stderr)

    mcp_transport = args.transport.lower()

    if mcp_transport == "stdio":
        mcp.run(transport="stdio")
    elif mcp_transport in {"http", "streamable-http"}:
        print(
            f"Transport: {mcp_transport} -> http://{args.host}:{args.port}",
            file=sys.stderr,
        )
        mcp.run(transport=mcp_transport, host=args.host, port=args.port)
    else:
        raise ValueError(f"Unsupported transport: {mcp_transport}")


if __name__ == "__main__":
    main()
