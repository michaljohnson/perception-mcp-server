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

# Remote SAM3 segmentation server (used by segmentation_remote nodes,
# not by perception MCP directly — but checked at startup so we surface
# the failure here instead of making segment_objects time out silently).
SAM3_REMOTE_URL = os.environ.get("SAM3_REMOTE_URL", "http://160.85.252.39:8001")

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


def _startup_health_check():
    """Verify all perception MCP runtime dependencies.

    Checks: rosbridge, tf2_buffer_server, vision API key, remote SAM3 server.
    Logs loud, actionable errors to stderr — server still starts so any tools
    that don't need the missing dependency can still run.
    """
    import urllib.error
    import urllib.request
    import websocket as _ws_lib

    # 1. Rosbridge (required for ALL perception tools)
    rosbridge_ok = False
    try:
        _conn = _ws_lib.create_connection(
            f"ws://{ROSBRIDGE_IP}:{ROSBRIDGE_PORT}", timeout=3.0
        )
        _conn.close()
        rosbridge_ok = True
        print(f"[health] rosbridge OK ({ROSBRIDGE_IP}:{ROSBRIDGE_PORT})", file=sys.stderr)
    except Exception as e:
        print(
            f"[health] ERROR: rosbridge NOT reachable at "
            f"ws://{ROSBRIDGE_IP}:{ROSBRIDGE_PORT} ({e}).\n"
            f"        All perception tool calls will fail. Start it with:\n"
            f"          ros2 launch rosbridge_server rosbridge_websocket_launch.xml",
            file=sys.stderr,
        )

    # 2. tf2_buffer_server (required for grasp + drop-pose tools)
    if rosbridge_ok:
        try:
            result = ws_manager.send_action_goal(
                action_name="/tf2_buffer_server",
                action_type="tf2_msgs/action/LookupTransform",
                goal={
                    "target_frame": "base_footprint",
                    "source_frame": "base_link",
                    "source_time": {"sec": 0, "nanosec": 0},
                    "timeout": {"sec": 1, "nanosec": 0},
                    "advanced": False,
                },
                timeout=4.0,
            )
            if result.get("transform"):
                print("[health] tf2_buffer_server OK", file=sys.stderr)
            else:
                raise RuntimeError(f"unexpected response: {result}")
        except Exception as e:
            print(
                f"[health] ERROR: tf2_buffer_server NOT responding ({e}).\n"
                f"        get_grasp_from_pointcloud will return grasp_pose=None.\n"
                f"        Pick agents typically misread this as segmentation failure.\n"
                f"        Start it with:\n"
                f"          ros2 run tf2_ros buffer_server",
                file=sys.stderr,
            )
    else:
        print(
            "[health] Skipping tf2_buffer_server check (needs rosbridge).",
            file=sys.stderr,
        )

    # 3. Vision API key (required for describe_scene + detect_objects)
    if VISION_BACKEND == "anthropic":
        if not VISION_API_KEY or VISION_API_KEY == "dummy" or not VISION_API_KEY.startswith("sk-"):
            print(
                "[health] ERROR: VISION_BACKEND=anthropic but no valid API key found.\n"
                "        describe_scene and detect_objects will fail with 401.\n"
                "        Set ANTHROPIC_API_KEY (or VISION_API_KEY) in the\n"
                "        perception-mcp-server environment / .env file.",
                file=sys.stderr,
            )
        else:
            print(
                f"[health] vision API key OK (anthropic, sk-...{VISION_API_KEY[-4:]})",
                file=sys.stderr,
            )
    elif VISION_BACKEND == "openai":
        if not VISION_API_KEY or VISION_API_KEY == "dummy":
            print(
                f"[health] WARN: VISION_BACKEND=openai with no API key set.\n"
                f"        OK for endpoints that don't auth (e.g. ZHAW Qwen3-VL),\n"
                f"        but will 401 on real OpenAI / paid endpoints.\n"
                f"        Endpoint: {VISION_BASE_URL}",
                file=sys.stderr,
            )
        else:
            print(
                f"[health] vision API key OK (openai, {VISION_BASE_URL})",
                file=sys.stderr,
            )

    # 4. Remote SAM3 server (used by segmentation_remote launches —
    #    not directly by this server, but failure surfaces as silent
    #    segment_objects timeouts, so check it here.)
    try:
        req = urllib.request.Request(SAM3_REMOTE_URL, method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            print(
                f"[health] SAM3 remote OK ({SAM3_REMOTE_URL}, HTTP {resp.status})",
                file=sys.stderr,
            )
    except urllib.error.HTTPError as e:
        # 404/etc are fine — server is up, just no root handler
        print(
            f"[health] SAM3 remote OK ({SAM3_REMOTE_URL}, HTTP {e.code})",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"[health] ERROR: SAM3 remote NOT reachable at {SAM3_REMOTE_URL} ({e}).\n"
            f"        segment_objects will hang or time out.\n"
            f"        Check ZHAW server status / SSH tunnel / VPN.\n"
            f"        Override with SAM3_REMOTE_URL env var if running locally.",
            file=sys.stderr,
        )


def main():
    """Main entry point for the MCP server."""
    args = parse_arguments()

    print(f"Vision backend: {VISION_BACKEND}", file=sys.stderr)
    if VISION_BACKEND == "openai":
        print(f"Vision base URL: {VISION_BASE_URL}", file=sys.stderr)
        print(f"Vision model: {VISION_MODEL or '(default)'}", file=sys.stderr)
    print("LangSAM: available via segmentation_node (icclab_summit_xl)", file=sys.stderr)
    print(f"Rosbridge: {ROSBRIDGE_IP}:{ROSBRIDGE_PORT}", file=sys.stderr)

    _startup_health_check()

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
