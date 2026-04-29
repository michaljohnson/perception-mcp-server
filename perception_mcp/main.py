"""Perception MCP Server - MCP instance and main entry point."""

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

# Remote SAM3 segmentation server (used by segmentation_remote nodes,
# not by perception MCP directly — but checked at startup so we surface
# the failure here instead of making segment_objects time out silently).
SAM3_REMOTE_URL = os.environ.get("SAM3_REMOTE_URL", "")

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
register_all_tools(mcp, ws_manager, camera_topics=CAMERA_TOPICS)


def parse_arguments():
    """Parse command line arguments for MCP server configuration."""
    parser = argparse.ArgumentParser(
        description="Perception MCP Server - segmentation, grasp / place pose, raw camera access for robots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py --transport streamable-http --port 8003
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

    Checks: rosbridge, tf2_buffer_server, remote SAM3 server.
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

    # 2. tf2_buffer_server (required for grasp + place-pose tools)
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
                f"        get_topdown_grasp_pose will return grasp_pose=None.\n"
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

    # 3. Remote SAM3 server (used by segmentation_remote launches —
    #    not directly by this server, but failure surfaces as silent
    #    segment_objects timeouts, so check it here when configured.)
    if not SAM3_REMOTE_URL:
        print(
            "[health] SAM3_REMOTE_URL not set; skipping segmentation backend "
            "check. Set it in .env to enable.",
            file=sys.stderr,
        )
    else:
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
                f"[health] ERROR: SAM3 remote NOT reachable at "
                f"{SAM3_REMOTE_URL} ({e}).\n"
                f"        segment_objects will hang or time out.\n"
                f"        Verify the segmentation backend is running and\n"
                f"        reachable from this host.",
                file=sys.stderr,
            )


def main():
    """Main entry point for the MCP server."""
    args = parse_arguments()

    print("Segmentation: relies on an external SAM3 ROS node (see README).", file=sys.stderr)
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
