"""Perception MCP Tools - Tool implementations organized by category."""

from fastmcp import FastMCP

from perception_mcp.tools.detection import register_detection_tools
from perception_mcp.tools.grasping import register_grasping_tools
from perception_mcp.tools.placing import register_placing_tools
from perception_mcp.tools.segmentation import register_segmentation_tools
from perception_mcp.utils.websocket import WebSocketManager


def register_all_tools(
    mcp: FastMCP,
    ws_manager: WebSocketManager,
    camera_topics: dict = None,
) -> None:
    """Register all perception tools with the provided FastMCP instance."""

    if camera_topics is None:
        camera_topics = {
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

    # Shared cache for segmentation results (pointcloud, etc.)
    segmentation_cache = {}

    register_detection_tools(mcp, ws_manager, camera_topics)
    register_grasping_tools(mcp, ws_manager, segmentation_cache)
    register_placing_tools(mcp, ws_manager, segmentation_cache)
    register_segmentation_tools(mcp, ws_manager, segmentation_cache)
