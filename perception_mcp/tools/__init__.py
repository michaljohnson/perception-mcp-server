"""Perception MCP Tools - Tool implementations organized by category."""

from fastmcp import FastMCP

from perception_mcp.tools.detection import register_detection_tools
from perception_mcp.tools.localization import register_localization_tools
from perception_mcp.tools.grasping import register_grasping_tools
from perception_mcp.tools.segmentation import register_segmentation_tools
from perception_mcp.utils.vision import create_vision_client
from perception_mcp.utils.websocket import WebSocketManager


def register_all_tools(
    mcp: FastMCP,
    ws_manager: WebSocketManager,
    vision_backend: str = "openai",
    vision_api_key: str = "",
    vision_model: str = "",
    vision_base_url: str = "",
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

    # Create vision client once, shared by all tools
    vision_client = create_vision_client(
        backend=vision_backend,
        api_key=vision_api_key,
        model=vision_model,
        base_url=vision_base_url,
    )

    register_detection_tools(mcp, ws_manager, vision_client, camera_topics)
    register_localization_tools(mcp, ws_manager, vision_client, camera_topics)
    register_grasping_tools(mcp, ws_manager, vision_client, camera_topics)
    register_segmentation_tools(mcp, ws_manager)
