"""LangSAM segmentation tools via the icclab_summit_xl segmentation ROS node.

Triggers the segmentation_node (GroundingDINO + SAM2) by publishing a text
prompt to /segment_text and waiting for the result on /segmentation_status.
The node also publishes /segmentation_mask and /segmented_pointcloud.

Requires: ros2 launch icclab_summit_xl segmentation.launch.py
"""

import json
import time

from fastmcp import FastMCP

from perception_mcp.utils.websocket import WebSocketManager


def register_segmentation_tools(
    mcp: FastMCP,
    ws_manager: WebSocketManager,
) -> None:
    """Register LangSAM segmentation tools (via ROS node)."""

    @mcp.tool(
        description=(
            "Segment objects using LangSAM (GroundingDINO + SAM2) on local GPU.\n\n"
            "Sends a text prompt to the segmentation_node ROS node which runs\n"
            "LangSAM locally on GPU. Produces pixel-precise segmentation masks\n"
            "and a segmented point cloud.\n\n"
            "PREREQUISITE: The segmentation node must be running:\n"
            "  ros2 launch icclab_summit_xl segmentation.launch.py\n\n"
            "The node publishes:\n"
            "  - /segmentation_mask (sensor_msgs/Image) — binary mask\n"
            "  - /segmented_pointcloud (sensor_msgs/PointCloud2) — filtered 3D points\n"
            "  - /segmentation_status (std_msgs/String) — SUCCESS, ERROR, etc.\n\n"
            "Example usage:\n"
            "- segment_objects(prompt='scissors')\n"
            "- segment_objects(prompt='red cube')\n"
            "- segment_objects(prompt='bottle')\n"
        ),
    )
    def segment_objects(prompt: str, timeout: float = 30.0) -> dict:
        """Segment objects by sending a text prompt to the LangSAM ROS node.

        Args:
            prompt: Text description of objects to find (e.g. 'scissors', 'cube').
            timeout: Max seconds to wait for segmentation result (default 30s,
                     first call may take longer due to model loading).

        Returns:
            dict with segmentation status and instructions for accessing results.
        """
        # Publish the text prompt to /segment_text
        try:
            ws = ws_manager._ensure_connected()
            pub_msg = {
                "op": "publish",
                "topic": "/segment_text",
                "type": "std_msgs/msg/String",
                "msg": {"data": prompt},
            }
            ws.send(json.dumps(pub_msg))
        except Exception as e:
            return {"error": f"Failed to publish prompt: {str(e)}"}

        # Wait for status response on /segmentation_status
        try:
            status_msg = ws_manager.subscribe_once(
                "/segmentation_status",
                msg_type="std_msgs/msg/String",
                timeout=timeout,
            )
            status = status_msg.get("data", "UNKNOWN")
        except TimeoutError:
            return {
                "error": f"Timeout ({timeout}s) waiting for segmentation result. "
                "Is the segmentation node running? Launch it with: "
                "ros2 launch icclab_summit_xl segmentation.launch.py"
            }
        except Exception as e:
            return {"error": f"Failed to get status: {str(e)}"}

        if status == "SUCCESS":
            return {
                "status": "SUCCESS",
                "prompt": prompt,
                "outputs": {
                    "mask_topic": "/segmentation_mask",
                    "pointcloud_topic": "/segmented_pointcloud",
                },
                "description": (
                    f"Segmentation of '{prompt}' completed. "
                    "The binary mask is published on /segmentation_mask "
                    "and the filtered point cloud on /segmented_pointcloud."
                ),
            }
        elif status == "NO_OBJECTS_FOUND":
            return {
                "status": "NO_OBJECTS_FOUND",
                "prompt": prompt,
                "description": f"No objects matching '{prompt}' were found in the current camera view.",
            }
        else:
            return {
                "status": status,
                "prompt": prompt,
                "description": f"Segmentation returned status: {status}",
            }
