"""LangSAM segmentation tools via the icclab_summit_xl segmentation ROS node.

Triggers the segmentation_node (GroundingDINO + SAM2) by publishing a text
prompt to /segment_text and waiting for the result on /segmentation_status.
The node also publishes /segmentation_mask and /segmented_pointcloud.

The segmented point cloud is cached in memory so that get_grasp_from_pointcloud
can read it without re-subscribing (avoids QoS/timing issues with rosbridge).

Requires: ros2 launch icclab_summit_xl segmentation.launch.py
"""

import json
import time
import threading
import uuid

from fastmcp import FastMCP

from perception_mcp.utils.websocket import WebSocketManager


def register_segmentation_tools(
    mcp: FastMCP,
    ws_manager: WebSocketManager,
    segmentation_cache: dict,
) -> None:
    """Register LangSAM segmentation tools (via ROS node)."""

    @mcp.tool(
        description=(
            "Segment objects using LangSAM (GroundingDINO + SAM2).\n\n"
            "Sends a text prompt to the segmentation_node ROS node which runs\n"
            "LangSAM. Produces pixel-precise segmentation masks\n"
            "and a segmented point cloud.\n\n"
            "PREREQUISITE: The segmentation node must be running:\n"
            "  ros2 launch icclab_summit_xl segmentation_remote.launch.py\n\n"
            "The segmented point cloud is cached so that\n"
            "get_grasp_from_pointcloud() can access it immediately.\n\n"
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
            dict with segmentation status and cached point cloud info.
        """
        # Start listening for the pointcloud BEFORE triggering segmentation,
        # using a dedicated websocket connection to avoid conflicts with the
        # main ws_manager connection (which will be used for status polling).
        pc_result = {"data": None, "error": None}

        def _capture_pointcloud():
            try:
                import websocket as _ws
                import base64
                import struct

                pc_ws = _ws.create_connection(ws_manager.url, timeout=timeout)
                sub_id = f"pc_capture:{uuid.uuid4().hex[:8]}"
                pc_ws.send(json.dumps({
                    "op": "subscribe",
                    "id": sub_id,
                    "topic": "/segmented_pointcloud",
                    "type": "sensor_msgs/msg/PointCloud2",
                }))

                # Wait for the message
                pc_ws.settimeout(timeout)
                while True:
                    raw = pc_ws.recv()
                    data = json.loads(raw)
                    if data.get("topic") == "/segmented_pointcloud":
                        # Parse the pointcloud inline
                        msg = data.get("msg", {})
                        points, colors, frame_id = ws_manager._parse_pointcloud(msg)
                        pc_result["data"] = (points, colors, frame_id)
                        break

                pc_ws.send(json.dumps({
                    "op": "unsubscribe",
                    "id": sub_id,
                    "topic": "/segmented_pointcloud",
                }))
                pc_ws.close()
            except Exception as e:
                pc_result["error"] = str(e)

        pc_thread = threading.Thread(target=_capture_pointcloud, daemon=True)
        pc_thread.start()

        # Give the subscriber a moment to register with rosbridge
        time.sleep(0.5)

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

        # Wait for a terminal status (skip intermediate SEGMENTING messages).
        try:
            deadline = time.monotonic() + timeout
            status = "SEGMENTING"
            while status == "SEGMENTING":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for segmentation result")
                status_msg = ws_manager.subscribe_once(
                    "/segmentation_status",
                    msg_type="std_msgs/msg/String",
                    timeout=remaining,
                )
                status = status_msg.get("data", "UNKNOWN")
        except TimeoutError:
            return {
                "error": f"Timeout ({timeout}s) waiting for segmentation result. "
                "Is the segmentation node running? Launch it with: "
                "ros2 launch icclab_summit_xl segmentation_remote.launch.py"
            }
        except Exception as e:
            return {"error": f"Failed to get status: {str(e)}"}

        if status == "SUCCESS":
            # Wait for the pointcloud capture thread to finish
            pc_thread.join(timeout=10.0)

            if pc_result["data"] is not None:
                points, colors, frame_id = pc_result["data"]
                segmentation_cache["points"] = points
                segmentation_cache["colors"] = colors
                segmentation_cache["frame_id"] = frame_id
                segmentation_cache["prompt"] = prompt
                segmentation_cache["timestamp"] = time.time()
                # Snapshot the camera→base_footprint TF at segmentation
                # time and cache it. This ensures that a later
                # get_grasp_from_pointcloud call produces a correct base-frame
                # centroid even if the base or arm moved in between: the
                # pointcloud points were captured relative to the camera pose
                # at THIS instant, and must be transformed with the TF at
                # THIS instant (not a fresh lookup at call time).
                segmentation_cache.pop("tf_translation", None)
                segmentation_cache.pop("tf_rotation", None)
                try:
                    tf_result = ws_manager.send_action_goal(
                        action_name="/tf2_buffer_server",
                        action_type="tf2_msgs/action/LookupTransform",
                        goal={
                            "target_frame": "base_footprint",
                            "source_frame": frame_id,
                            "source_time": {"sec": 0, "nanosec": 0},
                            "timeout": {"sec": 2, "nanosec": 0},
                            "advanced": False,
                        },
                        timeout=5.0,
                    )
                    tf = tf_result.get("transform", {}).get("transform", {})
                    segmentation_cache["tf_translation"] = tf.get("translation", {})
                    segmentation_cache["tf_rotation"] = tf.get("rotation", {})
                except Exception:
                    pass  # grasping tool will fall back to a fresh lookup
                pc_info = f" Point cloud cached ({len(points)} points)."
            else:
                pc_info = " Warning: point cloud not captured."

            return {
                "status": "SUCCESS",
                "prompt": prompt,
                "outputs": {
                    "mask_topic": "/segmentation_mask",
                    "pointcloud_topic": "/segmented_pointcloud",
                },
                "description": (
                    f"Segmentation of '{prompt}' completed."
                    f"{pc_info}"
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
