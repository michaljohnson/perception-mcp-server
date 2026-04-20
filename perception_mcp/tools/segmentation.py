"""SAM3 segmentation tools via the icclab_summit_xl segmentation ROS node.

Triggers a segmentation_node_remote instance by publishing a text prompt to
/segment_text (arm camera) or /front/segment_text (front camera) and waiting
for the result on /segmentation_status. The node also publishes
/segmentation_mask and /segmented_pointcloud (prefixed for the front camera).

The segmented point cloud is cached in memory so that get_grasp_from_pointcloud
can read it without re-subscribing (avoids QoS/timing issues with rosbridge).

Requires: start_ros_stack.sh launches both arm_segmentation_node and
front_segmentation_node.
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
            "Segment objects using SAM3 (remote GroundingDINO + SAM).\n\n"
            "Sends a text prompt to a segmentation_node_remote ROS instance\n"
            "and produces a pixel-precise mask plus a segmented point cloud.\n"
            "Two cameras are supported:\n"
            "  camera='arm'   (default) — arm_camera, use for grasping\n"
            "  camera='front' — front_rgbd_camera, use for approach/nav\n\n"
            "The arm camera can only see objects when the arm is lowered into\n"
            "a look-down pose; the front camera sees the scene from far away\n"
            "and is the right choice for the navigator to refine its approach\n"
            "pose while the arm stays in 'look_forward'.\n\n"
            "PREREQUISITE: both segmentation nodes must be running; they are\n"
            "started automatically by start_ros_stack.sh.\n\n"
            "The segmented point cloud is cached so that\n"
            "get_grasp_from_pointcloud() can access it immediately; the cached\n"
            "frame_id is the camera's optical frame and is transformed to\n"
            "base_footprint via TF inside get_grasp_from_pointcloud.\n\n"
            "Example usage:\n"
            "- segment_objects(prompt='red ball', camera='front')\n"
            "- segment_objects(prompt='scissors', camera='arm')\n"
            "- segment_objects(prompt='bottle')  # defaults to arm\n"
        ),
    )
    def segment_objects(
        prompt: str,
        camera: str = "arm",
        timeout: float = 30.0,
    ) -> dict:
        """Segment objects by sending a text prompt to a SAM3 ROS node.

        Args:
            prompt: Text description of objects to find (e.g. 'scissors', 'cube').
            camera: Which segmentation node to trigger — 'arm' (default) for
                    grasping, 'front' for far-away detection during navigation.
            timeout: Max seconds to wait for segmentation result (default 30s,
                     first call may take longer due to model loading).

        Returns:
            dict with segmentation status and cached point cloud info.
        """
        camera = (camera or "arm").lower()
        if camera == "arm":
            topic_prefix = ""
        elif camera == "front":
            topic_prefix = "/front"
        else:
            return {
                "error": (
                    f"Unknown camera '{camera}'. Use 'arm' (default) or 'front'."
                )
            }

        text_topic = f"{topic_prefix}/segment_text"
        status_topic = f"{topic_prefix}/segmentation_status"
        pointcloud_topic = f"{topic_prefix}/segmented_pointcloud"
        mask_topic = f"{topic_prefix}/segmentation_mask"
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
                    "topic": pointcloud_topic,
                    "type": "sensor_msgs/msg/PointCloud2",
                }))

                # Wait for the message
                pc_ws.settimeout(timeout)
                while True:
                    raw = pc_ws.recv()
                    data = json.loads(raw)
                    if data.get("topic") == pointcloud_topic:
                        # Parse the pointcloud inline
                        msg = data.get("msg", {})
                        points, colors, frame_id = ws_manager._parse_pointcloud(msg)
                        pc_result["data"] = (points, colors, frame_id)
                        break

                pc_ws.send(json.dumps({
                    "op": "unsubscribe",
                    "id": sub_id,
                    "topic": pointcloud_topic,
                }))
                pc_ws.close()
            except Exception as e:
                pc_result["error"] = str(e)

        pc_thread = threading.Thread(target=_capture_pointcloud, daemon=True)
        pc_thread.start()

        # Give the subscriber a moment to register with rosbridge
        time.sleep(0.5)

        # Publish the text prompt to the selected camera's /segment_text
        try:
            ws = ws_manager._ensure_connected()
            pub_msg = {
                "op": "publish",
                "topic": text_topic,
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
                    status_topic,
                    msg_type="std_msgs/msg/String",
                    timeout=remaining,
                )
                status = status_msg.get("data", "UNKNOWN")
        except TimeoutError:
            return {
                "error": (
                    f"Timeout ({timeout}s) waiting for segmentation result on "
                    f"{status_topic}. Is the {camera}-camera segmentation node "
                    f"running? Check start_ros_stack.sh logs."
                )
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
                segmentation_cache["camera"] = camera
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
                "camera": camera,
                "outputs": {
                    "mask_topic": mask_topic,
                    "pointcloud_topic": pointcloud_topic,
                },
                "description": (
                    f"Segmentation of '{prompt}' on {camera} camera completed."
                    f"{pc_info}"
                ),
            }
        elif status == "NO_OBJECTS_FOUND":
            return {
                "status": "NO_OBJECTS_FOUND",
                "prompt": prompt,
                "camera": camera,
                "description": (
                    f"No objects matching '{prompt}' were found in the "
                    f"{camera} camera view."
                ),
            }
        else:
            return {
                "status": status,
                "prompt": prompt,
                "camera": camera,
                "description": f"Segmentation returned status: {status}",
            }
