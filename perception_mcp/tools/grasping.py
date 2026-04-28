"""Grasp planning tools.

Computes grasp targets from segmented point clouds, transforms them
to the robot base frame via TF2, and returns ready-to-use grasp poses
for MoveIt execution.
"""

import numpy as np
from fastmcp import FastMCP

from perception_mcp.utils.websocket import WebSocketManager

# Robotiq 2F-140 finger length offset (wrist frame → fingertip)
GRIPPER_FINGER_OFFSET_M = 0.14

# Top-down grasp orientation: gripper pointing straight down
# (180° rotation about X axis — EEF frame has Z-up, so flip to Z-down)
TOP_DOWN_ORIENTATION = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}


def _quat_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix (pure numpy)."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),       1 - 2*(qx*qx + qy*qy)],
    ])


def _tf_lookup(ws_manager, source_frame, target_frame="base_footprint", timeout=5.0):
    """Look up TF transform via the tf2_buffer_server action.

    In ROS 2 Jazzy, LookupTransform is an action (not a service).

    Returns (translation, rotation) or raises on failure.
    translation: dict with x, y, z
    rotation: dict with x, y, z, w
    """
    result = ws_manager.send_action_goal(
        action_name="/tf2_buffer_server",
        action_type="tf2_msgs/action/LookupTransform",
        goal={
            "target_frame": target_frame,
            "source_frame": source_frame,
            "source_time": {"sec": 0, "nanosec": 0},
            "timeout": {"sec": int(timeout), "nanosec": 0},
            "advanced": False,
        },
        timeout=timeout + 2.0,
    )
    transform = result.get("transform", {}).get("transform", {})
    translation = transform.get("translation", {})
    rotation = transform.get("rotation", {})
    return translation, rotation


def _transform_point(point, translation, rotation):
    """Transform a 3D point using a TF translation + quaternion rotation."""
    R = _quat_to_rotation_matrix(
        rotation["x"], rotation["y"], rotation["z"], rotation["w"]
    )
    p = np.array([point["x"], point["y"], point["z"]])
    t = np.array([translation["x"], translation["y"], translation["z"]])
    result = R @ p + t
    return {"x": round(float(result[0]), 4),
            "y": round(float(result[1]), 4),
            "z": round(float(result[2]), 4)}


def register_grasping_tools(
    mcp: FastMCP,
    ws_manager: WebSocketManager,
    segmentation_cache: dict = None,
) -> None:
    """Register grasp planning tools."""
    if segmentation_cache is None:
        segmentation_cache = {}

    @mcp.tool(
        description=(
            "Compute a grasp pose from the segmented point cloud.\n\n"
            "Uses the cached point cloud from the last segment_objects() call.\n"
            "Computes the centroid, transforms it from camera frame to base_footprint\n"
            "via TF2, and returns a top-down grasp pose ready for MoveIt.\n\n"
            "The grasp pose accounts for the Robotiq 140 gripper finger length\n"
            "(14cm offset above the object centroid).\n\n"
            "PREREQUISITE: Call segment_objects() first to generate the point cloud.\n\n"
            "The returned grasp_pose can be passed directly to MoveIt's\n"
            "plan_to_pose or plan_and_execute tool.\n\n"
            "Example usage:\n"
            "- First: segment_objects(prompt='scissors')\n"
            "- Then: get_grasp_from_pointcloud(object_name='scissors')\n"
            "- Then: plan_and_execute with the returned grasp_pose\n"
        ),
    )
    def get_grasp_from_pointcloud(
        object_name: str,
        pointcloud_topic: str = "/segmented_pointcloud",
        timeout: float = 10.0,
    ) -> dict:
        """Compute grasp pose from a segmented point cloud.

        Args:
            object_name: Name of the segmented object (for labeling).
            pointcloud_topic: Topic to read the point cloud from.
            timeout: Max seconds to wait for point cloud message.

        Returns:
            dict with centroid (camera + base frame), grasp_pose for MoveIt,
            and bounding box dimensions.
        """
        # Get pointcloud from cache or topic
        if segmentation_cache.get("points") is not None:
            points = segmentation_cache["points"]
            frame_id = segmentation_cache["frame_id"]
        else:
            try:
                points, _, frame_id = ws_manager.get_pointcloud(
                    pointcloud_topic, timeout=timeout
                )
            except TimeoutError:
                return {
                    "error": (
                        f"No cached point cloud and timeout ({timeout}s) waiting "
                        f"on {pointcloud_topic}. Call segment_objects() first."
                    )
                }
            except Exception as e:
                return {"error": f"Failed to read point cloud: {str(e)}"}

        if len(points) == 0:
            return {
                "object_name": object_name,
                "error": "Point cloud is empty — segmentation may have failed.",
            }

        # Compute centroid and bounding box
        centroid = points.mean(axis=0)
        pt_min = points.min(axis=0)
        pt_max = points.max(axis=0)
        size = pt_max - pt_min

        centroid_camera = {
            "x": round(float(centroid[0]), 4),
            "y": round(float(centroid[1]), 4),
            "z": round(float(centroid[2]), 4),
        }

        bounding_box = {
            "min": {"x": round(float(pt_min[0]), 4), "y": round(float(pt_min[1]), 4), "z": round(float(pt_min[2]), 4)},
            "max": {"x": round(float(pt_max[0]), 4), "y": round(float(pt_max[1]), 4), "z": round(float(pt_max[2]), 4)},
            "size": {"x": round(float(size[0]), 4), "y": round(float(size[1]), 4), "z": round(float(size[2]), 4)},
        }

        # TF lookup: camera frame → base_footprint.
        # Prefer the TF snapshot captured at segmentation time (so the
        # cached pointcloud is transformed with the TF that was valid when
        # it was captured). Fall back to a fresh lookup if unavailable.
        try:
            cached_t = segmentation_cache.get("tf_translation")
            cached_r = segmentation_cache.get("tf_rotation")
            if cached_t and cached_r and all(
                k in cached_t for k in ("x", "y", "z")
            ) and all(k in cached_r for k in ("x", "y", "z", "w")):
                translation, rotation = cached_t, cached_r
            else:
                translation, rotation = _tf_lookup(
                    ws_manager, source_frame=frame_id
                )
            centroid_base = _transform_point(centroid_camera, translation, rotation)

            grasp_pose = {
                "frame_id": "base_footprint",
                "position": {
                    "x": centroid_base["x"],
                    "y": centroid_base["y"],
                    "z": round(centroid_base["z"] + GRIPPER_FINGER_OFFSET_M, 4),
                },
                "orientation": TOP_DOWN_ORIENTATION,
            }

            return {
                "object_name": object_name,
                "centroid_camera_frame": centroid_camera,
                "centroid_base_frame": centroid_base,
                "grasp_pose": grasp_pose,
                "bounding_box": bounding_box,
                "num_points": len(points),
                "camera_frame_id": frame_id,
                "gripper_offset_m": GRIPPER_FINGER_OFFSET_M,
            }

        except (TimeoutError, RuntimeError, Exception) as e:
            # TF failed — return camera-frame data with warning
            return {
                "object_name": object_name,
                "centroid_camera_frame": centroid_camera,
                "centroid_base_frame": None,
                "grasp_pose": None,
                "bounding_box": bounding_box,
                "num_points": len(points),
                "camera_frame_id": frame_id,
                "gripper_offset_m": GRIPPER_FINGER_OFFSET_M,
                "warning": (
                    f"TF lookup failed ({str(e)}). "
                    "Centroid is in camera frame only. "
                    "Ensure tf2 buffer_server is running: "
                    "ros2 run tf2_ros buffer_server"
                ),
            }

