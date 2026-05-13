"""Grasp planning tools.

Computes grasp targets from segmented point clouds, transforms them
to the robot base frame via TF2, and returns ready-to-use grasp poses
for MoveIt execution.
"""

import math

from fastmcp import FastMCP

from perception_mcp.utils.transforms import (
    GRIPPER_FINGER_OFFSET_M,
    TOP_DOWN_ORIENTATION,
    _PCA_ASPECT_RATIO_MIN,
    _oriented_topdown_quaternion,
    _principal_axis_angle_xy,
    _shortest_grasp_yaw,
    _tf_lookup,
    _transform_point,
    _transform_points,
)
from perception_mcp.utils.websocket import WebSocketManager


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
            "Orientation is shape-aware: a 2D PCA on the base-frame xy\n"
            "projection of the point cloud gives the object's principal\n"
            "axis, and the gripper yaw is set so the fingers close ACROSS\n"
            "the short axis. For approximately round / square top-down\n"
            "footprints (aspect ratio < 1.2) the default top-down\n"
            "orientation [1, 0, 0, 0] is returned unchanged. The response\n"
            "exposes principal_axis_angle_rad, principal_axis_angle_deg,\n"
            "principal_axis_aspect_ratio, and a boolean oriented flag so\n"
            "callers can inspect the geometry.\n\n"
            "PREREQUISITE: Call segment_objects() first to generate the point cloud.\n\n"
            "The returned grasp_pose can be passed directly to MoveIt's\n"
            "plan_to_pose or plan_and_execute tool.\n\n"
            "Example usage:\n"
            "- First: segment_objects(prompt='scissors')\n"
            "- Then: get_topdown_grasp_pose(object_name='scissors')\n"
            "- Then: plan_and_execute with the returned grasp_pose\n"
        ),
    )
    def get_topdown_grasp_pose(
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

            # 2D PCA on the base-frame xy projection of the point cloud
            # to align the gripper fingers with the object's short axis.
            # Round / square footprints (aspect ratio below the threshold)
            # fall back to the default top-down orientation.
            #
            # Empirical Robotiq 2F-140 finger-axis convention (validated
            # 2026-05-12 on a red shoe): the fingers close along the
            # gripper-tool X axis, NOT tool Y as initially assumed.
            # After the top-down flip (q_topdown = 180 deg about base-X)
            # the finger axis maps to base +X. To rotate the fingers
            # ACROSS the object's short axis, the gripper yaw therefore
            # equals the SHORT-axis angle = angle_long + pi/2.
            points_base = _transform_points(points, translation, rotation)
            angle_long, aspect_ratio = _principal_axis_angle_xy(points_base)
            if aspect_ratio >= _PCA_ASPECT_RATIO_MIN:
                grasp_yaw = _shortest_grasp_yaw(angle_long + math.pi / 2)
                orientation = _oriented_topdown_quaternion(grasp_yaw)
                orientation = {k: round(v, 6) for k, v in orientation.items()}
                oriented = True
            else:
                grasp_yaw = 0.0
                orientation = TOP_DOWN_ORIENTATION
                oriented = False

            grasp_pose = {
                "frame_id": "base_footprint",
                "position": {
                    "x": centroid_base["x"],
                    "y": centroid_base["y"],
                    "z": round(centroid_base["z"] + GRIPPER_FINGER_OFFSET_M, 4),
                },
                "orientation": orientation,
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
                "principal_axis_angle_rad": round(float(grasp_yaw), 4),
                "principal_axis_angle_deg": round(float(grasp_yaw) * 180.0 / 3.14159265358979, 2),
                "principal_axis_aspect_ratio": round(float(aspect_ratio), 3),
                "oriented": oriented,
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

