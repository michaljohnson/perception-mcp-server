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

    @mcp.tool(
        description=(
            "Compute a drop pose for placing an object into a container (bin,\n"
            "basket, bowl, drainer). Filters the cached segmented point cloud\n"
            "to the rim's top slice and returns the center of that slice in\n"
            "base_footprint — unbiased by back-wall points or asymmetric\n"
            "point density.\n\n"
            "Unlike get_grasp_from_pointcloud (which returns a raw mean of all\n"
            "points — biased toward whatever face of the container the camera\n"
            "sees most of), this tool:\n"
            "  1. transforms ALL segmented points to base_footprint,\n"
            "  2. filters to points within `slice_height` meters of z_max (the\n"
            "     rim),\n"
            "  3. returns the (x, y) midpoint of the rim's bounding box.\n"
            "The midpoint-of-bbox (not mean) further removes bias when only\n"
            "part of the rim is visible.\n\n"
            "Returns `drop_pose` — a pose AT the rim center at rim height.\n"
            "Add clearance (e.g. +0.35m) on the caller side before commanding\n"
            "MoveIt, so the object falls into the container.\n\n"
            "PREREQUISITE: call segment_objects(prompt='<container>',\n"
            "camera='front') first. The arm camera is blocked by the held\n"
            "object during placing.\n"
        ),
    )
    def get_container_drop_pose(
        object_name: str,
        slice_height: float = 0.03,
        pointcloud_topic: str = "/segmented_pointcloud",
        timeout: float = 10.0,
    ) -> dict:
        """Compute a rim-centered drop pose for a segmented container.

        Args:
            object_name: Name of the segmented container (for labeling).
            slice_height: Thickness of the top-rim band to keep, in meters.
                Default 0.03m = 3cm. Too thin and you lose points, too thick
                and you include inner-wall points that drag the midpoint
                toward the back wall.
            pointcloud_topic: Topic to read the point cloud from if not
                cached.
            timeout: Max seconds to wait for point cloud message.

        Returns:
            dict with rim center (base frame), drop_pose (at rim height in
            base_footprint), slice bounding box, and point counts.
        """
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
                        f"No cached point cloud and timeout ({timeout}s) "
                        f"waiting on {pointcloud_topic}. Call "
                        "segment_objects(camera='front') first."
                    )
                }
            except Exception as e:
                return {"error": f"Failed to read point cloud: {str(e)}"}

        if len(points) == 0:
            return {
                "object_name": object_name,
                "error": "Point cloud is empty — segmentation may have failed.",
            }

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
        except Exception as e:
            return {
                "object_name": object_name,
                "error": (
                    f"TF lookup failed ({str(e)}). Cannot compute drop pose "
                    "without transform to base_footprint."
                ),
            }

        R = _quat_to_rotation_matrix(
            rotation["x"], rotation["y"], rotation["z"], rotation["w"]
        )
        t = np.array([translation["x"], translation["y"], translation["z"]])
        points_base = (R @ points.T).T + t  # (N, 3) in base_footprint

        z_max = float(points_base[:, 2].max())
        z_min = float(points_base[:, 2].min())
        slice_mask = points_base[:, 2] >= (z_max - slice_height)
        top_slice = points_base[slice_mask]

        if len(top_slice) < 3:
            return {
                "object_name": object_name,
                "error": (
                    f"Top slice has only {len(top_slice)} points after "
                    f"filtering (slice_height={slice_height}m). Rim likely "
                    "not visible; try wider slice or re-segment."
                ),
                "num_points_total": len(points_base),
                "z_min": round(z_min, 4),
                "z_max": round(z_max, 4),
            }

        x_min, x_max = float(top_slice[:, 0].min()), float(top_slice[:, 0].max())
        y_min, y_max = float(top_slice[:, 1].min()), float(top_slice[:, 1].max())
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        rim_z = float(top_slice[:, 2].mean())

        drop_pose = {
            "frame_id": "base_footprint",
            "position": {
                "x": round(cx, 4),
                "y": round(cy, 4),
                "z": round(rim_z, 4),
            },
            "orientation": TOP_DOWN_ORIENTATION,
        }

        rim_center = {
            "x": round(cx, 4),
            "y": round(cy, 4),
            "z": round(rim_z, 4),
        }

        return {
            "object_name": object_name,
            "rim_center_base_frame": rim_center,
            "drop_pose": drop_pose,
            "rim_bbox": {
                "x_min": round(x_min, 4), "x_max": round(x_max, 4),
                "y_min": round(y_min, 4), "y_max": round(y_max, 4),
                "width_x": round(x_max - x_min, 4),
                "width_y": round(y_max - y_min, 4),
            },
            "slice_height_m": slice_height,
            "num_points_total": len(points_base),
            "num_points_in_slice": int(slice_mask.sum()),
            "z_max": round(z_max, 4),
            "z_min": round(z_min, 4),
            "note": (
                "drop_pose z is the rim top. Add vertical clearance "
                "(e.g. +0.35m) before sending to MoveIt."
            ),
        }
