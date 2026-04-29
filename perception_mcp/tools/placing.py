"""Place-related drop-pose tools.

Provides `get_topdown_placing_pose`, a single primitive that computes a
top-down drop pose from a segmented (or cropped raw) point cloud.
Same algorithm for containers and surfaces — vertical clearance is the
only knob (container 0.35m, surface 0.20m).

Algorithm (mirrors pick):
  1. Read cached cloud (default) or raw topic; transform to
     base_footprint via cached or fresh TF.
  2. Optional xy crop (raw-depth mode).
  3. cx, cy = mean of points; top_z = 95th-percentile of z.
  4. drop_z = top_z + clearance; orientation = strict top-down
     (1, 0, 0, 0).

PRECONDITION: caller must pre-position the robot so the target is in
camera view and within UR5 top-down reach (target xy distance from base
< ~0.6m). The place agent's `creep_closer` helper handles the drive-in
when the first drop-pose call reports a far target. At close, top-down
range there is no need for normal-based wall rejection, DBSCAN
clustering, or tilt heuristics — simple statistics are enough.
"""

import numpy as np
from fastmcp import FastMCP

from perception_mcp.utils.transforms import (
    TOP_DOWN_ORIENTATION,
    _quat_to_rotation_matrix,
    _tf_lookup,
)
from perception_mcp.utils.websocket import WebSocketManager


def register_placing_tools(
    mcp: FastMCP,
    ws_manager: WebSocketManager,
    segmentation_cache: dict = None,
) -> None:
    """Register place-pose tools."""
    if segmentation_cache is None:
        segmentation_cache = {}

    @mcp.tool(
        description=(
            "Compute a top-down drop pose from a segmented (or cropped raw) "
            "point cloud. Same algorithm for containers and surfaces — "
            "vertical clearance is the only knob:\n"
            "  - Container (bin/basket/bowl): top_clearance_m=0.35\n"
            "  - Surface  (table/counter):    top_clearance_m=0.20\n\n"
            "Algorithm:\n"
            "  1. Read cached SAM3 cloud (default) or raw depth topic.\n"
            "  2. Transform to base_footprint via cached/fresh TF.\n"
            "  3. Optional xy crop (raw-depth mode).\n"
            "  4. cx, cy = mean of point xy; top_z = 95th percentile of z.\n"
            "  5. drop_z = top_z + top_clearance_m; orientation = strict\n"
            "     top-down (1, 0, 0, 0).\n\n"
            "PRECONDITION: pre-position the robot so the target is in the\n"
            "camera's view and within UR5 top-down reach (target xy from\n"
            "base < ~0.6m). Use the place agent's creep_closer helper if\n"
            "the first call reports a far target.\n\n"
            "MODES:\n"
            "  - use_cached=True (default): use the SAM3-segmented cloud\n"
            "    cached by segment_objects(). Call segment_objects() first.\n"
            "  - use_cached=False: read a raw depth topic. Combine with\n"
            "    crop_center_x/y so attention is restricted to the target\n"
            "    region. Useful when SAM3 fails on the target view\n"
            "    (e.g. bin rim from arm-cam top-down)."
        ),
    )
    def get_topdown_placing_pose(
        object_name: str,
        top_clearance_m: float = 0.20,
        pointcloud_topic: str = "/segmented_pointcloud",
        use_cached: bool = True,
        crop_center_x: float = None,
        crop_center_y: float = None,
        crop_radius_m: float = 0.30,
        timeout: float = 10.0,
    ) -> dict:
        """Top-down drop pose: mean(xy) + 95th-percentile(z) + clearance.

        Args:
            object_name: Label for the response (informational only).
            top_clearance_m: Vertical clearance above the high-z point.
                0.35m for drop-INTO-container; 0.20m for place-ONTO-surface.
            pointcloud_topic: Topic to read if use_cached=False or cache
                empty. Default `/segmented_pointcloud` (SAM3 output).
                Use `/arm_camera/points` for raw arm depth or
                `/front_rgbd_camera/points` for raw front depth.
            use_cached: If True (default) use the SAM3-segmented cloud
                cached by segment_objects(). Set False for raw-depth
                fallback — must combine with crop_center_x/y so the
                algorithm only looks at the target region.
            crop_center_x, crop_center_y: If set, filter points after
                TF transform to within `crop_radius_m` of (x, y) in
                base_footprint. Required for raw-depth mode.
            crop_radius_m: Half-extent of the crop window (default 0.30m).
            timeout: Max seconds to wait for a point cloud message.

        Returns:
            On success: surface_height_m (top_z), surface_centroid (in
            base_footprint), place_pose (clearance applied, top-down,
            ready for MoveIt), and diagnostics.
            On failure: dict with `error` describing why.
        """
        used_cache = False
        if use_cached and segmentation_cache.get("points") is not None:
            points = segmentation_cache["points"]
            frame_id = segmentation_cache["frame_id"]
            used_cache = True
        else:
            try:
                points, _, frame_id = ws_manager.get_pointcloud(
                    pointcloud_topic, timeout=timeout
                )
            except TimeoutError:
                return {
                    "error": (
                        f"Timeout ({timeout}s) waiting for point cloud "
                        f"on {pointcloud_topic}. "
                        + (
                            "Call segment_objects() first, "
                            "or set use_cached=False with a raw depth topic."
                            if use_cached
                            else "Topic may not be publishing."
                        )
                    )
                }
            except Exception as e:
                return {"error": f"Failed to read point cloud: {str(e)}"}

        if len(points) < 50:
            return {
                "object_name": object_name,
                "error": (
                    f"Point cloud too small ({len(points)} points, need ≥50)."
                ),
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
                    f"TF lookup failed ({str(e)}). Cannot compute drop "
                    "pose without transform to base_footprint."
                ),
            }

        R = _quat_to_rotation_matrix(
            rotation["x"], rotation["y"], rotation["z"], rotation["w"]
        )
        t = np.array([translation["x"], translation["y"], translation["z"]])
        points_base = (R @ points.T).T + t

        num_points_pre_crop = len(points_base)
        crop_applied = False
        if crop_center_x is not None and crop_center_y is not None:
            dx = points_base[:, 0] - crop_center_x
            dy = points_base[:, 1] - crop_center_y
            crop_mask = (dx * dx + dy * dy) <= (crop_radius_m * crop_radius_m)
            points_base = points_base[crop_mask]
            crop_applied = True
            if len(points_base) < 50:
                return {
                    "object_name": object_name,
                    "error": (
                        f"After cropping to ({crop_center_x:.3f}, "
                        f"{crop_center_y:.3f}) ± {crop_radius_m}m, only "
                        f"{len(points_base)} points remain (need ≥50). "
                        f"Pre-crop count was {num_points_pre_crop}. Crop "
                        "window may not contain the target."
                    ),
                    "num_points_pre_crop": num_points_pre_crop,
                    "num_points_post_crop": len(points_base),
                }

        cx = round(float(points_base[:, 0].mean()), 4)
        cy = round(float(points_base[:, 1].mean()), 4)
        top_z = round(float(np.percentile(points_base[:, 2], 95)), 4)
        drop_z = round(top_z + top_clearance_m, 4)

        place_pose = {
            "frame_id": "base_footprint",
            "position": {"x": cx, "y": cy, "z": drop_z},
            "orientation": dict(TOP_DOWN_ORIENTATION),
        }

        return {
            "object_name": object_name,
            "surface_height_m": top_z,
            "surface_centroid": {"x": cx, "y": cy, "z": top_z},
            "place_pose": place_pose,
            "top_clearance_m": top_clearance_m,
            "method": "mean_xy_p95_z",
            "num_points_used": len(points_base),
            "num_points_pre_crop": num_points_pre_crop,
            "crop_applied": crop_applied,
            "source": "cache" if used_cache else f"topic:{pointcloud_topic}",
            "note": (
                f"place_pose.z = surface_height_m ({top_z}) + "
                f"top_clearance_m ({top_clearance_m}). Send to MoveIt "
                "without further vertical offset."
            ),
        }
