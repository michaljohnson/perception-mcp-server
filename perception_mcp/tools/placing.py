"""Place-related drop-pose tools.

Provides `get_topdown_placing_pose`, a single primitive that computes a
top-down drop pose from a segmented (or cropped raw) point cloud.
Two modes selected by `object_height_m`:

  - **Container mode** (`object_height_m=0`, default): drop INTO a deep
    target (bin/basket/bowl). The held object falls vertically into the
    container, so `wrist_z = top_z + top_clearance_m` is enough — no need
    to account for the held object dimensions. Typical caller passes
    `top_clearance_m=0.35`.

  - **Surface mode** (`object_height_m > 0`): set DOWN onto a flat
    surface (table/counter). The held object hangs below the wrist by
    `_GRIPPER_FINGER_LENGTH + object_height_m`. Without that correction,
    the released object clips into the surface. Formula becomes:
        wrist_z = top_z + _GRIPPER_FINGER_LENGTH + object_height_m + top_clearance_m
    where `top_clearance_m` becomes the air gap above the surface (e.g.
    0.05m). Caller passes `object_height_m` per held object (e.g. 0.12 for
    coke can, 0.10 for shoe).

Algorithm (both modes):
  1. Read cached cloud (default) or raw topic; transform to
     base_footprint via cached or fresh TF.
  2. Optional xy crop (raw-depth mode).
  3. cx, cy = mean of points; top_z = 95th-percentile of z.
  4. wrist_z = top_z + clearance correction (mode-dependent, see above).
  5. orientation = strict top-down (1, 0, 0, 0).

PRECONDITION: caller must pre-position the robot so the target is in
camera view and within UR5 top-down reach (target xy distance from base
< ~0.6m). The place agent handles the drive-in when the first drop-pose
call reports a far target. At close top-down range there's no need for
normal-based wall rejection, DBSCAN clustering, or tilt heuristics —
simple statistics are enough.
"""

import numpy as np
from fastmcp import FastMCP

# Distance from gripper wrist link to the inner finger pads (where a
# grasped object sits). UR5e + Robotiq 2F-140 ≈ 14cm. Used in surface
# mode to compute correct wrist Z so the held object's bottom lands at
# `surface + top_clearance_m` rather than the wrist.
_GRIPPER_FINGER_LENGTH = 0.14

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
            "point cloud. Two modes selected by `object_height_m`:\n\n"
            "  - **Container mode** (object_height_m=0, default): drop "
            "INTO a deep target (bin/basket). Held object falls vertically. "
            "Pass `top_clearance_m=0.35`. Wrist ends up at top_z + 0.35.\n"
            "  - **Surface mode** (object_height_m > 0): set DOWN on a "
            "flat surface (table/counter). Tool accounts for "
            "gripper finger length (~14cm) + object height so the held "
            "object's bottom lands at `surface + top_clearance_m`. Pass "
            "object_height_m per held object (~0.12 for coke can, ~0.10 "
            "for shoe) and top_clearance_m as the desired air gap (e.g. "
            "0.05).\n\n"
            "Algorithm (both modes):\n"
            "  1. Read cached SAM3 cloud (default) or raw depth topic.\n"
            "  2. Transform to base_footprint via cached/fresh TF.\n"
            "  3. Optional xy crop (raw-depth mode).\n"
            "  4. cx, cy = mean of point xy; top_z = 95th percentile of z.\n"
            "  5. wrist_z formula:\n"
            "       container: wrist_z = top_z + top_clearance_m\n"
            "       surface:   wrist_z = top_z + 0.14 (finger) + "
            "object_height_m + top_clearance_m\n"
            "     orientation = strict top-down (1, 0, 0, 0).\n\n"
            "PRECONDITION: pre-position the robot so the target is in the\n"
            "camera's view and within UR5 top-down reach (target xy from\n"
            "base < ~0.6m). Use the place agent's drive-closer logic if\n"
            "the first call reports a far target.\n\n"
            "POINTCLOUD SOURCE:\n"
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
        object_height_m: float = 0.0,
        pointcloud_topic: str = "/segmented_pointcloud",
        use_cached: bool = True,
        crop_center_x: float = None,
        crop_center_y: float = None,
        crop_radius_m: float = 0.30,
        timeout: float = 10.0,
    ) -> dict:
        """Top-down drop pose: mean(xy) + 95th-percentile(z) + mode-aware
        vertical correction for finger + object height (surface mode).

        Args:
            object_name: Label for the response (informational only).
            top_clearance_m: Vertical clearance above the high-z point.
                Container mode: 0.35m typical (drop INTO bin).
                Surface mode: ~0.05m (air gap above surface).
            object_height_m: Height of the held object in meters. Setting
                this to a positive value enables SURFACE mode and adds
                `_GRIPPER_FINGER_LENGTH + object_height_m` to wrist_z so
                the released object lands at `surface + top_clearance_m`.
                Default 0.0 = container mode (legacy behavior).
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
            base_footprint), place_pose (clearance + finger + object_height
            applied, top-down, ready for MoveIt), mode ('surface' or
            'container'), and diagnostics.
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

        if object_height_m > 0:
            mode = "surface"
            drop_z = round(
                top_z + _GRIPPER_FINGER_LENGTH + object_height_m + top_clearance_m,
                4,
            )
            note = (
                f"place_pose.z = surface_height_m ({top_z}) + "
                f"finger ({_GRIPPER_FINGER_LENGTH}) + "
                f"object_height_m ({object_height_m}) + "
                f"top_clearance_m ({top_clearance_m}). "
                "Wrist ends up high enough that held object's bottom "
                f"sits at surface + {top_clearance_m}m. "
                "Send to MoveIt without further vertical offset."
            )
        else:
            mode = "container"
            drop_z = round(top_z + top_clearance_m, 4)
            note = (
                f"place_pose.z = surface_height_m ({top_z}) + "
                f"top_clearance_m ({top_clearance_m}). "
                "Container mode (object_height_m=0): wrist above the "
                "container rim, held object falls in. Pass "
                "object_height_m > 0 if placing onto a flat surface "
                "to avoid clipping the held object into it. "
                "Send to MoveIt without further vertical offset."
            )

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
            "object_height_m": object_height_m,
            "gripper_finger_length_m": _GRIPPER_FINGER_LENGTH,
            "mode": mode,
            "method": "mean_xy_p95_z",
            "num_points_used": len(points_base),
            "num_points_pre_crop": num_points_pre_crop,
            "crop_applied": crop_applied,
            "source": "cache" if used_cache else f"topic:{pointcloud_topic}",
            "note": note,
        }
