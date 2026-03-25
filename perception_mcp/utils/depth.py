"""Depth image processing utilities.

Provides functions to project 2D pixel coordinates to 3D points
using depth images and camera intrinsics.
"""

import numpy as np


def pixel_to_3d(
    u: int,
    v: int,
    depth_image: np.ndarray,
    camera_info: dict,
    patch_size: int = 5,
) -> dict:
    """Project a 2D pixel coordinate to a 3D point using depth data.

    Uses a patch around the pixel to get a more robust depth estimate,
    filtering out NaN and zero values.

    Args:
        u: Pixel x coordinate
        v: Pixel y coordinate
        depth_image: Depth image as float32 array (values in meters)
        camera_info: dict with fx, fy, cx, cy from camera intrinsics
        patch_size: Size of the patch to average depth over (default: 5)

    Returns:
        dict with:
            - x, y, z: 3D coordinates in camera frame (meters)
            - valid: bool indicating if depth was valid
            - depth_value: raw depth value at the pixel
    """
    height, width = depth_image.shape[:2]

    # Clamp to image bounds
    u = max(0, min(u, width - 1))
    v = max(0, min(v, height - 1))

    # Extract patch for robust depth estimation
    half = patch_size // 2
    v_min = max(0, v - half)
    v_max = min(height, v + half + 1)
    u_min = max(0, u - half)
    u_max = min(width, u + half + 1)

    patch = depth_image[v_min:v_max, u_min:u_max].flatten()

    # Filter out invalid values (NaN, inf, zero, very far)
    valid_mask = np.isfinite(patch) & (patch > 0.01) & (patch < 20.0)
    valid_depths = patch[valid_mask]

    if len(valid_depths) == 0:
        return {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "valid": False,
            "depth_value": float(depth_image[v, u]),
        }

    # Use median for robustness against outliers
    depth = float(np.median(valid_depths))

    # Unproject using pinhole camera model
    fx = camera_info["fx"]
    fy = camera_info["fy"]
    cx = camera_info["cx"]
    cy = camera_info["cy"]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "valid": True,
        "depth_value": depth,
    }


def bbox_to_3d(
    bbox: list,
    depth_image: np.ndarray,
    camera_info: dict,
) -> dict:
    """Convert a 2D bounding box to a 3D position using depth.

    Projects the center of the bounding box to 3D space.

    Args:
        bbox: [x1, y1, x2, y2] bounding box in pixel coordinates
        depth_image: Depth image as float32 array (meters)
        camera_info: Camera intrinsics dict

    Returns:
        dict with:
            - position: {x, y, z} in camera frame
            - center_pixel: {u, v} center of bounding box
            - valid: bool
            - bbox_width_3d: estimated width in meters
            - bbox_height_3d: estimated height in meters
    """
    x1, y1, x2, y2 = bbox
    center_u = int((x1 + x2) / 2)
    center_v = int((y1 + y2) / 2)

    # Get 3D position of center
    point_3d = pixel_to_3d(center_u, center_v, depth_image, camera_info)

    result = {
        "position": {"x": point_3d["x"], "y": point_3d["y"], "z": point_3d["z"]},
        "center_pixel": {"u": center_u, "v": center_v},
        "valid": point_3d["valid"],
    }

    # Estimate 3D size if depth is valid
    if point_3d["valid"]:
        depth = point_3d["z"]
        fx = camera_info["fx"]
        fy = camera_info["fy"]
        bbox_w_pixels = x2 - x1
        bbox_h_pixels = y2 - y1
        result["bbox_width_3d"] = float(bbox_w_pixels * depth / fx)
        result["bbox_height_3d"] = float(bbox_h_pixels * depth / fy)
    else:
        result["bbox_width_3d"] = 0.0
        result["bbox_height_3d"] = 0.0

    return result
