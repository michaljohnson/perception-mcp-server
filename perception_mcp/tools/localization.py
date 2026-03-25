"""3D localization tools.

Combines vision LLM detection with depth projection to get
3D positions of objects in the camera frame.
"""

from fastmcp import FastMCP

from perception_mcp.utils.depth import bbox_to_3d, pixel_to_3d
from perception_mcp.utils.websocket import WebSocketManager


def register_localization_tools(
    mcp: FastMCP,
    ws_manager: WebSocketManager,
    vision_client,
    camera_topics: dict,
) -> None:
    """Register 3D localization tools."""

    @mcp.tool(
        description=(
            "Get the 3D position of an object in the camera frame.\n\n"
            "Detects the object using vision AI, then projects its pixel\n"
            "coordinates to 3D using the depth camera. Returns position in\n"
            "the camera optical frame (x=right, y=down, z=forward).\n\n"
            "Example usage:\n"
            "- get_object_3d_pose(object_name='scissors')\n"
            "- get_object_3d_pose(object_name='cube', camera='arm')\n"
        ),
    )
    def get_object_3d_pose(object_name: str, camera: str = "front") -> dict:
        """Get the 3D position of a named object.

        Args:
            object_name: Name of the object to localize (e.g., 'scissors', 'cube')
            camera: Which camera to use - 'front' (default) or 'arm'.

        Returns:
            dict with:
                - object_name: str
                - found: bool
                - position_camera_frame: {x, y, z} in camera optical frame (meters)
                - center_pixel: {u, v} pixel coordinates
                - confidence: float
                - bbox: [x1, y1, x2, y2] pixel bounding box
                - bbox_size_3d: {width, height} estimated 3D size in meters
        """
        topics = camera_topics.get(camera, camera_topics["front"])
        rgb_topic = topics["rgb"]
        depth_topic = topics["depth"]
        camera_info_topic = topics["camera_info"]

        # Get RGB image
        try:
            img_bytes = ws_manager.get_compressed_image(rgb_topic, timeout=10.0)
        except Exception as e:
            return {"error": f"Failed to get RGB image: {str(e)}"}

        # Detect object
        try:
            detection = vision_client.detect_objects(img_bytes, prompt=object_name)
        except Exception as e:
            return {"error": f"Vision API call failed: {str(e)}"}

        objects = detection.get("objects", [])
        if not objects:
            return {
                "object_name": object_name,
                "found": False,
                "error": f"Object '{object_name}' not detected in current view.",
                "scene_description": detection.get("scene_description", ""),
            }

        # Take the highest confidence detection
        best = max(objects, key=lambda o: o.get("confidence", 0))
        bbox = best.get("bbox")

        if not bbox or len(bbox) != 4:
            return {
                "object_name": object_name,
                "found": True,
                "error": "Object detected but no valid bounding box returned.",
                "detection": best,
            }

        # Get depth image and camera info
        try:
            depth_image = ws_manager.get_depth_image(depth_topic, timeout=10.0)
            camera_info = ws_manager.get_camera_info(camera_info_topic, timeout=5.0)
        except Exception as e:
            # Return 2D detection even if depth fails
            return {
                "object_name": object_name,
                "found": True,
                "position_camera_frame": None,
                "center_pixel": {
                    "u": best.get("center_x", int((bbox[0] + bbox[2]) / 2)),
                    "v": best.get("center_y", int((bbox[1] + bbox[3]) / 2)),
                },
                "confidence": best.get("confidence", 0),
                "bbox": bbox,
                "error": f"Depth unavailable: {str(e)}. 2D detection only.",
            }

        # Project to 3D
        result_3d = bbox_to_3d(bbox, depth_image, camera_info)

        return {
            "object_name": object_name,
            "found": True,
            "position_camera_frame": result_3d["position"],
            "center_pixel": result_3d["center_pixel"],
            "confidence": best.get("confidence", 0),
            "bbox": bbox,
            "bbox_size_3d": {
                "width": result_3d["bbox_width_3d"],
                "height": result_3d["bbox_height_3d"],
            },
            "depth_valid": result_3d["valid"],
            "description": best.get("description", ""),
        }

    @mcp.tool(
        description=(
            "Get the 3D position of a specific pixel coordinate using depth.\n\n"
            "Useful when you already know where an object is in the image\n"
            "and just need its 3D position. Does NOT require vision API.\n\n"
            "Example usage:\n"
            "- get_point_3d(u=320, v=240)  # center of 640x480 image\n"
            "- get_point_3d(u=320, v=240, camera='arm')\n"
        ),
    )
    def get_point_3d(u: int, v: int, camera: str = "front") -> dict:
        """Get 3D position of a pixel coordinate using depth camera.

        Args:
            u: Pixel x coordinate
            v: Pixel y coordinate
            camera: Which camera to use - 'front' (default) or 'arm'.

        Returns:
            dict with x, y, z in camera frame (meters) and validity flag.
        """
        topics = camera_topics.get(camera, camera_topics["front"])
        depth_topic = topics["depth"]
        camera_info_topic = topics["camera_info"]

        try:
            depth_image = ws_manager.get_depth_image(depth_topic, timeout=10.0)
            camera_info = ws_manager.get_camera_info(camera_info_topic, timeout=5.0)
        except Exception as e:
            return {"error": f"Failed to get depth data: {str(e)}"}

        return pixel_to_3d(u, v, depth_image, camera_info)
