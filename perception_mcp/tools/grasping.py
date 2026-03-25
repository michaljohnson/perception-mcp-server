"""Grasp planning tools.

Combines detection, depth projection, and LLM reasoning to
generate grasp poses and check reachability.
"""

from fastmcp import FastMCP

from perception_mcp.utils.depth import bbox_to_3d, pixel_to_3d
from perception_mcp.utils.websocket import WebSocketManager


def register_grasping_tools(
    mcp: FastMCP,
    ws_manager: WebSocketManager,
    vision_client,
    camera_topics: dict,
) -> None:
    """Register grasp planning tools."""

    @mcp.tool(
        description=(
            "Plan a grasp for a specific object.\n\n"
            "Detects the object, estimates its 3D position, and uses vision AI\n"
            "to reason about the best grasp approach (top-down vs side,\n"
            "gripper orientation, approach direction).\n\n"
            "Returns the object's 3D position in camera frame and grasp strategy.\n"
            "The caller should transform this to the robot base frame for MoveIt.\n\n"
            "Example usage:\n"
            "- get_grasp_pose(object_name='scissors')\n"
            "- get_grasp_pose(object_name='cube', camera='front')  # use front camera\n"
        ),
    )
    def get_grasp_pose(object_name: str, camera: str = "arm") -> dict:
        """Plan a grasp for a named object.

        Args:
            object_name: Name of the object to grasp (e.g., 'scissors', 'cube')
            camera: Which camera to use - 'arm' (default for grasping) or 'front'.

        Returns:
            dict with:
                - object_name: str
                - found: bool
                - position_camera_frame: {x, y, z} in camera optical frame
                - grasp_type: "top_down", "side", or "angled"
                - approach_direction: approach direction
                - gripper_orientation: suggested gripper orientation
                - estimated_width_cm: object width for gripper opening
                - obstacles_nearby: list of nearby obstacles
                - notes: additional grasp planning notes
        """
        topics = camera_topics.get(camera, camera_topics["arm"])
        rgb_topic = topics["rgb"]
        depth_topic = topics["depth"]
        camera_info_topic = topics["camera_info"]

        # Get RGB image
        try:
            img_bytes = ws_manager.get_compressed_image(rgb_topic, timeout=10.0)
        except Exception as e:
            return {"error": f"Failed to get RGB image: {str(e)}"}

        # Get grasp strategy from vision LLM
        try:
            grasp_info = vision_client.estimate_grasp_approach(img_bytes, object_name)
        except Exception as e:
            return {"error": f"Vision API call failed: {str(e)}"}

        if not grasp_info.get("object_found", False):
            return {
                "object_name": object_name,
                "found": False,
                "error": f"Object '{object_name}' not found in current view.",
                "grasp_info": grasp_info,
            }

        # Get depth and camera info for 3D localization
        position_3d = None
        depth_valid = False

        center_u = grasp_info.get("center_x", 0)
        center_v = grasp_info.get("center_y", 0)

        try:
            depth_image = ws_manager.get_depth_image(depth_topic, timeout=10.0)
            camera_info = ws_manager.get_camera_info(camera_info_topic, timeout=5.0)

            point = pixel_to_3d(
                int(center_u), int(center_v), depth_image, camera_info
            )
            if point["valid"]:
                position_3d = {"x": point["x"], "y": point["y"], "z": point["z"]}
                depth_valid = True
        except Exception:
            pass  # Return grasp strategy without 3D if depth fails

        return {
            "object_name": object_name,
            "found": True,
            "position_camera_frame": position_3d,
            "center_pixel": {"u": int(center_u), "v": int(center_v)},
            "depth_valid": depth_valid,
            "grasp_type": grasp_info.get("grasp_type", "unknown"),
            "approach_direction": grasp_info.get("approach_direction", "unknown"),
            "gripper_orientation": grasp_info.get("gripper_orientation", "unknown"),
            "estimated_width_cm": grasp_info.get("estimated_width_cm", 0),
            "obstacles_nearby": grasp_info.get("obstacles_nearby", []),
            "notes": grasp_info.get("notes", ""),
        }

    @mcp.tool(
        description=(
            "Check if an object is within the robot arm's reachable workspace.\n\n"
            "Estimates the 3D distance to the object and checks if it's\n"
            "likely reachable by the arm (typically < 0.8m for most arms).\n\n"
            "Example usage:\n"
            "- check_reachability(object_name='cube')\n"
            "- check_reachability(object_name='scissors', max_reach=0.7)\n"
            "- check_reachability(object_name='cup', camera='front')\n"
        ),
    )
    def check_reachability(object_name: str, max_reach: float = 0.8, camera: str = "arm") -> dict:
        """Check if an object is within arm reach.

        Args:
            object_name: Object to check reachability for
            max_reach: Maximum reach distance in meters (default: 0.8m)
            camera: Which camera to use - 'arm' (default for reachability) or 'front'.

        Returns:
            dict with:
                - reachable: bool
                - distance: float (meters)
                - position_camera_frame: {x, y, z}
                - suggestion: str (e.g., "move 0.3m closer")
        """
        topics = camera_topics.get(camera, camera_topics["arm"])
        rgb_topic = topics["rgb"]
        depth_topic = topics["depth"]
        camera_info_topic = topics["camera_info"]

        # Detect object
        try:
            img_bytes = ws_manager.get_compressed_image(rgb_topic, timeout=10.0)
        except Exception as e:
            return {"error": f"Failed to get image: {str(e)}"}

        try:
            detection = vision_client.detect_objects(img_bytes, prompt=object_name)
        except Exception as e:
            return {"error": f"Vision API call failed: {str(e)}"}

        objects = detection.get("objects", [])
        if not objects:
            return {
                "object_name": object_name,
                "reachable": False,
                "error": f"Object '{object_name}' not visible.",
                "suggestion": "Navigate to find the object first.",
            }

        best = max(objects, key=lambda o: o.get("confidence", 0))
        bbox = best.get("bbox")

        if not bbox or len(bbox) != 4:
            return {
                "object_name": object_name,
                "reachable": False,
                "error": "No valid bounding box.",
            }

        # Get 3D position
        try:
            depth_image = ws_manager.get_depth_image(depth_topic, timeout=10.0)
            camera_info = ws_manager.get_camera_info(camera_info_topic, timeout=5.0)
        except Exception as e:
            return {"error": f"Depth unavailable: {str(e)}"}

        result_3d = bbox_to_3d(bbox, depth_image, camera_info)

        if not result_3d["valid"]:
            return {
                "object_name": object_name,
                "reachable": False,
                "error": "Invalid depth at object location.",
            }

        pos = result_3d["position"]
        distance = (pos["x"] ** 2 + pos["y"] ** 2 + pos["z"] ** 2) ** 0.5

        reachable = distance <= max_reach

        suggestion = ""
        if not reachable:
            move_closer = distance - max_reach + 0.1  # 10cm margin
            suggestion = f"Move approximately {move_closer:.2f}m closer to the object."

        return {
            "object_name": object_name,
            "reachable": reachable,
            "distance": round(distance, 3),
            "position_camera_frame": pos,
            "max_reach": max_reach,
            "suggestion": suggestion if suggestion else "Object is within reach.",
        }
