"""Grasp planning tools.

Combines detection, depth projection, and LLM reasoning to
generate grasp poses and check reachability. Also provides
pointcloud-based grasp target computation and planning scene
collision object management.
"""

import time

import numpy as np
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

    @mcp.tool(
        description=(
            "Compute a grasp target from the segmented point cloud.\n\n"
            "Reads the /segmented_pointcloud topic (published by segment_objects)\n"
            "and computes the centroid as the grasp target position. Also returns\n"
            "the bounding box dimensions of the object.\n\n"
            "PREREQUISITE: Call segment_objects() first to generate the point cloud.\n\n"
            "Returns the grasp position in the camera frame. The caller must\n"
            "transform this to the robot base frame for MoveIt.\n\n"
            "Example usage:\n"
            "- First: segment_objects(prompt='red cube')\n"
            "- Then: get_grasp_from_pointcloud(object_name='red cube')\n"
        ),
    )
    def get_grasp_from_pointcloud(
        object_name: str,
        pointcloud_topic: str = "/segmented_pointcloud",
        timeout: float = 10.0,
    ) -> dict:
        """Compute grasp target from a segmented point cloud.

        Args:
            object_name: Name of the segmented object (for labeling).
            pointcloud_topic: Topic to read the point cloud from.
            timeout: Max seconds to wait for point cloud message.

        Returns:
            dict with:
                - object_name: str
                - centroid: {x, y, z} grasp target in camera frame (meters)
                - bounding_box: {min: {x,y,z}, max: {x,y,z}, size: {x,y,z}}
                - num_points: int
                - frame_id: str
        """
        try:
            points, colors, frame_id = ws_manager.get_pointcloud(
                pointcloud_topic, timeout=timeout
            )
        except TimeoutError:
            return {
                "error": (
                    f"Timeout ({timeout}s) waiting for point cloud on "
                    f"{pointcloud_topic}. Did you call segment_objects() first?"
                )
            }
        except Exception as e:
            return {"error": f"Failed to read point cloud: {str(e)}"}

        if len(points) == 0:
            return {
                "object_name": object_name,
                "error": "Point cloud is empty — segmentation may have failed.",
            }

        # Compute centroid
        centroid = points.mean(axis=0)

        # Compute axis-aligned bounding box
        pt_min = points.min(axis=0)
        pt_max = points.max(axis=0)
        size = pt_max - pt_min

        return {
            "object_name": object_name,
            "centroid": {
                "x": round(float(centroid[0]), 4),
                "y": round(float(centroid[1]), 4),
                "z": round(float(centroid[2]), 4),
            },
            "bounding_box": {
                "min": {
                    "x": round(float(pt_min[0]), 4),
                    "y": round(float(pt_min[1]), 4),
                    "z": round(float(pt_min[2]), 4),
                },
                "max": {
                    "x": round(float(pt_max[0]), 4),
                    "y": round(float(pt_max[1]), 4),
                    "z": round(float(pt_max[2]), 4),
                },
                "size": {
                    "x": round(float(size[0]), 4),
                    "y": round(float(size[1]), 4),
                    "z": round(float(size[2]), 4),
                },
            },
            "num_points": len(points),
            "frame_id": frame_id,
        }

    @mcp.tool(
        description=(
            "Add a collision object (box) to the MoveIt planning scene.\n\n"
            "Publishes a CollisionObject to /planning_scene so MoveIt can\n"
            "plan around the object. The box is placed at the given position\n"
            "with the given dimensions.\n\n"
            "Use the bounding_box output from get_grasp_from_pointcloud()\n"
            "to get position and size values.\n\n"
            "Example usage:\n"
            "- add_collision_object('red_cube', 0.5, -0.1, 0.8, 0.05, 0.05, 0.05)\n"
        ),
    )
    def add_collision_object(
        object_name: str,
        x: float,
        y: float,
        z: float,
        size_x: float,
        size_y: float,
        size_z: float,
        frame_id: str = "arm_camera_color_optical_frame",
    ) -> dict:
        """Add a box collision object to the MoveIt planning scene.

        Args:
            object_name: Unique ID for the collision object.
            x: Box center X position in meters.
            y: Box center Y position in meters.
            z: Box center Z position in meters.
            size_x: Box size along X in meters.
            size_y: Box size along Y in meters.
            size_z: Box size along Z in meters.
            frame_id: TF frame the position is expressed in.

        Returns:
            dict with status of the operation.
        """
        # Build moveit_msgs/msg/CollisionObject
        collision_object = {
            "header": {
                "stamp": {"sec": 0, "nanosec": 0},
                "frame_id": frame_id,
            },
            "id": object_name,
            "operation": 0,  # ADD
            "primitives": [
                {
                    "type": 1,  # BOX
                    "dimensions": [float(size_x), float(size_y), float(size_z)],
                }
            ],
            "primitive_poses": [
                {
                    "position": {
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                    },
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "w": 1.0,
                    },
                }
            ],
        }

        # Wrap in PlanningScene message
        planning_scene_msg = {
            "world": {
                "collision_objects": [collision_object],
            },
            "is_diff": True,
        }

        try:
            ws_manager.publish(
                "/planning_scene",
                "moveit_msgs/msg/PlanningScene",
                planning_scene_msg,
            )
        except Exception as e:
            return {"error": f"Failed to publish collision object: {str(e)}"}

        return {
            "status": "SUCCESS",
            "object_name": object_name,
            "frame_id": frame_id,
            "position": {"x": x, "y": y, "z": z},
            "size": {"x": size_x, "y": size_y, "z": size_z},
            "description": (
                f"Collision object '{object_name}' added to planning scene "
                f"at ({x:.3f}, {y:.3f}, {z:.3f}) in frame '{frame_id}'."
            ),
        }
