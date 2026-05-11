"""TF, rotation, and gripper-geometry helpers shared by the grasping
and placing tools.

In ROS 2 Jazzy, `LookupTransform` is an action (not a service), so we
talk to a tf2 buffer_server action via the WebSocket manager. We use a
namespaced canonical instance at `/canonical_tf/tf2_buffer_server` so
the lookup doesn't race against the five default-named buffer servers
spawned by other rclpy components (MoveIt, rviz, segmentation nodes,
etc.). See `_tf_lookup` for the rationale. The pose-computation tools
(`get_topdown_grasp_pose`, `get_topdown_placing_pose`) call `_tf_lookup`
to bring camera-frame point clouds into the robot's `base_footprint`
frame.

`TOP_DOWN_ORIENTATION` is the quaternion for a strict top-down gripper
approach: 180° rotation about X (the EEF frame has Z-up at rest, so we
flip to Z-down). Used by both grasp and drop poses.

`GRIPPER_FINGER_OFFSET_M` is the distance from the gripper's wrist
frame to the fingertips. MoveIt plans poses for the wrist link, so any
tool computing a fingertip target needs to subtract / add this offset
to convert between wrist and fingertip space.
"""

import numpy as np

# Top-down approach orientation: gripper / drop pose pointing straight
# down. 180° rotation about X axis.
TOP_DOWN_ORIENTATION = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}

# Robotiq 2F-140 finger length offset (wrist frame → fingertip).
# When the wrist is at z = wrist_z (top-down), the fingertips reach
# down to z = wrist_z - GRIPPER_FINGER_OFFSET_M.
GRIPPER_FINGER_OFFSET_M = 0.14


def _quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def _tf_lookup(
    ws_manager,
    source_frame: str,
    target_frame: str = "base_footprint",
    timeout: float = 5.0,
) -> tuple[dict, dict]:
    """Look up a TF transform via the canonical buffer server action.

    Calls `/canonical_tf/tf2_buffer_server` (namespaced) instead of the
    default `/tf2_buffer_server` to avoid a race condition: MoveIt, rviz,
    moveit_mcp_server, the segmentation nodes, and other rclpy components
    each spawn their own `tf2_ros.Buffer.create_server()` advertising the
    default action name with independent Buffer caches. ROS 2 discovery
    routes a goal request to whichever one binds first, producing
    intermittent wrong transforms (camera-frame depth leaking through
    as base-frame z, etc.).

    The canonical instance is launched by start_mcp_servers.sh under
    namespace /canonical_tf with use_sim_time:=true.

    Returns:
        (translation, rotation) where translation is a dict {x, y, z}
        and rotation is a dict {x, y, z, w}. Raises on lookup failure.
    """
    result = ws_manager.send_action_goal(
        action_name="/canonical_tf/tf2_buffer_server",
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


def _transform_point(point: dict, translation: dict, rotation: dict) -> dict:
    """Transform a 3D point dict via a TF translation + quaternion rotation.

    Args:
        point: {x, y, z} in source frame.
        translation: {x, y, z} from `_tf_lookup`.
        rotation: {x, y, z, w} quaternion from `_tf_lookup`.

    Returns:
        {x, y, z} in target frame, rounded to 4 decimals.
    """
    R = _quat_to_rotation_matrix(
        rotation["x"], rotation["y"], rotation["z"], rotation["w"]
    )
    p = np.array([point["x"], point["y"], point["z"]])
    t = np.array([translation["x"], translation["y"], translation["z"]])
    out = R @ p + t
    return {
        "x": round(float(out[0]), 4),
        "y": round(float(out[1]), 4),
        "z": round(float(out[2]), 4),
    }
