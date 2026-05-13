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


def _transform_points(points: np.ndarray, translation: dict, rotation: dict) -> np.ndarray:
    """Vectorised version of `_transform_point` for an Nx3 numpy array.

    Returns an Nx3 numpy array (no rounding — caller can decide).
    """
    R = _quat_to_rotation_matrix(
        rotation["x"], rotation["y"], rotation["z"], rotation["w"]
    )
    t = np.array([translation["x"], translation["y"], translation["z"]])
    return (R @ points.T).T + t


# Aspect ratio (long / short principal axis) below which the object is
# considered approximately rotationally symmetric in top-down view; for
# such shapes the grasp yaw is undefined and we fall back to the default
# top-down orientation. 1.2 chosen empirically: a cube projects to a
# square (ratio ~1.0); a 14cm coke can on its base also ~1.0; a shoe
# is typically 2.5-3.0. The threshold sits comfortably between.
_PCA_ASPECT_RATIO_MIN = 1.2


def _principal_axis_angle_xy(points_base: np.ndarray) -> tuple[float, float]:
    """Compute the principal-axis yaw and aspect ratio in the xy plane.

    Args:
        points_base: Nx3 array of points already transformed to the
            target frame (base_footprint). Only the (x, y) columns are
            used — z is dropped because grasp yaw is a planar question.

    Returns:
        ``(angle_long_rad, aspect_ratio)`` where ``angle_long_rad`` is
        the angle of the LONG axis in base xy (atan2 of the leading
        eigenvector), and ``aspect_ratio`` is ``sqrt(lambda_long /
        lambda_short)`` from the 2x2 covariance eigendecomposition.

        On degenerate input (<3 points) returns ``(0.0, 1.0)`` — the
        caller's aspect-ratio threshold will then fall back to plain
        top-down.
    """
    if points_base.shape[0] < 3:
        return 0.0, 1.0
    xy = points_base[:, :2] - points_base[:, :2].mean(axis=0)
    cov = np.cov(xy, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues; long axis is the LAST.
    lam_short, lam_long = float(eigvals[0]), float(eigvals[1])
    if lam_short <= 1e-9:
        return 0.0, 1.0
    long_axis = eigvecs[:, 1]
    angle_long = float(np.arctan2(long_axis[1], long_axis[0]))
    aspect_ratio = float(np.sqrt(lam_long / lam_short))
    return angle_long, aspect_ratio


def _shortest_grasp_yaw(angle_long_rad: float) -> float:
    """Wrap a long-axis angle to [-pi/2, pi/2] for minimum wrist rotation.

    The Robotiq 2F-140 is 180°-symmetric (rotating the gripper by pi
    around the approach axis produces the same physical orientation
    with fingers swapped), so any two yaws differing by pi are
    interchangeable. Choose the representative in [-pi/2, pi/2] so the
    wrist takes the shortest path from the default look_forward pose.
    """
    yaw = angle_long_rad
    while yaw > np.pi / 2:
        yaw -= np.pi
    while yaw < -np.pi / 2:
        yaw += np.pi
    return yaw


def _quat_multiply(q1: dict, q2: dict) -> dict:
    """Hamilton product q1 * q2 for two {x, y, z, w} quaternions.

    The resulting quaternion represents the rotation of applying q2
    first, then q1 (the convention that matches `v' = q ⊗ v ⊗ q*`).
    """
    x1, y1, z1, w1 = q1["x"], q1["y"], q1["z"], q1["w"]
    x2, y2, z2, w2 = q2["x"], q2["y"], q2["z"], q2["w"]
    return {
        "x": w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        "y": w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        "z": w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        "w": w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    }


def _oriented_topdown_quaternion(yaw_rad: float) -> dict:
    """Compose the canonical top-down flip with a base-frame yaw.

    The default `TOP_DOWN_ORIENTATION` rotates the gripper's tool frame
    180° about X (z-up → z-down, fingers along base -y). To rotate the
    fingers in the horizontal plane we apply a yaw about the BASE z
    axis AFTER the flip, so the operation in quaternion form is
    ``q_yaw * q_topdown``.

    Returns the composed {x, y, z, w} quaternion.
    """
    half = yaw_rad / 2.0
    q_yaw = {"x": 0.0, "y": 0.0, "z": float(np.sin(half)), "w": float(np.cos(half))}
    return _quat_multiply(q_yaw, TOP_DOWN_ORIENTATION)
