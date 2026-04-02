"""WebSocket manager for rosbridge communication.

Handles subscribing to image/depth topics and receiving messages
from the ROS system via rosbridge websocket protocol.
"""

import base64
import json
import struct
import threading
import time
import uuid

import numpy as np
import websocket


class WebSocketManager:
    """Manages WebSocket connections to rosbridge for topic subscriptions."""

    def __init__(self, host: str, port: int, default_timeout: float = 10.0):
        self.host = host
        self.port = port
        self.default_timeout = default_timeout
        self._ws = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def _ensure_connected(self) -> websocket.WebSocket:
        """Ensure we have an active websocket connection."""
        with self._lock:
            if self._ws is None or not self._ws.connected:
                self._ws = websocket.create_connection(
                    self.url, timeout=self.default_timeout
                )
            return self._ws

    def subscribe_once(
        self, topic: str, msg_type: str = "", timeout: float = None
    ) -> dict:
        """Subscribe to a topic and return the first message received.

        Args:
            topic: ROS topic to subscribe to
            msg_type: ROS message type (optional)
            timeout: Timeout in seconds (default: self.default_timeout)

        Returns:
            dict: The received message
        """
        timeout = timeout or self.default_timeout
        ws = self._ensure_connected()

        sub_id = f"subscribe:{topic}:{uuid.uuid4().hex[:8]}"

        # Subscribe
        subscribe_msg = {
            "op": "subscribe",
            "id": sub_id,
            "topic": topic,
            "type": msg_type,
        }
        ws.send(json.dumps(subscribe_msg))

        # Wait for message
        ws.settimeout(timeout)
        try:
            while True:
                raw = ws.recv()
                data = json.loads(raw)
                if data.get("topic") == topic:
                    # Unsubscribe
                    unsub_msg = {"op": "unsubscribe", "id": sub_id, "topic": topic}
                    ws.send(json.dumps(unsub_msg))
                    return data.get("msg", {})
        except websocket.WebSocketTimeoutException:
            # Unsubscribe on timeout
            unsub_msg = {"op": "unsubscribe", "id": sub_id, "topic": topic}
            try:
                ws.send(json.dumps(unsub_msg))
            except Exception:
                pass
            raise TimeoutError(f"Timeout waiting for message on {topic}")

    def get_compressed_image(
        self, topic: str, timeout: float = None
    ) -> bytes:
        """Subscribe to a compressed image topic and return raw JPEG bytes.

        Args:
            topic: Compressed image topic (sensor_msgs/CompressedImage)
            timeout: Timeout in seconds

        Returns:
            bytes: Raw JPEG image data
        """
        msg = self.subscribe_once(
            topic, msg_type="sensor_msgs/msg/CompressedImage", timeout=timeout
        )
        # rosbridge base64-encodes the data field
        img_data = msg.get("data", "")
        return base64.b64decode(img_data)

    def get_depth_image(
        self, topic: str, timeout: float = None
    ) -> np.ndarray:
        """Subscribe to a depth image topic and return as numpy array.

        Args:
            topic: Depth image topic (sensor_msgs/Image)
            timeout: Timeout in seconds

        Returns:
            np.ndarray: Depth image as float32 array (meters)
        """
        msg = self.subscribe_once(
            topic, msg_type="sensor_msgs/msg/Image", timeout=timeout
        )

        encoding = msg.get("encoding", "32FC1")
        width = msg.get("width", 0)
        height = msg.get("height", 0)
        data = base64.b64decode(msg.get("data", ""))

        if encoding in ("32FC1", "float32"):
            depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
        elif encoding in ("16UC1", "mono16"):
            depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
            depth = depth.astype(np.float32) / 1000.0  # mm to meters
        else:
            raise ValueError(f"Unsupported depth encoding: {encoding}")

        return depth

    def get_camera_info(self, topic: str, timeout: float = None) -> dict:
        """Subscribe to a camera_info topic and return intrinsics.

        Args:
            topic: CameraInfo topic

        Returns:
            dict with keys: fx, fy, cx, cy, width, height
        """
        msg = self.subscribe_once(
            topic, msg_type="sensor_msgs/msg/CameraInfo", timeout=timeout
        )

        k_matrix = msg.get("k", msg.get("K", [0] * 9))
        return {
            "fx": k_matrix[0],
            "fy": k_matrix[4],
            "cx": k_matrix[2],
            "cy": k_matrix[5],
            "width": msg.get("width", 640),
            "height": msg.get("height", 480),
        }

    @staticmethod
    def _parse_pointcloud(msg: dict) -> tuple[np.ndarray, np.ndarray, str]:
        """Parse a rosbridge PointCloud2 message into numpy arrays.

        Args:
            msg: The rosbridge PointCloud2 message dict.

        Returns:
            Tuple of (points, colors, frame_id).
        """
        width = msg.get("width", 0)
        height = msg.get("height", 1)
        point_step = msg.get("point_step", 24)
        frame_id = msg.get("header", {}).get("frame_id", "")
        data = base64.b64decode(msg.get("data", ""))

        num_points = width * height
        if num_points == 0 or len(data) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), frame_id

        # Build field offset lookup from the message
        field_offsets = {}
        for field in msg.get("fields", []):
            field_offsets[field["name"]] = field["offset"]

        x_off = field_offsets.get("x", 0)
        y_off = field_offsets.get("y", 4)
        z_off = field_offsets.get("z", 8)
        rgb_off = field_offsets.get("rgb", 16)
        has_rgb = "rgb" in field_offsets

        points = np.empty((num_points, 3), dtype=np.float32)
        colors = np.empty((num_points, 3), dtype=np.float32)

        for i in range(num_points):
            base = i * point_step
            points[i, 0] = struct.unpack_from('<f', data, base + x_off)[0]
            points[i, 1] = struct.unpack_from('<f', data, base + y_off)[0]
            points[i, 2] = struct.unpack_from('<f', data, base + z_off)[0]

            if has_rgb:
                rgb_int = struct.unpack_from('<I', data, base + rgb_off)[0]
                colors[i, 0] = ((rgb_int >> 16) & 0xFF) / 255.0  # R
                colors[i, 1] = ((rgb_int >> 8) & 0xFF) / 255.0   # G
                colors[i, 2] = (rgb_int & 0xFF) / 255.0           # B
            else:
                colors[i] = [0.0, 0.0, 0.0]

        # Filter out invalid points (NaN, inf, zero)
        valid = np.isfinite(points).all(axis=1) & (np.abs(points).sum(axis=1) > 0)
        return points[valid], colors[valid], frame_id

    def get_pointcloud(
        self, topic: str, timeout: float = None
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Subscribe to a PointCloud2 topic and return points and colors.

        Args:
            topic: PointCloud2 topic
            timeout: Timeout in seconds

        Returns:
            Tuple of (points, colors, frame_id).
        """
        msg = self.subscribe_once(
            topic, msg_type="sensor_msgs/msg/PointCloud2", timeout=timeout
        )
        return self._parse_pointcloud(msg)

    def call_service(
        self, service: str, service_type: str, args: dict = None, timeout: float = None
    ) -> dict:
        """Call a ROS service via rosbridge and return the response.

        Args:
            service: ROS service name (e.g. '/buffer_server/lookup_transform')
            service_type: ROS service type (e.g. 'tf2_msgs/srv/LookupTransform')
            args: Service request arguments as dict
            timeout: Timeout in seconds (default: self.default_timeout)

        Returns:
            dict: The service response values

        Raises:
            TimeoutError: If the service does not respond in time
            RuntimeError: If the service call fails
        """
        timeout = timeout or self.default_timeout
        ws = self._ensure_connected()

        call_id = f"call_service:{service}:{uuid.uuid4().hex[:8]}"
        msg = {
            "op": "call_service",
            "id": call_id,
            "service": service,
            "type": service_type,
            "args": args or {},
        }
        ws.send(json.dumps(msg))

        ws.settimeout(timeout)
        try:
            while True:
                raw = ws.recv()
                data = json.loads(raw)
                if data.get("op") == "service_response" and data.get("id") == call_id:
                    if not data.get("result", True):
                        raise RuntimeError(
                            f"Service call to {service} failed: {data.get('values', {})}"
                        )
                    return data.get("values", {})
        except websocket.WebSocketTimeoutException:
            raise TimeoutError(f"Timeout calling service {service}")

    def send_action_goal(
        self, action_name: str, action_type: str, goal: dict, timeout: float = None
    ) -> dict:
        """Send an action goal via rosbridge and wait for the result.

        Args:
            action_name: ROS action name (e.g. '/tf2_buffer_server')
            action_type: ROS action type (e.g. 'tf2_msgs/action/LookupTransform')
            goal: Action goal as dict
            timeout: Timeout in seconds (default: self.default_timeout)

        Returns:
            dict: The action result

        Raises:
            TimeoutError: If the action does not complete in time
            RuntimeError: If the action fails
        """
        timeout = timeout or self.default_timeout
        ws = self._ensure_connected()

        goal_id = f"action:{action_name}:{uuid.uuid4().hex[:8]}"
        msg = {
            "op": "send_action_goal",
            "id": goal_id,
            "action": action_name,
            "action_type": action_type,
            "args": goal,
        }
        ws.send(json.dumps(msg))

        ws.settimeout(timeout)
        try:
            while True:
                raw = ws.recv()
                data = json.loads(raw)
                if data.get("id") == goal_id and data.get("op") == "action_result":
                    if not data.get("result", True):
                        raise RuntimeError(
                            f"Action {action_name} failed: {data.get('values', {})}"
                        )
                    return data.get("values", {})
        except websocket.WebSocketTimeoutException:
            raise TimeoutError(f"Timeout waiting for action {action_name}")

    def publish(self, topic: str, msg_type: str, msg: dict) -> None:
        """Publish a message to a ROS topic via rosbridge.

        Args:
            topic: ROS topic to publish to
            msg_type: ROS message type (e.g. 'moveit_msgs/msg/CollisionObject')
            msg: Message data as dict
        """
        ws = self._ensure_connected()
        pub_msg = {
            "op": "publish",
            "topic": topic,
            "type": msg_type,
            "msg": msg,
        }
        ws.send(json.dumps(pub_msg))

    def close(self):
        """Close the websocket connection."""
        with self._lock:
            if self._ws and self._ws.connected:
                self._ws.close()
            self._ws = None
