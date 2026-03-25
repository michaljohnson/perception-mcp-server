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

    def close(self):
        """Close the websocket connection."""
        with self._lock:
            if self._ws and self._ws.connected:
                self._ws.close()
            self._ws = None
