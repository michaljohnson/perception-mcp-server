"""Object detection tools using vision LLM (Claude or Qwen3-VL).

Provides tools for detecting objects in camera images and describing scenes.
"""

import io
import os

from fastmcp import FastMCP
from PIL import Image, ImageDraw

from perception_mcp.utils.websocket import WebSocketManager

# Directory for saving annotated images
CAMERA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "camera")


def register_detection_tools(
    mcp: FastMCP,
    ws_manager: WebSocketManager,
    vision_client,
    camera_topics: dict,
) -> None:
    """Register object detection tools."""

    @mcp.tool(
        description=(
            "Detect objects in the robot's camera view using vision AI.\n\n"
            "Can detect any object by name (e.g., 'scissors', 'brown basket', 'cube').\n"
            "Returns bounding boxes, confidence scores, and pixel coordinates.\n\n"
            "Example usage:\n"
            "- detect_objects(prompt='scissors')\n"
            "- detect_objects(prompt='scissors', camera='arm')  # use arm camera\n"
            "- detect_objects()  # detect all objects using front camera\n"
        ),
    )
    def detect_objects(prompt: str = "", camera: str = "front") -> dict:
        """Detect objects in the current camera image.

        Args:
            prompt: Object to search for (e.g., 'scissors', 'cube').
                    Leave empty to detect all visible objects.
            camera: Which camera to use - 'front' (default) or 'arm'.

        Returns:
            dict with detected objects, their bounding boxes, and pixel coordinates.
        """
        topics = camera_topics.get(camera, camera_topics["front"])
        rgb_topic = topics["rgb"]

        try:
            img_bytes = ws_manager.get_compressed_image(rgb_topic, timeout=10.0)
        except TimeoutError:
            return {"error": f"Timeout getting image from {rgb_topic}"}
        except Exception as e:
            return {"error": f"Failed to get image: {str(e)}"}

        try:
            result = vision_client.detect_objects(img_bytes, prompt=prompt)
        except Exception as e:
            return {"error": f"Vision API call failed: {str(e)}"}

        # Save annotated image
        _save_annotated_image(img_bytes, result.get("objects", []))

        return result

    @mcp.tool(
        description=(
            "Describe the scene visible from the robot's camera.\n\n"
            "Provides a detailed description including room type, visible objects,\n"
            "surfaces, and navigation hints. Useful for exploration and semantic search.\n\n"
            "Example usage:\n"
            "- describe_scene()  # What does the robot see? (front camera)\n"
            "- describe_scene(camera='arm')  # What does the arm camera see?\n"
        ),
    )
    def describe_scene(camera: str = "front") -> dict:
        """Describe the current scene from the robot's camera.

        Args:
            camera: Which camera to use - 'front' (default) or 'arm'.

        Returns:
            dict with scene description, room type, objects, surfaces,
            and navigation hints.
        """
        topics = camera_topics.get(camera, camera_topics["front"])
        rgb_topic = topics["rgb"]

        try:
            img_bytes = ws_manager.get_compressed_image(rgb_topic, timeout=10.0)
        except TimeoutError:
            return {"error": f"Timeout getting image from {rgb_topic}"}
        except Exception as e:
            return {"error": f"Failed to get image: {str(e)}"}

        try:
            return vision_client.describe_scene(img_bytes)
        except Exception as e:
            return {"error": f"Vision API call failed: {str(e)}"}


def _save_annotated_image(img_bytes: bytes, objects: list) -> None:
    """Save image with bounding boxes drawn on detected objects."""
    try:
        os.makedirs(CAMERA_DIR, exist_ok=True)
        img = Image.open(io.BytesIO(img_bytes))
        draw = ImageDraw.Draw(img)

        for obj in objects:
            bbox = obj.get("bbox")
            name = obj.get("name", "unknown")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1 - 12), name, fill="red")

        img.save(os.path.join(CAMERA_DIR, "detected.jpeg"), "JPEG")
    except Exception:
        pass  # Don't fail the tool if annotation fails
