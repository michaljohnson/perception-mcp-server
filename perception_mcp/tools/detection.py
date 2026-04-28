"""Camera-image tool — returns raw front/arm/both camera frames.

The agent does its own visual reasoning on the pixels (it's a multimodal
model). No inner LLM call is made here — we just hand back JPEG bytes.
"""

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from perception_mcp.utils.websocket import WebSocketManager


def register_detection_tools(
    mcp: FastMCP,
    ws_manager: WebSocketManager,
    camera_topics: dict,
) -> None:
    """Register the camera-look tool."""

    @mcp.tool(
        description=(
            "Return raw camera frame(s) as JPEG image(s) so the calling\n"
            "agent can reason on pixels directly with its own vision\n"
            "capability.\n\n"
            "  - camera='front': body-mounted camera, sees the room and\n"
            "    the robot's own arm.\n"
            "  - camera='arm':   wrist-mounted camera, sees what the\n"
            "    gripper is reaching for.\n"
            "  - camera='both':  front + arm back-to-back in one call.\n"
            "    Useful for area / room judgments where the two angles\n"
            "    complement each other.\n\n"
            "Use for area verification after navigation, gripper-state\n"
            "checks after pick / place, and any 'what does the robot\n"
            "actually see right now?' question."
        ),
    )
    def look(camera: str = "front") -> list[Image] | Image:
        """Return raw camera frame(s) as Image content block(s).

        Args:
            camera: 'front' | 'arm' | 'both'. Default 'front'.

        Returns:
            A single Image for 'front' / 'arm', or a list[Image] of
            [front, arm] for 'both'.
        """
        if camera == "both":
            cams = ["front", "arm"]
        elif camera in camera_topics:
            cams = [camera]
        else:
            raise ValueError(
                f"Invalid camera {camera!r}: choose from "
                f"'front', 'arm', or 'both'"
            )

        images: list[Image] = []
        for c in cams:
            rgb_topic = camera_topics[c]["rgb"]
            img_bytes = ws_manager.get_compressed_image(rgb_topic, timeout=10.0)
            images.append(Image(data=img_bytes, format="jpeg"))

        return images if len(images) > 1 else images[0]
