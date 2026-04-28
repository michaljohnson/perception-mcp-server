"""Vision API wrapper for scene understanding (describe_scene tool).

Supports multiple backends, selected at startup via the VISION_BACKEND env
var:
- "anthropic": Anthropic-API-compatible vision models (Claude family).
- "openai":    OpenAI-API-compatible endpoints — hosted OpenAI, vLLM,
                Ollama, LM Studio, or any service that speaks the
                OpenAI Chat Completions API.
"""

import base64
import json
import re


def create_vision_client(
    backend: str = "anthropic",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
) -> "VisionClient":
    """Factory to create the appropriate vision client.

    Args:
        backend: "anthropic" or "openai"
        api_key: API key for the service
        model: Model ID (defaults per backend if empty)
        base_url: Base URL for OpenAI-compatible endpoints

    Returns:
        VisionClient instance
    """
    if backend == "openai":
        return OpenAIVisionClient(
            api_key=api_key,
            model=model or "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
            base_url=base_url or "http://127.0.0.1:8000/v1",
        )
    else:
        return AnthropicVisionClient(
            api_key=api_key,
            model=model or "claude-sonnet-4-20250514",
        )


# --------------------------------------------------------------------------
# Shared prompts
# --------------------------------------------------------------------------

def _scene_prompt() -> str:
    return (
        "You are a robot's perception system. Analyze this image from the robot's camera.\n\n"
        "Return a JSON object with:\n"
        '- "description": detailed scene description\n'
        '- "room_type": estimated room type (kitchen, bedroom, living_room, etc.)\n'
        '- "objects": list of visible object names\n'
        '- "surfaces": list of surfaces that objects could be placed on/picked from\n'
        '- "navigation_hints": suggestions for where to go or look next\n\n'
        "Return ONLY valid JSON, no markdown or extra text."
    )


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks and thinking tags."""
    text = text.strip()

    # Remove <think>...</think> blocks (Qwen3 thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Try to find JSON object in the response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from surrounding text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"error": "Failed to parse response", "raw_response": text[:500]}


# --------------------------------------------------------------------------
# Anthropic (Claude) backend
# --------------------------------------------------------------------------

class AnthropicVisionClient:
    """Vision client using Anthropic Claude API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _call(self, image_bytes: bytes, text: str, image_format: str = "jpeg") -> dict:
        b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_format}",
                                "data": b64_image,
                            },
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ],
        )
        return _parse_json_response(response.content[0].text)

    def describe_scene(self, image_bytes: bytes, image_format: str = "jpeg") -> dict:
        return self._call(image_bytes, _scene_prompt(), image_format)


# --------------------------------------------------------------------------
# OpenAI-compatible backend (Qwen3-VL, Ollama, vLLM, etc.)
# --------------------------------------------------------------------------

class OpenAIVisionClient:
    """Vision client using OpenAI-compatible API (for Qwen3-VL, etc.)."""

    def __init__(
        self,
        api_key: str = "dummy",
        model: str = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
        base_url: str = "http://127.0.0.1:8000/v1",
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _call(self, image_bytes: bytes, text: str, image_format: str = "jpeg") -> dict:
        b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{b64_image}",
                            },
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ],
        )
        return _parse_json_response(response.choices[0].message.content)

    def describe_scene(self, image_bytes: bytes, image_format: str = "jpeg") -> dict:
        return self._call(image_bytes, _scene_prompt(), image_format)
