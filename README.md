# perception-mcp-server

An [MCP](https://modelcontextprotocol.io) server that exposes a mobile-manipulation robot's perception primitives — segmentation, grasp planning, drop-pose planning, and scene description — as tools an LLM agent can call.

It is the perception layer in a larger stack where an LLM-driven agent uses MCP tools to drive a real (or simulated) robot through pick / place / navigate tasks. Other layers in the stack (exposed as separate MCP servers) typically handle motion planning (MoveIt), navigation (nav2), and direct ROS topic access.

## What problem this solves

To pick or place an object with a manipulator, an agent needs to translate "the object I want is somewhere in the camera view" into "a pose `(x, y, z, qx, qy, qz, qw)` in the robot's base frame that the motion planner can plan to." That requires:

1. **Segmentation** — find the object in the image and produce a 3D point cloud of just that object.
2. **Frame transform** — convert the cloud from the camera's optical frame into the robot's base frame (so a planner that thinks in base coordinates can use it).
3. **Pose computation** — turn the point cloud into a single pose, with appropriate offsets for gripper geometry, drop clearance, etc.
4. **Scene-level reasoning** — for navigation, a coarser "what room am I in, what's around me?" check.

This server packages those four operations as four MCP tools.

## Tool overview

| Tool | Returns | When to use |
|---|---|---|
| `segment_objects` | mask + segmented point cloud (cached internally) | Always first — feeds the cache the other tools read. |
| `get_grasp_from_pointcloud` | top-down `grasp_pose` in robot base frame | After `segment_objects` on the **arm** camera. For picking. |
| `get_topdown_drop_pose` | top-down `place_pose` in robot base frame | After `segment_objects`. For placing onto a surface or into a container. |
| `describe_scene` | room type, objects, surfaces, navigation hints | Coarse high-level scene check from a forward-facing camera. For arrival verification, not manipulation. |

### Sync execution

All four tools are plain `def`. We briefly wrapped `segment_objects` in `asyncio.to_thread` to free the FastMCP event loop during its multi-second blocking I/O, but reverted because it added a duplicated wrapper / impl pair without any observable improvement under sequential agent flows. If you ever add concurrent perception calls (e.g. parallel arm + front segmentation), revisit this — but for now sync is simpler and equivalent.

### Typical call sequence

A complete pick attempt looks like this from the agent's side:

```
segment_objects(prompt="red cup", camera="arm")
  → status: SUCCESS, point cloud cached
get_grasp_from_pointcloud(object_name="red cup")
  → grasp_pose at (0.65, -0.10, 0.42), top-down orientation
[hand off to a motion-planning MCP to actually move the arm]
```

A pick + place sequence adds:

```
segment_objects(prompt="trash bin", camera="arm")
  → status: SUCCESS
get_topdown_drop_pose(object_name="trash bin", top_clearance_m=0.35)
  → place_pose 35 cm above the highest point of the bin
[hand off to motion planning to release the held object there]
```

---

## `segment_objects(prompt, camera="arm", timeout=30.0)`

Sends `prompt` (free-form text, e.g. `"red cup"`, `"black scissors"`, `"tall rectangular bin"`) to a remote SAM3-style segmentation pipeline running as a ROS node. The node combines a vision-language detector (e.g. GroundingDINO) with a segmentation model (SAM 2 / SAM 3) to produce a pixel mask plus a 3D point cloud of just the requested object.

The actual segmentation runs **outside this server**. This MCP tool publishes the prompt to a ROS topic, waits for a status reply, and captures the resulting point cloud over a dedicated websocket subscription.

### Two cameras

Most mobile manipulators have one camera looking forward (for navigation) and one camera on the wrist (for close-range manipulation). This server supports both via separate ROS nodes:

- `camera="arm"` — wrist-mounted camera. Use for grasping.
- `camera="front"` — body-mounted forward camera. Use for navigation / approach verification.

The two cameras are served by separate segmentation nodes publishing on disjoint topic prefixes:

| | Arm | Front |
|---|---|---|
| Trigger | `/segment_text` | `/front/segment_text` |
| Status | `/segmentation_status` | `/front/segmentation_status` |
| Mask | `/segmentation_mask` | `/front/segmentation_mask` |
| Point cloud | `/segmented_pointcloud` | `/front/segmented_pointcloud` |

### Internal cache (the key design choice)

On `SUCCESS` the tool caches:
- `points`, `colors`, `frame_id` of the segmented cloud
- `tf_translation`, `tf_rotation` from the camera optical frame to the robot base frame, **at the instant of segmentation**
- diagnostics (`prompt`, `camera`, `timestamp`)

The TF snapshot matters: a downstream `get_grasp_from_pointcloud` call may fire after the arm has moved, but the snapshot pins the transform that was valid for *those specific points*. A live TF lookup at compute time would silently apply the wrong transform.

### Returns

```json
{"status": "SUCCESS" | "NO_OBJECTS_FOUND" | "<other>",
 "prompt": "<prompt>", "camera": "arm" | "front",
 "outputs": {"mask_topic": "...", "pointcloud_topic": "..."},
 "description": "..."}
```

### Prerequisites

- `rosbridge_websocket` running and reachable.
- A segmentation ROS node running for whichever camera you query (arm + front nodes if you want both).
- A backend segmentation server (a Grounding-DINO + SAM HTTP service) reachable from the segmentation ROS node. URL is configured on the ROS node side, not in this server.

---

## `get_grasp_from_pointcloud(object_name, pointcloud_topic="/segmented_pointcloud", timeout=10.0)`

Computes a top-down grasp pose from the cached point cloud (or, as a fallback, from a fresh subscription on `pointcloud_topic`). Pure numpy — no learned model, no normal estimation, no clustering.

### Algorithm

1. Read point cloud from the cache (default) or the provided topic.
2. `centroid = points.mean(axis=0)`; bounding box is `(min, max, max-min)`.
3. TF-transform the centroid from camera optical frame to base frame, using the cached snapshot when available, otherwise a live `_tf_lookup` via the `/tf2_buffer_server` action.
4. Apply a vertical gripper-finger offset to z (default 14 cm — Robotiq 2F-140; adjust at the top of `grasping.py` if your gripper differs).
5. Set orientation to strict top-down `(x=1, y=0, z=0, w=0)`.

### Returns (success)

```json
{"object_name": "...",
 "centroid_camera_frame": {"x":..,"y":..,"z":..},
 "centroid_base_frame":   {"x":..,"y":..,"z":..},
 "grasp_pose": {"frame_id":"base_footprint",
                "position":{"x":..,"y":..,"z":..},
                "orientation":{"x":1,"y":0,"z":0,"w":0}},
 "bounding_box": {"min":{...}, "max":{...}, "size":{...}},
 "num_points": N, "camera_frame_id":"...", "gripper_offset_m":0.14}
```

If TF fails, the response includes `centroid_camera_frame` only and a `warning` field — callers should treat this as a failure (no `grasp_pose` to plan to).

### Limitations

- **Top-down grasps only.** No yaw / approach direction reasoning. If the object lies flat on a surface and is grippable from above, this works well; if not, you need a richer pose computation than this primitive offers.
- **Centroid-based.** Works for compact, roughly symmetric objects (cups, balls, small tools). For long or very irregular objects (a rake, a coiled cable) the centroid is not the right grasp point.

### Prerequisites

Call `segment_objects(camera="arm", ...)` first. The cache is camera-agnostic, but front-camera grasps are unreliable in practice — the front camera's geometry is not optimized for close-range manipulation.

---

## `get_topdown_drop_pose(object_name, top_clearance_m=0.20, ...)`

Computes a top-down drop pose using simple statistics on the cached (or raw) point cloud. The same algorithm handles surfaces (drop ON) and containers (drop INTO) — only the vertical clearance differs.

### Algorithm

1. Read point cloud from the cache (default) or the provided raw depth topic.
2. Optional xy crop (raw-depth mode): keep only points within `crop_radius_m` of `(crop_center_x, crop_center_y)` in base frame.
3. `cx, cy = mean(points.xy)`. The horizontal center of the visible region.
4. `top_z = 95th percentile of points.z`. The "top" of whatever is in front of you. p95 instead of max so a stray noisy point doesn't lift the drop pose into the air.
5. `drop_z = top_z + top_clearance_m`. Adds vertical safety margin.
6. Orientation: strict top-down.

### Clearance

- **Surface** (table, counter, shelf, desk): `top_clearance_m = 0.20`. Accounts for gripper finger length + a small safety gap.
- **Container** (bin, basket, bowl, drainer, box): `top_clearance_m = 0.35`. Drops the held object cleanly into the opening rather than scraping the rim.

### Modes

- `use_cached=True` (default) — use the SAM3 cloud cached by `segment_objects()`.
- `use_cached=False` — read a raw depth topic. **Must combine with `crop_center_x/y`** so the algorithm only looks at the target region. Useful when SAM3 fails on the target view (e.g. a bin rim seen straight down) and you can supply a coarse target xy from a previous step.

### Returns (success)

```json
{"object_name": "...",
 "surface_height_m": <top_z>,
 "surface_centroid": {"x":cx,"y":cy,"z":top_z},
 "place_pose": {"frame_id":"base_footprint",
                "position":{"x":cx,"y":cy,"z":drop_z},
                "orientation":{"x":1,"y":0,"z":0,"w":0}},
 "top_clearance_m": <clearance>,
 "method": "mean_xy_p95_z",
 "num_points_used": N, ...}
```

### Limitations

- `mean(xy)` is biased toward whichever side of the target the camera sees more of. A bin viewed horizontally has its near rim oversampled, pulling the centroid forward. Mitigation: either drive close enough that the bias is small, or position the arm directly above the coarse target and re-segment from above.
- Raw-depth mode (`use_cached=False`) bypasses SAM3 — useful as a fallback but loses the object-level reasoning, so the crop must be tight.

---

## `describe_scene()`

Captures one frame from the front camera, sends it to the configured vision LLM (Anthropic, OpenAI-compatible, etc.), and returns a structured scene description.

### Returns

```json
{"description": "...",
 "room_type": "kitchen" | "bedroom" | "living_room" | ...,
 "objects": ["...", ...],
 "surfaces": ["...", ...],
 "navigation_hints": "..."}
```

### When to use

Coarse semantic check after a navigation goal completes — does the robot see what the agent expected to see? Treat this as a sanity gate, not a precise perception primitive: vision LLMs can return inconsistent `room_type` labels across viewing angles. If you need a strict check, gate on the `objects` list rather than `room_type`.

This tool is **not** appropriate for grasp / drop planning — its output is text descriptions, not pixel-accurate masks or 3D points.

### Vision backends

- `VISION_BACKEND=anthropic` — any Anthropic vision-capable model (e.g. Claude 4.x).
- `VISION_BACKEND=openai` — any OpenAI-compatible endpoint: hosted OpenAI, vLLM, Ollama, LM Studio, etc.

The backend is wired in `perception_mcp/utils/vision.py`. Adding a third backend means subclassing `VisionClient` and registering a factory branch in `create_vision_client`.

---

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `ROSBRIDGE_IP` | `127.0.0.1` | rosbridge host. |
| `ROSBRIDGE_PORT` | `9090` | rosbridge port. |
| `VISION_BACKEND` | `openai` | `anthropic` or `openai`. |
| `VISION_API_KEY` | `dummy` | API key for the chosen vision provider. Provider-agnostic. |
| `VISION_MODEL` | backend-specific default | Any model id supported by the chosen backend. |
| `VISION_BASE_URL` | `http://127.0.0.1:8000/v1` | Used only when `VISION_BACKEND=openai`. |
| `SAM3_REMOTE_URL` | (unset) | Optional health check at startup, so a missing segmentation backend fails loud rather than silently inside `segment_objects`. |

`load_dotenv` reads `.env` at the server root if present.

## Running

```bash
# Default (OpenAI-compatible endpoint, e.g. vLLM)
python server.py --transport streamable-http --port 8003

# Anthropic Claude
VISION_BACKEND=anthropic VISION_API_KEY=sk-ant-... python server.py

# Local Ollama
VISION_BACKEND=openai VISION_BASE_URL=http://localhost:11434/v1 python server.py
```

## Installation

```bash
pip install -e .
```

The server depends on `fastmcp`, `numpy`, `opencv-python`, `websocket-client`, `python-dotenv`, plus `anthropic` and/or `openai` for whichever vision backend you select. See `pyproject.toml` for exact versions.

## Prerequisites at runtime

- `rosbridge_websocket` reachable on `ROSBRIDGE_IP:PORT`.
- `tf2_buffer_server` action server running, so TF lookups succeed.
- For `segment_objects`: at least one segmentation ROS node running (arm camera, front camera, or both), backed by a reachable segmentation server.
- For `describe_scene`: a reachable vision-LLM endpoint and a valid `VISION_API_KEY` for paid providers.

## Architecture notes

- The server is a **thin MCP wrapper** around ROS topics, ROS actions, and a vision LLM. It does no learning of its own; the heavy lifting (SAM, GroundingDINO, the LLM) runs elsewhere.
- The shared `segmentation_cache` and the `WebSocketManager` connection are **not thread-safe under parallel calls.** Currently safe because typical agent flows call perception sequentially; if you intentionally parallelize (e.g. concurrent arm + front segmentation), add a `threading.Lock` around cache writes and either partition the cache per-camera or wrap the websocket manager.
- All TF lookups go through the `/tf2_buffer_server` action (LookupTransform is an action in ROS 2 Jazzy and later, not a service).
- Pose computation tools (`get_grasp_from_pointcloud`, `get_topdown_drop_pose`) intentionally use plain numpy. They were rewritten from a heavier Open3D + DBSCAN + normal-filter pipeline; the simpler statistics performed equally well in practice and freed ~300 MB of dependencies plus a substantial chunk of process RAM.

## Layout

```
perception-mcp-server/
├── server.py                       # entry point; CLI args + transport
├── pyproject.toml
├── perception_mcp/
│   ├── main.py                     # tool registration + health checks
│   ├── tools/
│   │   ├── segmentation.py         # segment_objects (async)
│   │   ├── grasping.py             # get_grasp_from_pointcloud + _tf_lookup helper
│   │   ├── placing.py              # get_topdown_drop_pose
│   │   └── detection.py            # describe_scene
│   └── utils/
│       ├── websocket.py            # WebSocketManager: rosbridge / TF2 / topic I/O
│       └── vision.py               # AnthropicVisionClient / OpenAIVisionClient
```

## Limitations

- **Top-down only.** Grasp and drop poses are always strictly top-down `(1,0,0,0)`. Side / angled approaches are not supported by these primitives.
- **Single-cache design.** `segment_objects` overwrites a single internal cache. There is no per-camera cache, no history, and no thread-safety. Sequential agent flows are fine; concurrent flows need locks (see Architecture notes).
- **Coupled to ROS 2 (rosbridge) for I/O.** Replacing rosbridge with native ROS 2 client libs (`rclpy`) is possible but not done; rosbridge is convenient for cross-process tool integration but adds latency.
- **`describe_scene` is opinionated.** The JSON schema (`room_type`, `objects`, `surfaces`, `navigation_hints`) is hard-coded in the prompt; if you want different fields, edit `_scene_prompt()` in `vision.py`.

## License

(Add your license of choice here, e.g. Apache-2.0, MIT.)
