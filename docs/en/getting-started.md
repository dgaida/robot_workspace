# Quick Reference Guide

Essential commands and patterns for the Robot Workspace package.

---

## Table of Contents

- [Installation](#installation)
- [Core Imports](#core-imports)
- [Pose Objects](#pose-objects)
- [Workspaces](#workspaces)
- [Objects](#objects)
- [Spatial Queries](#spatial-queries)
- [Serialization](#serialization)
- [Common Patterns](#common-patterns)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# Basic installation
pip install -e .

# With Niryo support
pip install -e ".[niryo]"

# Development installation
pip install -e ".[dev]"

# All features
pip install -e ".[all]"
```

---

## Core Imports

```python
# Essential classes
from robot_workspace import (
    PoseObjectPNP,           # 6-DOF pose
    Object,                  # Detected object
    Objects,                 # Object collection
    Location,                # Spatial relationships
    Workspace,               # Abstract workspace
    Workspaces,              # Workspace collection
    NiryoWorkspace,          # Niryo implementation
    NiryoWorkspaces,         # Niryo collection
    WidowXWorkspace,         # WidowX implementation
    WidowXWorkspaces,        # WidowX collection
    ConfigManager,           # Configuration management
)

# For mocking (testing/demos)
from unittest.mock import Mock
```

---

## Pose Objects

### Create a Pose

```python
# Create 6-DOF pose
pose = PoseObjectPNP(
    x=0.2,      # meters
    y=0.1,      # meters
    z=0.3,      # meters
    roll=0.0,   # radians
    pitch=1.57, # radians (~90°)
    yaw=0.0     # radians
)
```

### Pose Operations

```python
# Arithmetic
new_pose = pose1 + pose2
difference = pose1 - pose2

# Copy with offsets
offset_pose = pose.copy_with_offsets(
    x_offset=0.05,
    yaw_offset=0.5
)

# Convert to list
pose_list = pose.to_list()  # [x, y, z, roll, pitch, yaw]

# Get transformation matrix
matrix = pose.to_transformation_matrix()  # 4x4 numpy array

# Get quaternion
quat = pose.quaternion  # [qx, qy, qz, qw]

# Check equality
is_equal = pose1 == pose2
is_close = pose1.approx_eq(pose2, eps_position=0.1, eps_orientation=0.1)
```

---

## Workspaces

### Initialize with Mock Environment

```python
from unittest.mock import Mock

# Create mock environment
def create_mock_env():
    env = Mock()
    env.use_simulation.return_value = True

    def mock_transform(ws_id, u_rel, v_rel, yaw):
        x = 0.4 - u_rel * 0.3
        y = 0.15 - v_rel * 0.3
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    env.get_robot_target_pose_from_rel = mock_transform
    return env

# Create workspaces
env = create_mock_env()
workspaces = NiryoWorkspaces(env, verbose=False)
```

### Initialize with Configuration

```python
from robot_workspace import NiryoWorkspaces, ConfigManager

# Using configuration file
workspaces = NiryoWorkspaces(
    environment,
    config_path='config/niryo_config.yaml'
)

# Or manually with ConfigManager
config_mgr = ConfigManager()
config_mgr.load_from_yaml('config/niryo_config.yaml')
ws_config = config_mgr.get_workspace_config('niryo_ws')
workspace = NiryoWorkspace.from_config(ws_config, environment)
```

### Access Workspaces

```python
# Get home workspace
home = workspaces.get_home_workspace()

# Get by ID
ws = workspaces.get_workspace_by_id("niryo_ws")

# Get by index
ws = workspaces.get_workspace(0)

# Get all IDs
ids = workspaces.get_workspace_ids()

# Get workspace properties
width = workspace.width_m()
height = workspace.height_m()
center = workspace.xy_center_wc()
obs_pose = workspace.observation_pose()
```

### Coordinate Transformation

```python
# Transform relative image coords to world coords
world_pose = workspace.transform_camera2world_coords(
    workspace_id="niryo_ws",
    u_rel=0.5,  # [0-1] horizontal
    v_rel=0.5,  # [0-1] vertical
    yaw=0.0     # object orientation
)
```

---

## Objects

### Create an Object

```python
# Without mask
obj = Object(
    label="pencil",
    u_min=100, v_min=100,  # Bounding box
    u_max=200, v_max=200,
    mask_8u=None,
    workspace=workspace
)

# With segmentation mask
import numpy as np
mask = np.zeros((640, 480), dtype=np.uint8)
mask[100:200, 100:200] = 255

obj_with_mask = Object(
    label="cube",
    u_min=100, v_min=100,
    u_max=200, v_max=200,
    mask_8u=mask,
    workspace=workspace
)
```

### Access Object Properties

```python
# Basic properties
label = obj.label()
position = obj.coordinate()  # [x, y]
x, y = obj.x_com(), obj.y_com()
full_pose = obj.pose_com()

# Dimensions
width = obj.width_m()
height = obj.height_m()
area = obj.size_m2()

# Orientation
rotation = obj.gripper_rotation()  # radians
```

### Update Object Position

```python
# Update XY position
obj.set_position([0.3, 0.05])

# Update full pose (including rotation)
new_pose = obj.pose_com().copy_with_offsets(
    x_offset=0.1,
    yaw_offset=0.5
)
obj.set_pose_com(new_pose)
```

---

## Spatial Queries

### Create Collection

```python
objects = Objects([obj1, obj2, obj3])
```

### Location-Based Queries

```python
# Find objects by location
left_objects = objects.get_detected_objects(
    location=Location.LEFT_NEXT_TO,
    coordinate=[0.25, 0.0]
)

right_objects = objects.get_detected_objects(
    location=Location.RIGHT_NEXT_TO,
    coordinate=[0.25, 0.0]
)

above_objects = objects.get_detected_objects(
    location=Location.ABOVE,
    coordinate=[0.25, 0.0]
)

below_objects = objects.get_detected_objects(
    location=Location.BELOW,
    coordinate=[0.25, 0.0]
)

# Within 2cm radius
nearby = objects.get_detected_objects(
    location=Location.CLOSE_TO,
    coordinate=[0.25, 0.0]
)
```

### Label-Based Queries

```python
# Find by label
pens = objects.get_detected_objects(label="pen")

# Combine location and label
left_pens = objects.get_detected_objects(
    location=Location.LEFT_NEXT_TO,
    coordinate=[0.25, 0.0],
    label="pen"
)

# Get specific object
obj = objects.get_detected_object(
    coordinate=[0.2, 0.1],
    label="pencil"
)
```

### Distance and Size Queries

```python
# Find nearest
nearest, distance = objects.get_nearest_detected_object([0.2, 0.1])

# Find largest/smallest
largest, size = objects.get_largest_detected_object()
smallest, size = objects.get_smallest_detected_object()

# Sort by size
sorted_asc = objects.get_detected_objects_sorted(ascending=True)
sorted_desc = objects.get_detected_objects_sorted(ascending=False)
```

---

## Serialization

### Object Serialization

```python
# To dictionary
obj_dict = obj.to_dict()

# To JSON string
json_str = obj.to_json()

# From dictionary
reconstructed = Object.from_dict(obj_dict, workspace)

# From JSON string
reconstructed = Object.from_json(json_str, workspace)
```

### Collection Serialization

```python
# To list of dicts
dict_list = Objects.objects_to_dict_list(objects)

# From list of dicts
reconstructed = Objects.dict_list_to_objects(dict_list, workspace)

# Get serializable results directly
obj_dict = objects.get_detected_object(
    [0.2, 0.1],
    serializable=True
)

sorted_dicts = objects.get_detected_objects_sorted(
    serializable=True
)
```

---

## Common Patterns

### Pick and Place Workflow

```python
# 1. Initialize
workspaces = NiryoWorkspaces(environment)
workspace = workspaces.get_home_workspace()

# 2. Detect objects
detected = Objects([obj1, obj2, obj3])

# 3. Find target
target = detected.get_detected_object(
    coordinate=[0.2, 0.1],
    label="cube"
)

# 4. Get pickup info
pickup_pose = target.pose_com()
rotation = target.gripper_rotation()

# 5. Execute pick (using robot API)
# robot.pick(pickup_pose, rotation)

# 6. Define place location
place_pose = PoseObjectPNP(0.3, -0.1, 0.05, 0.0, 1.57, 0.0)

# 7. Execute place
# robot.place(place_pose)

# 8. Update object in memory
target.set_pose_com(place_pose)
```

### Sorting by Size

```python
# Get sorted objects
sorted_objs = objects.get_detected_objects_sorted(ascending=True)

# Place in order
for i, obj in enumerate(sorted_objs):
    place_pose = PoseObjectPNP(
        x=0.3,
        y=-0.1 + i * 0.05,  # 5cm spacing
        z=0.05,
        roll=0.0,
        pitch=1.57,
        yaw=obj.gripper_rotation()
    )
    # robot.pick_and_place(obj.pose_com(), place_pose)
    obj.set_pose_com(place_pose)
```

### Multi-Workspace Transfer

```python
# Get workspaces
left_ws = workspaces.get_workspace_left()
right_ws = workspaces.get_workspace_right()

# Get objects in each
left_objects = Objects([...])
right_objects = Objects([...])

# Transfer object
transfer_obj = left_objects.get_detected_object([0.2, 0.1])
if transfer_obj:
    left_objects.remove(transfer_obj)

    # New position in right workspace
    new_pose = right_ws.transform_camera2world_coords(
        right_ws.id(), 0.5, 0.5, 0.0
    )
    transfer_obj.set_pose_com(new_pose)
    right_objects.append(transfer_obj)
```

### LLM Integration

```python
# Format for LLM
description = obj.as_string_for_llm()
# Output: "- 'pencil' at world coordinates [0.20, 0.10] with..."

# Format for chat
chat_msg = obj.as_string_for_chat_window()
# Output: "Detected a new object: pencil at..."

# Get all objects as text
for obj in objects:
    print(obj.as_string_for_llm())
```

---

## Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=robot_workspace --cov-report=term

# Specific test file
pytest tests/objects/test_object.py

# Specific test
pytest tests/objects/test_object.py::test_object_initialization

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Run Demo

```bash
# Run main demo (no hardware needed)
python main.py
```

---

## Troubleshooting

### Common Issues

```python
# Import error
# Solution: Install in editable mode
pip install -e .

# Missing workspace image shape
try:
    obj = Object(...)
except ValueError as e:
    print("Set workspace image shape first")
    workspace.set_img_shape((640, 480, 3))

# Object not found
obj = objects.get_detected_object([0.2, 0.1])
if obj is None:
    print("No object at that location")

# Workspace not visible
visible_ws = workspaces.get_visible_workspace(camera_pose)
if visible_ws is None:
    print("No workspace visible from camera")
```

### Debug Mode

```python
# Enable verbose output
workspaces = NiryoWorkspaces(environment, verbose=True)
workspace = NiryoWorkspace("niryo_ws", environment, verbose=True)
obj = Object(..., verbose=True)
objects = Objects([...], verbose=True)
```

---

## Coordinate Systems Reference

```
Image Coordinates (pixels)
    ↓ normalize by image dimensions
Relative Coordinates (0-1)
    ↓ transform_camera2world_coords()
World Coordinates (meters)
```

### Niryo Coordinate System

```
Y (left/right)
↑
│    Workspace
│   ┌─────────┐
│   │         │
│   │    •    │  ← Center
│   │         │
│   └─────────┘
└──────────────→ X (forward/back)

Z (up/down) ⊙ (out of page)
```

### Location Filters

- `LEFT_NEXT_TO`: y > coordinate[1]
- `RIGHT_NEXT_TO`: y < coordinate[1]
- `ABOVE`: x > coordinate[0]
- `BELOW`: x < coordinate[0]
- `CLOSE_TO`: within 2cm radius

---

## Quick Command Reference

| Task | Command |
|------|---------|
| Install package | `pip install -e .` |
| Run tests | `pytest` |
| Run demo | `python main.py` |
| Format code | `black .` |
| Lint code | `ruff check .` |
| Type check | `mypy robot_workspace` |
| Generate coverage | `pytest --cov --cov-report=html` |

---

## File Structure

```
robot_workspace/
├── objects/              # Object representation
│   ├── pose_object.py   # 6-DOF pose
│   ├── object.py        # Single object
│   ├── objects.py       # Collection
│   └── object_api.py    # API interface
├── workspaces/          # Workspace management
│   ├── workspace.py     # Abstract base
│   ├── workspaces.py    # Collection
│   ├── niryo_*.py       # Niryo implementation
│   └── widowx_*.py      # WidowX implementation
├── config.py            # Configuration
└── common/              # Utilities
    └── logger.py        # Logging
```

---

## Links

- **Documentation**: [docs/README.md](README.md)
- **API Reference**: [docs/api.md](api.md)
- **Examples**: [docs/examples.md](examples.md)
- **Installation**: [docs/INSTALL.md](INSTALL.md)
- **Testing**: [docs/TESTING.md](TESTING.md)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/dgaida/robot_workspace/issues)
- **Email**: daniel.gaida@th-koeln.de
- **Repository**: https://github.com/dgaida/robot_workspace

---

**Version**: 0.1.2
**Author**: Daniel Gaida
**Last Updated**: December 2024
