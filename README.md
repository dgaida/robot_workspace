# Robot Workspace

A comprehensive Python framework for robotic pick-and-place operations with vision-based object detection and manipulation capabilities.

## Badges

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality](https://github.com/dgaida/robot_workspace/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/robot_workspace/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/robot_workspace/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/robot_workspace/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/robot_workspace/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/robot_workspace/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Overview

This project provides a complete software stack for controlling robotic arms (currently supporting Niryo Ned2 and WidowX) with integrated computer vision for object detection, workspace management, and intelligent manipulation. The system combines real-time camera processing, Redis-based communication, and natural language interaction capabilities.

## Key Features

- **Multi-Robot Support**: Modular architecture supporting Niryo Ned2 and WidowX robotic arms
- **Workspace Management**: Flexible workspace definition and camera-to-world coordinate transformation
- **Simulation Support**: Compatible with both real robots and Gazebo simulation

## Architecture

### Core Components

```
robot_workspace/
├── objects/             # Object detection and representation
│   ├── object.py
│   ├── objects.py
│   └── pose_object.py
└── workspaces/          # Workspace definitions and management
    ├── workspace.py
    ├── workspaces.py
    └── niryo_workspace.py
```

## Installation

### Prerequisites

- Python 3.8+

## Quick Start

### Basic Usage

```python
from robot_workspace.environment import Environment
import threading
import time

# Initialize environment
env = Environment(
    el_api_key="your_elevenlabs_key",  # For TTS
    use_simulation=False,               # True for Gazebo
    robot_id="niryo",                   # or "widowx"
    verbose=True
)

# Start camera updates in background
def start_camera_updates(environment, visualize=False):
    def loop():
        for img in environment.update_camera_and_objects(visualize=visualize):
            pass
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t

camera_thread = start_camera_updates(env, visualize=True)

# Move to observation pose
env.robot_move2observation_pose(env.get_workspace_home_id())

# Wait for objects to be detected
time.sleep(2)

# Get detected objects
detected_objects = env.get_detected_objects()
print(f"Detected {len(detected_objects)} objects")

# Pick and place an object
robot = env.robot()
robot.pick_place_object(
    object_label="pencil",
    pick_location=[-0.1, 0.01],
    place_location=[0.1, 0.11],
    location="right next to"
)
```

### Advanced Features

#### Workspace Management

```python
# Get workspace information
workspace = env.get_workspace(0)
print(f"Workspace size: {workspace.width_m()}m x {workspace.height_m()}m")

# Transform coordinates
pose = workspace.transform_camera2world_coords(
    workspace_id="niryo_ws",
    u_rel=0.5,  # Center of image
    v_rel=0.5,
    yaw=0.0
)
```

## Configuration

### Adding New Workspaces

For Niryo robots, edit `niryo_workspace.py`:

```python
def _set_observation_pose(self) -> None:
    if self._id == "my_new_workspace":
        self._observation_pose = PoseObjectPNP(
            x=0.18, y=0.0, z=0.36,
            roll=0.0, pitch=math.pi/2, yaw=0.0
        )
```

## API Reference

### Object Class

- `label()` - Object label/name
- `xy_com()` - Center of mass coordinates
- `shape_m()` - Width and height in meters
- `gripper_rotation()` - Optimal gripper orientation
- `to_dict()` - Serialize to dictionary
- `from_dict(data, workspace)` - Deserialize from dictionary

### Objects Class

- `get_detected_object(coordinate, label)` - Find object at location
- `get_nearest_detected_object(coordinate, label)` - Find nearest object
- `get_largest_detected_object()` - Get largest object
- `get_detected_objects_sorted(ascending)` - Sort by size

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the existing architecture patterns
2. Documentation is updated
3. Type hints are included

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub.
