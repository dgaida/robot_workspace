# Robot Workspace

A Python framework for robotic workspace management with vision-based object detection and coordinate transformations. This package provides the core data structures and utilities for managing robot workspaces, detected objects, and coordinate transformations between camera and world frames.

## Badges

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality](https://github.com/dgaida/robot_workspace/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/robot_workspace/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/robot_workspace/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/robot_workspace/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/robot_workspace/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/robot_workspace/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Overview

The `robot_workspace` package provides a comprehensive framework for managing robotic workspaces, including:

- **Coordinate Transformations**: Transform between camera and world coordinate frames
- **Object Representation**: Rich object models with position, dimensions, segmentation, and orientation
- **Workspace Management**: Define and manage multiple workspaces with different configurations
- **Spatial Queries**: Find objects by location, size, or proximity
- **Serialization**: JSON-based serialization for data persistence and communication
- **Robot Support**: Currently supports Niryo Ned2 robots (extendable to other platforms)

## Key Features

- **Object Detection Integration**: Represent detected objects with bounding boxes, segmentation masks, and physical properties
- **Flexible Coordinate Systems**: Transform between relative image coordinates and world coordinates
- **Spatial Reasoning**: Query objects by spatial relationships (left/right/above/below/close to)
- **Size-based Queries**: Find largest, smallest, or sorted objects
- **LLM-Ready Formatting**: Generate natural language descriptions of objects and scenes
- **Comprehensive Testing**: >90% test coverage with unit and integration tests
- **Type Hints**: Full type annotations for better IDE support and code quality

## Architecture

### Core Components

```
robot_workspace/
├── objects/                 # Object detection and representation
│   ├── object.py           # Single object with properties and methods
│   ├── objects.py          # Collection of objects with spatial queries
│   ├── object_api.py       # API interface for objects
│   └── pose_object.py      # 6-DOF pose representation (x, y, z, roll, pitch, yaw)
├── workspaces/             # Workspace definitions and management
│   ├── workspace.py        # Abstract workspace base class
│   ├── workspaces.py       # Collection of workspaces
│   ├── niryo_workspace.py  # Niryo Ned2 workspace implementation
│   └── niryo_workspaces.py # Niryo workspace collection
└── common/                 # Utilities
    └── logger.py           # Logging decorators
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/dgaida/robot_workspace.git
cd robot_workspace

# Install in development mode
pip install -e .
```

### With Niryo Robot Support

```bash
pip install -e ".[niryo]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

This installs additional tools for development:
- pytest and pytest-cov for testing
- black for code formatting
- ruff for linting
- mypy for type checking
- pre-commit hooks

## Quick Start

### Working with Pose Objects

```python
from robot_workspace.objects.pose_object import PoseObjectPNP

# Create a 6-DOF pose
pose = PoseObjectPNP(
    x=0.2, y=0.1, z=0.3,
    roll=0.0, pitch=1.57, yaw=0.0
)

print(pose)  # Display position and orientation

# Pose arithmetic
offset = PoseObjectPNP(x=0.05, y=0.02, z=0.0)
new_pose = pose + offset

# Convert to different representations
pose_list = pose.to_list()
quaternion = pose.quaternion
transform_matrix = pose.to_transformation_matrix()
```

### Managing Workspaces

```python
from robot_workspace.workspaces.niryo_workspaces import NiryoWorkspaces

# Create workspace collection (simulation mode)
workspaces = NiryoWorkspaces(use_simulation=True, verbose=False)

# Get the home workspace
workspace = workspaces.get_home_workspace()

print(f"Workspace: {workspace.id()}")
print(f"Dimensions: {workspace.width_m():.3f}m x {workspace.height_m():.3f}m")

# Get observation pose (where robot should be to view workspace)
obs_pose = workspace.observation_pose()
print(f"Observation pose: {obs_pose}")

# Transform camera coordinates to world coordinates
world_pose = workspace.transform_camera2world_coords(
    workspace_id=workspace.id(),
    u_rel=0.5,  # Center of image (0-1 range)
    v_rel=0.5,
    yaw=0.0
)
```

### Working with Objects

```python
from robot_workspace.objects.object import Object
from robot_workspace.objects.objects import Objects
from robot_workspace.objects.object_api import Location

# Create an object (requires a workspace)
obj = Object(
    label="pencil",
    u_min=100, v_min=100,    # Bounding box in pixels
    u_max=200, v_max=200,
    mask_8u=None,             # Optional segmentation mask
    workspace=workspace
)

# Access object properties
print(f"Label: {obj.label()}")
print(f"Position (x, y): {obj.coordinate()}")
print(f"Dimensions: {obj.width_m():.3f}m x {obj.height_m():.3f}m")
print(f"Size: {obj.size_m2()*10000:.2f} cm²")
print(f"Gripper rotation: {obj.gripper_rotation():.3f} rad")

# Create a collection of objects
objects = Objects([obj1, obj2, obj3])

# Spatial queries
left_objects = objects.get_detected_objects(
    location=Location.LEFT_NEXT_TO,
    coordinate=[0.2, 0.0]
)

# Find nearest object
nearest, distance = objects.get_nearest_detected_object([0.2, 0.1])

# Size-based queries
largest, size = objects.get_largest_detected_object()
sorted_objs = objects.get_detected_objects_sorted(ascending=True)
```

### Object Serialization

```python
# Serialize to dictionary
obj_dict = obj.to_dict()

# Serialize to JSON
json_str = obj.to_json()

# Deserialize from dictionary
reconstructed = Object.from_dict(obj_dict, workspace)

# Collection serialization
dict_list = Objects.objects_to_dict_list(objects)
reconstructed_objects = Objects.dict_list_to_objects(dict_list, workspace)
```

### LLM-Friendly Formatting

```python
# Generate natural language descriptions
llm_description = obj.as_string_for_llm()
# Output: "- 'pencil' at world coordinates [0.20, 0.10] with a width of
#          0.05 meters, a height of 0.08 meters and a size of 40.00
#          square centimeters."

chat_description = obj.as_string_for_chat_window()
# Output: "Detected a new object: pencil at world coordinate (0.20, 0.10)
#          with orientation 0.5 rad and size 0.05 m x 0.08 m."
```

## Running the Demo

The package includes a comprehensive demonstration script:

```bash
python main.py
```

This will demonstrate:
- Pose object creation and manipulation
- Workspace management
- Object creation and properties
- Spatial queries and filtering
- Serialization and deserialization
- LLM-friendly formatting

## API Reference

### PoseObjectPNP

Represents a 6-DOF pose (position + orientation).

**Key Methods:**
- `to_list()` - Convert to [x, y, z, roll, pitch, yaw]
- `to_transformation_matrix()` - Get 4x4 transformation matrix
- `quaternion` - Get quaternion representation [qx, qy, qz, qw]
- `copy_with_offsets()` - Create new pose with offsets
- `approx_eq()` - Check approximate equality

### Object

Represents a detected object in the workspace.

**Key Properties:**
- `label()` - Object label/name
- `coordinate()` - [x, y] world coordinates
- `xy_com()` - Center of mass coordinates
- `shape_m()` - (width, height) in meters
- `size_m2()` - Area in square meters
- `gripper_rotation()` - Optimal gripper orientation

**Key Methods:**
- `to_dict()` / `to_json()` - Serialize
- `from_dict()` / `from_json()` - Deserialize (class methods)
- `as_string_for_llm()` - LLM-friendly description
- `as_string_for_chat_window()` - Chat-friendly description

### Objects

Collection class for managing multiple objects.

**Key Methods:**
- `get_detected_object(coordinate, label)` - Find specific object
- `get_detected_objects(location, coordinate, label)` - Filter objects
- `get_nearest_detected_object(coordinate, label)` - Find nearest
- `get_largest_detected_object()` - Get largest by size
- `get_detected_objects_sorted(ascending)` - Sort by size

**Location Filters:**
- `Location.LEFT_NEXT_TO` / `Location.RIGHT_NEXT_TO`
- `Location.ABOVE` / `Location.BELOW`
- `Location.CLOSE_TO` (within 2cm)

### Workspace

Abstract base class for robot workspaces.

**Key Methods:**
- `transform_camera2world_coords(workspace_id, u_rel, v_rel, yaw)` - Coordinate transformation
- `is_visible(camera_pose)` - Check if workspace is visible
- `observation_pose()` - Get optimal viewing pose

**Key Properties:**
- `id()` - Workspace identifier
- `width_m()` / `height_m()` - Dimensions in meters
- `xy_ul_wc()`, `xy_lr_wc()` - Corner coordinates

## Testing

The package includes comprehensive tests with >90% coverage.

### Run All Tests

```bash
pytest
```

### Run with Coverage Report

```bash
pytest --cov=robot_workspace --cov-report=html --cov-report=term
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/objects/
pytest tests/workspaces/

# Integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

See `tests/README.md` for detailed testing documentation.

## Development

### Code Quality

This project uses:
- **Black** for code formatting (line length: 127)
- **Ruff** for fast Python linting
- **mypy** for type checking
- **pre-commit** hooks for automated checks

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Code Style

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy robot_workspace --ignore-missing-imports
```

## Adding Support for New Robots

To add support for a new robot platform:

1. **Create a new workspace class** inheriting from `Workspace`:

```python
from robot_workspace.workspaces.workspace import Workspace

class MyRobotWorkspace(Workspace):
    def transform_camera2world_coords(self, workspace_id, u_rel, v_rel, yaw=0.0):
        # Implement coordinate transformation
        pass

    def _set_observation_pose(self):
        # Define observation pose
        self._observation_pose = PoseObjectPNP(...)

    def _set_4corners_of_workspace(self):
        # Define workspace corners
        pass
```

2. **Create a workspaces collection class**:

```python
from robot_workspace.workspaces.workspaces import Workspaces

class MyRobotWorkspaces(Workspaces):
    def __init__(self, use_simulation=False, verbose=False):
        super().__init__(verbose)
        workspace = MyRobotWorkspace("my_workspace_id", verbose)
        self.append_workspace(workspace)
```

## Project Structure

```
robot_workspace/
├── robot_workspace/         # Source code
│   ├── objects/            # Object representation
│   ├── workspaces/         # Workspace management
│   └── common/             # Utilities
├── tests/                   # Test suite
│   ├── objects/            # Object tests
│   ├── workspaces/         # Workspace tests
│   └── conftest.py         # Test configuration
├── .github/workflows/       # CI/CD workflows
├── main.py                  # Demo script
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the existing style (Black, Ruff)
2. All tests pass: `pytest`
3. New features include tests
4. Documentation is updated
5. Type hints are included

### Development Workflow

```bash
# Create a feature branch
git checkout -b feature/my-feature

# Make changes and test
pytest

# Format and lint
black .
ruff check . --fix

# Commit with clear messages
git commit -m "Add feature: description"

# Push and create pull request
git push origin feature/my-feature
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{robot_workspace,
  author = {Gaida, Daniel},
  title = {Robot Workspace: A Framework for Robotic Workspace Management},
  year = {2025},
  url = {https://github.com/dgaida/robot_workspace}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/dgaida/robot_workspace/issues)
- **Documentation**: See `tests/README.md` for testing guide
- **Examples**: Run `python main.py` for demonstrations

## Changelog

### v0.1.0 (2025)
- Initial release
- Support for Niryo Ned2 robots
- Object detection and representation
- Workspace management
- Coordinate transformations
- Serialization support
- Comprehensive test suite

## Acknowledgments

- Built for the Niryo Ned2 robotic platform
- Designed for integration with computer vision systems
- Supports both real robots and Gazebo simulation

---

**Author**: Daniel Gaida
**Email**: daniel.gaida@th-koeln.de
**Repository**: https://github.com/dgaida/robot_workspace
