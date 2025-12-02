# Robot Workspace

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/dgaida/robot_workspace/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/robot_workspace)
[![Code Quality](https://github.com/dgaida/robot_workspace/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/robot_workspace/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/robot_workspace/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/robot_workspace/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/robot_workspace/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/robot_workspace/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A comprehensive Python framework for robotic workspace management with vision-based object detection and coordinate transformations. This package provides the essential data structures and utilities for managing robot workspaces, detecting objects, and transforming coordinates between camera and world frames.

---

## 🎯 Overview

The `robot_workspace` package provides a complete framework for managing robotic workspaces, including:

- **🎯 Coordinate Transformations**: Seamlessly transform between camera and world coordinate frames
- **📦 Object Representation**: Rich object models with position, dimensions, segmentation masks, and orientation
- **🗺️ Workspace Management**: Define and manage multiple workspaces with different configurations
- **🔍 Spatial Queries**: Find objects by location, size, proximity, or custom criteria
- **💾 Serialization**: JSON-based serialization for data persistence and communication
- **🤖 Robot Support**: Native support for Niryo Ned2 and WidowX 250 6DOF robots (extensible to other platforms)
- **🧪 Mock Environment**: Full testing and demo support without requiring hardware

---

## ✨ Key Features

### Vision & Detection
- Integrate object detection with bounding boxes, segmentation masks, and physical properties
- Calculate center of mass and optimal gripper orientations
- Support for multi-object tracking and management

### Coordinate Systems
- Transform between relative image coordinates (0-1) and world coordinates (meters)
- Handle multiple workspace configurations with different camera poses
- Automatic workspace boundary detection

### Spatial Reasoning
- Query objects by spatial relationships (left/right/above/below/close to)
- Find nearest objects to specified coordinates
- Filter by size, label, or custom criteria

### LLM Integration
- Generate natural language descriptions of objects and scenes
- Structured output formats for AI agent integration
- Easy-to-parse object properties

### Quality Assurance
- >90% test coverage with comprehensive unit and integration tests
- Full type annotations for better IDE support
- Extensive documentation and examples

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Examples](#-examples)
- [Documentation](#-documentation)
- [Testing](#-testing)
- [Adding Robot Support](#-adding-robot-support)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

---

## 🏗️ Architecture

### Core Components

```
robot_workspace/
├── objects/                   # Object detection and representation
│   ├── object.py              # Single object with properties and methods
│   ├── objects.py             # Collection of objects with spatial queries
│   ├── object_api.py          # API interface for objects
│   └── pose_object.py         # 6-DOF pose representation (x, y, z, roll, pitch, yaw)
├── workspaces/                # Workspace definitions and management
│   ├── workspace.py           # Abstract workspace base class
│   ├── workspaces.py          # Collection of workspaces
│   ├── niryo_workspace.py     # Niryo Ned2 workspace implementation
│   ├── niryo_workspaces.py    # Niryo workspace collection
│   ├── widowx_workspace.py    # WidowX 250 6DOF implementation
│   └── widowx_workspaces.py   # WidowX workspace collection
└── common/                    # Utilities
    └── logger.py              # Logging decorators
```

### Architecture Diagram

![Architecture diagram](docs/architecture_diagram.png)

### Coordinate Systems

The package handles three coordinate systems:

1. **Image Coordinates (Pixels)**: Raw camera pixel coordinates
2. **Relative Coordinates (0-1)**: Normalized workspace-independent coordinates
3. **World Coordinates (Meters)**: Robot base frame coordinates

```
Image (u, v) → Relative (u_rel, v_rel) → World (x, y, z) + Orientation
```

For detailed information, see [Architecture Documentation](docs/README.md).

---

## 📦 Installation

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

### With Robot Support

```bash
# Niryo Ned2 support
pip install -e ".[niryo]"

# All features
pip install -e ".[all]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

This installs additional development tools:
- pytest and pytest-cov for testing
- black for code formatting
- ruff for linting
- mypy for type checking
- pre-commit hooks

---

## 🚀 Quick Start

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

### Managing Workspaces (Mock Demo)

```python
from unittest.mock import Mock
from robot_workspace.workspaces.niryo_workspaces import NiryoWorkspaces
from robot_workspace.objects.pose_object import PoseObjectPNP

# Create a mock environment (no hardware needed!)
def create_mock_environment():
    env = Mock()
    env.use_simulation.return_value = True

    def mock_transform(ws_id, u_rel, v_rel, yaw):
        x = 0.4 - u_rel * 0.3
        y = 0.15 - v_rel * 0.3
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    env.get_robot_target_pose_from_rel = mock_transform
    return env

# Initialize workspace collection
mock_env = create_mock_environment()
workspaces = NiryoWorkspaces(mock_env, verbose=False)

# Get workspace information
workspace = workspaces.get_home_workspace()
print(f"Workspace: {workspace.id()}")
print(f"Dimensions: {workspace.width_m():.3f}m x {workspace.height_m():.3f}m")

# Transform coordinates
world_pose = workspace.transform_camera2world_coords(
    workspace_id=workspace.id(),
    u_rel=0.5,  # Center of image
    v_rel=0.5,
    yaw=0.0
)
print(f"World coordinates: {world_pose}")
```

### Working with Objects

```python
from robot_workspace.objects.object import Object
from robot_workspace.objects.objects import Objects
from robot_workspace.objects.object_api import Location

# Create an object (requires a workspace)
obj = Object(
    label="pencil",
    u_min=100, v_min=100,
    u_max=200, v_max=200,
    mask_8u=None,
    workspace=workspace
)

# Access object properties
print(f"Position: {obj.coordinate()}")
print(f"Size: {obj.width_m():.3f}m x {obj.height_m():.3f}m")
print(f"Area: {obj.size_m2()*10000:.2f} cm²")

# Create a collection and perform spatial queries
objects = Objects([obj1, obj2, obj3])

# Find objects to the left of a coordinate
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

---

## 📚 Examples

### Running the Demo

The package includes a comprehensive demonstration script that uses mocked components:

```bash
python main.py
```

This demonstrates:
- Pose object creation and manipulation
- Workspace management (with mock environment)
- Object creation and properties
- Spatial queries and filtering
- Serialization and deserialization
- LLM-friendly formatting

**No robot hardware required for the demo!**

### More Examples

See [examples.md](docs/examples.md) for detailed usage examples including:
- Object detection workflows
- Multi-workspace management
- Serialization patterns
- Integration with robot controllers

---

## 📖 Documentation

### API Reference
- [API Documentation](docs/api.md) - Complete API reference
- [Architecture Documentation](docs/README.md) - System design and patterns
- [Examples](docs/examples.md) - Usage examples and recipes

### Key Classes

- **PoseObjectPNP**: 6-DOF pose with position and orientation
- **Object**: Detected object with physical properties
- **Objects**: Collection with spatial query capabilities
- **Workspace**: Abstract workspace base class
- **NiryoWorkspace**: Niryo Ned2 implementation
- **WidowXWorkspace**: WidowX 250 6DOF implementation

---

## 🧪 Testing

The package includes comprehensive tests with >90% coverage.

### Run All Tests

```bash
pytest
```

### Run with Coverage Report

```bash
pytest --cov=robot_workspace --cov-report=html --cov-report=term
```

### Run Specific Tests

```bash
# Unit tests only
pytest tests/objects/
pytest tests/workspaces/

# Integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Test Coverage

The test suite covers:
- **PoseObjectPNP**: Initialization, arithmetic, transformations, quaternions
- **Object**: Creation, serialization, properties, mask operations
- **Objects**: Collection operations, spatial queries, filtering
- **Workspace**: Initialization, transformations, visibility checks
- **Integration**: End-to-end workflows and multi-component interactions

See [tests/README.md](tests/README.md) for detailed testing documentation.

---

## 🔧 Adding Robot Support

To add support for a new robot platform:

### 1. Create a Workspace Class

```python
from robot_workspace.workspaces.workspace import Workspace
from robot_workspace.objects.pose_object import PoseObjectPNP

class MyRobotWorkspace(Workspace):
    def __init__(self, workspace_id: str, environment, verbose: bool = False):
        self._environment = environment
        super().__init__(workspace_id, verbose)

    def transform_camera2world_coords(self, workspace_id, u_rel, v_rel, yaw=0.0):
        # Implement coordinate transformation using your robot's API
        return self._environment.get_robot_target_pose_from_rel(
            workspace_id, u_rel, v_rel, yaw
        )

    def _set_observation_pose(self):
        # Define observation pose for your workspace
        self._observation_pose = PoseObjectPNP(
            x=0.3, y=0.0, z=0.25,
            roll=0.0, pitch=1.57, yaw=0.0
        )

    def _set_4corners_of_workspace(self):
        # Define workspace corners
        self._xy_ul_wc = self.transform_camera2world_coords(self._id, 0.0, 0.0)
        self._xy_lr_wc = self.transform_camera2world_coords(self._id, 1.0, 1.0)
        # ... set other corners
```

### 2. Create a Workspaces Collection

```python
from robot_workspace.workspaces.workspaces import Workspaces

class MyRobotWorkspaces(Workspaces):
    def __init__(self, environment, verbose: bool = False):
        super().__init__(verbose)
        workspace = MyRobotWorkspace("my_workspace_id", environment, verbose)
        self.append_workspace(workspace)
```

### 3. Integrate with Environment

Your `Environment` class should provide:
- `use_simulation()` - Returns True if in simulation mode
- `get_robot_target_pose_from_rel(workspace_id, u_rel, v_rel, yaw)` - Coordinate transformation

See [WidowXWorkspace](robot_workspace/workspaces/widowx_workspace.py) for a complete example.

---

## 💻 Development

### Code Quality Tools

This project uses:
- **Black** for code formatting (line length: 127)
- **Ruff** for fast Python linting
- **mypy** for type checking
- **pre-commit** hooks for automated checks

### Setup Pre-commit Hooks

```bash
pre-commit install
```

### Manual Code Quality Checks

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy robot_workspace --ignore-missing-imports
```

### Project Structure

```
robot_workspace/
├── .github/workflows/      # CI/CD workflows
├── docs/                   # Documentation
├── robot_workspace/        # Source code
├── tests/                  # Test suite
├── main.py                 # Demo script
├── pyproject.toml         # Project configuration
├── requirements.txt       # Dependencies
└── README.md              # This file
```

---

## 🤝 Contributing

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📝 Citation

If you use this package in your research, please cite:

```bibtex
@software{robot_workspace,
  author = {Gaida, Daniel},
  title = {Robot Workspace: A Framework for Robotic Workspace Management},
  year = {2025},
  url = {https://github.com/dgaida/robot_workspace}
}
```

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/dgaida/robot_workspace/issues)
- **Documentation**: See [docs/](docs/) directory
- **Examples**: Run `python main.py` for demonstrations
- **Email**: daniel.gaida@th-koeln.de

---

## 🙏 Acknowledgments

- Built for the Niryo Ned2 and WidowX 250 6DOF robotic platforms
- Designed for integration with computer vision systems
- Supports both real robots and Gazebo simulation
- Mock environment enables hardware-free development and testing

---

## 🗺️ Roadmap

- [ ] Additional robot platform support
- [ ] Enhanced multi-workspace coordination
  - Automatic object handoff between workspaces
  - Synchronized multi-workspace scanning
  - Cross-workspace object tracking and state management
  - Collision-free multi-arm coordination
  - Shared memory pools for collaborative tasks
  - Priority-based workspace arbitration
- [ ] Integration with popular ML frameworks
- [ ] ROS2 compatibility layer
- [ ] Web-based visualization tools

---

## 🔗 Related Projects

This package is part of a larger ecosystem for robotic manipulation and AI-driven control:

- **[robot_environment](https://github.com/dgaida/robot_environment)** - Complete robot control framework for pick-and-place operations with Niryo Ned2 and WidowX 250 6DOF robots
- **[vision_detect_segment](https://github.com/dgaida/vision_detect_segment)** - Real-time object detection and segmentation system with YOLO integration
- **[robot_mcp](https://github.com/dgaida/robot_mcp)** - Model Context Protocol (MCP) server enabling LLM-based natural language control of robotic systems

---

**Author**: Daniel Gaida
**Email**: daniel.gaida@th-koeln.de
**Repository**: https://github.com/dgaida/robot_workspace
**Version**: 0.1.0
