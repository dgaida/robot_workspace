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

**Note**: More examples in [examples.md](docs/examples.md).

## Running the Demo

The package includes a comprehensive demonstration script that uses mocked components:

```bash
python main.py
```

This will demonstrate:
- Pose object creation and manipulation
- Workspace management (with mock environment)
- Object creation and properties
- Spatial queries and filtering
- Serialization and deserialization
- LLM-friendly formatting

**No robot hardware required for the demo!**

## API Reference

[API.md](docs/api.md)

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
        self._observation_pose = PoseObjectPNP(...)

    def _set_4corners_of_workspace(self):
        # Define workspace corners using transform method
        self._xy_ul_wc = self.transform_camera2world_coords(self._id, 0.0, 0.0)
        self._xy_lr_wc = self.transform_camera2world_coords(self._id, 1.0, 1.0)
        # ... set other corners
```

2. **Create a workspaces collection class**:

```python
from robot_workspace.workspaces.workspaces import Workspaces

class MyRobotWorkspaces(Workspaces):
    def __init__(self, environment, verbose: bool = False):
        super().__init__(verbose)
        workspace = MyRobotWorkspace("my_workspace_id", environment, verbose)
        self.append_workspace(workspace)
```

3. **Integrate with your robot's Environment**:

Your `Environment` class should provide:
- `use_simulation()` - Returns True if in simulation mode
- `get_robot_target_pose_from_rel(workspace_id, u_rel, v_rel, yaw)` - Transforms relative coordinates to world pose

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

[CONTRIBUTING.md](CONTRIBUTING.md)

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

## Acknowledgments

- Built for the Niryo Ned2 robotic platform
- Designed for integration with computer vision systems
- Supports both real robots and Gazebo simulation

---

**Author**: Daniel Gaida
**Email**: daniel.gaida@th-koeln.de
**Repository**: https://github.com/dgaida/robot_workspace
