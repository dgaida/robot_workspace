# Robot Workspace Architecture Documentation

Comprehensive documentation for the `robot_workspace` package architecture, design patterns, and implementation details.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Coordinate Systems](#coordinate-systems)
5. [Design Patterns](#design-patterns)
6. [Extension Guide](#extension-guide)

## Overview

The `robot_workspace` package provides a framework for managing robotic workspaces with vision-based object detection. It separates concerns between:

- **Object Representation** - Detected objects with physical properties
- **Workspace Management** - Robot workspace definitions and coordinate transforms
- **Spatial Reasoning** - Queries and relationships between objects

### Key Features

- 6-DOF pose representation (position + orientation)
- Camera-to-world coordinate transformations
- Object detection with segmentation masks
- Spatial queries (left/right/above/below/near)
- JSON serialization for inter-process communication
- Multi-workspace support
- Extensible for different robot platforms

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                     │
│  (LLM-based control, Pick-and-place tasks, etc.)        │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Robot Workspace Package                    │
│                                                         │
│   ┌──────────────────┐      ┌──────────────────┐        │
│   │ Object Layer     │      │ Workspace Layer  │        │
│   │                  │      │                  │        │
│   │ - PoseObjectPNP  │◄────►│ - Workspace      │        │
│   │ - Object         │      │ - Workspaces     │        │
│   │ - Objects        │      │ - NiryoWorkspace │        │
│   │ - Location       │      │ - WidowXWorkspace│        │
│   └──────────────────┘      └──────────────────┘        │
│                                                         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Robot Control Layer                        │
│  (PyNiryo, ROS, Robot SDK, etc.)                        │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│            Robot Hardware / Simulation                  │
│  (Niryo Ned2, WidowX 250, Gazebo, etc.)                 │
└─────────────────────────────────────────────────────────┘
```

### UML Class Diagram

![UML Class Diagram](robot_workspace_uml.png)

See [robot_workspace_uml.tex](tex/robot_workspace_uml.tex) for the LaTeX source.

### Component Interaction Diagram

![Architecture Diagram](architecture_diagram.png)

See [architecture_diagram.tex](tex/architecture_diagram.tex) for the LaTeX source.

## Core Components

### 1. Pose Representation (`PoseObjectPNP`)

Represents a 6-DOF pose in 3D space.

**Key Features:**
- Position: `(x, y, z)` in meters
- Orientation: `(roll, pitch, yaw)` in radians
- Arithmetic operations (`+`, `-`)
- Transformation matrices (4×4 homogeneous)
- Quaternion conversion
- Approximate equality checks

**Usage:**
```python
from robot_workspace import PoseObjectPNP

# Create pose
pose = PoseObjectPNP(x=0.2, y=0.1, z=0.3,
                      roll=0.0, pitch=1.57, yaw=0.0)

# Arithmetic
offset = PoseObjectPNP(x=0.05, y=0.02, z=0.0)
new_pose = pose + offset

# Transformations
matrix = pose.to_transformation_matrix()
quaternion = pose.quaternion
```

### 2. Object Representation (`Object`)

Represents a detected object with full spatial information.

**Properties:**
- Label (e.g., "pencil", "cube")
- Bounding box (pixel coordinates)
- World coordinates (meters)
- Physical dimensions (width, height, area)
- Segmentation mask (optional)
- Gripper rotation (for optimal pickup)

**Coordinate Systems:**
- **Image coordinates**: `(u, v)` in pixels
- **Relative coordinates**: `(u_rel, v_rel)` normalized [0, 1]
- **World coordinates**: `(x, y, z)` in meters

**Usage:**
```python
from robot_workspace import Object

# Create object (requires workspace)
obj = Object(
    label="pencil",
    u_min=100, v_min=100,
    u_max=200, v_max=200,
    mask_8u=segmentation_mask,  # Optional
    workspace=workspace
)

# Access properties
position = obj.coordinate()  # [x, y] world coords
dimensions = obj.shape_m()   # (width, height) in meters
size = obj.size_m2()         # Area in m²
rotation = obj.gripper_rotation()  # Optimal pickup angle
```

### 3. Object Collection (`Objects`)

Collection class extending Python's `List` with spatial query methods.

**Query Methods:**
- `get_detected_object(coordinate, label)` - Find by location and label
- `get_nearest_detected_object(coordinate)` - Nearest neighbor search
- `get_largest_detected_object()` - Size-based queries
- `get_detected_objects(location, coordinate, label)` - Spatial filtering

**Spatial Filters:**
- `Location.LEFT_NEXT_TO` - Objects with y > coordinate[1]
- `Location.RIGHT_NEXT_TO` - Objects with y < coordinate[1]
- `Location.ABOVE` - Objects with x > coordinate[0]
- `Location.BELOW` - Objects with x < coordinate[0]
- `Location.CLOSE_TO` - Objects within 2cm radius

**Usage:**
```python
from robot_workspace import Objects, Location

objects = Objects([obj1, obj2, obj3])

# Spatial queries
left_objects = objects.get_detected_objects(
    location=Location.LEFT_NEXT_TO,
    coordinate=[0.2, 0.0]
)

# Find nearest
nearest, distance = objects.get_nearest_detected_object([0.2, 0.1])

# Size-based
largest, size = objects.get_largest_detected_object()
sorted_objs = objects.get_detected_objects_sorted(ascending=True)
```

### 4. Workspace Management (`Workspace`)

Abstract base class defining a robot workspace.

**Responsibilities:**
- Camera-to-world coordinate transformation
- Workspace boundary definition
- Visibility detection
- Observation pose management

**Key Methods:**
```python
# Transform relative image coords to world coords
pose = workspace.transform_camera2world_coords(
    workspace_id="niryo_ws",
    u_rel=0.5,  # Center of image [0-1]
    v_rel=0.5,
    yaw=0.0     # Object orientation
)

# Get workspace properties
width = workspace.width_m()
height = workspace.height_m()
center = workspace.xy_center_wc()

# Check visibility
is_visible = workspace.is_visible(camera_pose)
```

**Concrete Implementations:**
- `NiryoWorkspace` - For Niryo Ned2 robot
- `WidowXWorkspace` - For WidowX 250 6DOF robot

### 5. Workspace Collections (`Workspaces`)

Manages multiple workspaces.

**Features:**
- Workspace lookup by ID
- Visible workspace detection
- Home workspace management
- Multi-workspace support

**Usage:**
```python
from robot_workspace import NiryoWorkspaces

workspaces = NiryoWorkspaces(environment, verbose=False)

# Access workspaces
home_ws = workspaces.get_home_workspace()
ws_by_id = workspaces.get_workspace_by_id("niryo_ws")

# Get visible workspace
visible_ws = workspaces.get_visible_workspace(camera_pose)
```

## Coordinate Systems

### Three Coordinate Systems

The package uses three distinct coordinate systems:

#### 1. Image Coordinates (Pixels)

- **Origin**: Top-left corner of image
- **Units**: Pixels
- **Range**: `u ∈ [0, width]`, `v ∈ [0, height]`
- **Usage**: Raw camera image processing

```
(0, 0) ────────────► u (width)
  │
  │
  │
  ▼
  v (height)
```

#### 2. Relative Coordinates (Normalized)

- **Origin**: Top-left corner of workspace
- **Units**: Normalized [0, 1]
- **Range**: `u_rel, v_rel ∈ [0, 1]`
- **Usage**: Workspace-independent calculations

```
(0, 0) ────────────► u_rel (1.0)
  │
  │  Workspace region
  │  (normalized)
  ▼
v_rel (1.0)
```

**Conversion:**
```python
u_rel = u_pixels / image_width
v_rel = v_pixels / image_height
```

#### 3. World Coordinates (Robot Base Frame)

- **Origin**: Robot base
- **Units**: Meters
- **Axes** (Niryo Ned2):
  - `x`: Forward (away from base)
  - `y`: Right (when facing robot)
  - `z`: Up

```
        Y (left)
        ↑
        │
0.087 ──┼──────────── Upper workspace boundary
        │
    0 ──┼────────────  Center line (Y=0)
        │
-0.087 ─┼──────────── Lower workspace boundary
        │
        └────────────→ X (forward)
      0.163        0.337
     (closer)     (farther)
```

#### 4. Camera Coordinate System:

```
  (u_min, v_min)
         ────────────
        │            │
  v_rel │            │
        │            │
        └──────────── (u_max, v_max)
            → u_rel

```

### Coordinate Transformation Pipeline

```
Image Coordinates (u, v)
         ↓
    [Normalize by image dimensions]
         ↓
Relative Coordinates (u_rel, v_rel)
         ↓
    [Workspace.transform_camera2world_coords()]
    [Uses robot-specific calibration]
         ↓
World Coordinates (x, y, z, roll, pitch, yaw)
```

**Example:**
```python
# Object at pixel (320, 240) in 640×480 image
u, v = 320, 240
u_rel = 320 / 640  # 0.5 (center horizontally)
v_rel = 240 / 480  # 0.5 (center vertically)

# Transform to world coordinates
pose = workspace.transform_camera2world_coords(
    "niryo_ws", u_rel=0.5, v_rel=0.5, yaw=0.0
)
# Result: pose.x ≈ 0.25, pose.y ≈ 0.0, pose.z ≈ 0.01
```

### Coordinate System Notes

**Niryo Ned2:**
- Width goes along y-axis
- Height goes along x-axis
- Gripper-mounted camera

**WidowX 250:**
- Uses third-person camera view
- Different transformation calibration
- May require manual corner detection

## Design Patterns

### 1. Abstract Factory Pattern

**Workspace Creation:**
```python
class Workspace(ABC):
    @abstractmethod
    def transform_camera2world_coords(...):
        pass

class NiryoWorkspace(Workspace):
    def transform_camera2world_coords(...):
        return self._environment.get_robot_target_pose_from_rel(...)

class WidowXWorkspace(Workspace):
    def transform_camera2world_coords(...):
        # Custom implementation for WidowX
        pass
```

### 2. Collection Pattern

**Objects as Enhanced List:**
```python
class Objects(List):
    def get_largest_detected_object(self):
        return max(self, key=lambda obj: obj.size_m2())

    def get_detected_objects(self, location, coordinate):
        # Spatial filtering
        pass
```

### 3. Serialization Pattern

**JSON Serialization:**
```python
class Object:
    def to_dict(self) -> Dict:
        return {
            "label": self._label,
            "position": {...},
            "dimensions": {...}
        }

    @classmethod
    def from_dict(cls, data: Dict, workspace):
        return cls(...)
```

### 4. Coordinate Transformation Strategy

**Robot-Specific Transformations:**
```python
# Strategy interface
def transform_camera2world_coords(workspace_id, u_rel, v_rel, yaw):
    pass

# Niryo strategy (uses PyNiryo API)
# WidowX strategy (uses custom calibration)
```

## Extension Guide

### Adding Support for a New Robot

#### Step 1: Create Workspace Class

```python
from robot_workspace.workspaces import Workspace

class MyRobotWorkspace(Workspace):
    def __init__(self, workspace_id, environment, verbose=False):
        self._environment = environment
        super().__init__(workspace_id, verbose)

    def transform_camera2world_coords(self, workspace_id,
                                      u_rel, v_rel, yaw=0.0):
        # Implement transformation using your robot's API
        x, y = self._calculate_world_position(u_rel, v_rel)
        z = 0.05  # Height above workspace
        roll, pitch = 0.0, 1.57  # Downward gripper
        return PoseObjectPNP(x, y, z, roll, pitch, yaw)

    def _set_observation_pose(self):
        # Define where camera should be to view workspace
        self._observation_pose = PoseObjectPNP(...)

    def _set_4corners_of_workspace(self):
        # Define workspace boundaries
        self._xy_ul_wc = self.transform_camera2world_coords(
            self._id, 0.0, 0.0
        )
        # ... set other corners
```

#### Step 2: Create Workspaces Collection

```python
from robot_workspace.workspaces import Workspaces

class MyRobotWorkspaces(Workspaces):
    def __init__(self, environment, verbose=False):
        super().__init__(verbose)

        # Add your workspaces
        workspace = MyRobotWorkspace("main_workspace",
                                    environment, verbose)
        self.append_workspace(workspace)
```

#### Step 3: Integrate with Robot API

Your `Environment` class should provide:

```python
class MyRobotEnvironment:
    def use_simulation(self):
        return False  # or True if in sim

    def get_robot_target_pose_from_rel(self, ws_id, u_rel, v_rel, yaw):
        # Transform relative coords to world pose
        # This is robot-specific
        return PoseObjectPNP(...)
```

### Custom Object Queries

Extend the `Objects` class:

```python
from robot_workspace import Objects

class MyObjects(Objects):
    def get_objects_in_region(self, x_min, x_max, y_min, y_max):
        return Objects(
            obj for obj in self
            if x_min <= obj.x_com() <= x_max
            and y_min <= obj.y_com() <= y_max
        )

    def get_objects_by_color(self, color):
        # Custom filtering
        pass
```

### Custom Spatial Locations

Add new location types:

```python
from robot_workspace import Location

# Extend the enum (in your code)
class ExtendedLocation(Location):
    DIAGONAL_TO = "diagonal to"
    SURROUNDING = "surrounding"
```

## Configuration

### Workspace Configuration

Workspaces are defined in the workspace implementation:

```python
def _set_observation_pose(self):
    if self._id == "niryo_ws":
        self._observation_pose = PoseObjectPNP(
            x=0.173, y=-0.002, z=0.277,
            roll=-3.042, pitch=1.327, yaw=-3.027
        )
    elif self._id == "gazebo_1":
        self._observation_pose = PoseObjectPNP(
            x=0.18, y=0, z=0.36,
            roll=2.4, pitch=π/2, yaw=2.4
        )
```

### Multi-Workspace Setup

```python
class NiryoWorkspaces(Workspaces):
    def __init__(self, environment, verbose=False):
        super().__init__(verbose)

        workspace_ids = ["niryo_ws", "niryo_ws_right"]

        for ws_id in workspace_ids:
            workspace = NiryoWorkspace(ws_id, environment, verbose)
            self.append_workspace(workspace)
```

## Error Handling

### Common Patterns

```python
# Object not found
obj = objects.get_detected_object([0.2, 0.0], label="cube")
if obj is None:
    print("Object not found")
    # Handle missing object

# Workspace visibility
visible_ws = workspaces.get_visible_workspace(camera_pose)
if visible_ws is None:
    print("No workspace visible")
    # Move camera or handle error

# Serialization
try:
    reconstructed = Object.from_dict(obj_dict, workspace)
    if reconstructed is None:
        print("Failed to reconstruct object")
except Exception as e:
    print(f"Deserialization error: {e}")
```

## Performance Considerations

### Coordinate Transformations

- Transformations are computed on-demand
- Cache transformed coordinates if used repeatedly
- Batch transformations when possible

### Object Queries

- Spatial queries are O(n) where n = number of objects
- Use specific filters to reduce search space
- Consider spatial indexing for large collections

### Serialization

- JSON serialization is relatively fast
- Segmentation masks increase serialization size
- Consider compressing masks for network transfer

## Testing

See [tests/README.md](../tests/README.md) for comprehensive testing guide.

**Quick test:**
```bash
pytest                    # Run all tests
pytest tests/objects/     # Test objects only
pytest tests/workspaces/  # Test workspaces only
pytest --cov              # With coverage
```

## Additional Resources

- [Installation Guide](INSTALL.md)
- [API Reference](api.md)
- [Examples](examples.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Test Documentation](../tests/README.md)

---

**Author**: Daniel Gaida
**Email**: daniel.gaida@th-koeln.de
**Repository**: https://github.com/dgaida/robot_workspace
