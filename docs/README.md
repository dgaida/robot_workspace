# Robot Workspace Architecture Documentation

This document describes the architectural design of the `robot_workspace` package, detailing component interactions and data flows.

## Overview

## System Architecture

![UML Class Diagramm](robot_workspace_uml.png)

## Core Components

### 4. Object Representation Layer

#### Object (`objects/object.py`)

Represents a detected object with full spatial information.

**Properties:**
- Label and workspace reference
- Pixel coordinates (bounding box)
- World coordinates (pose)
- Physical dimensions (meters)
- Segmentation mask (optional)
- Gripper rotation (for optimal pickup)

**Coordinate Systems:**
- Image coordinates: `(u, v)` pixels
- Relative coordinates: `(u_rel, v_rel)` normalized [0,1]
- World coordinates: `(x, y, z)` meters

**Serialization:**
```python
# Convert to JSON for Redis
obj_dict = obj.to_dict()
json_str = obj.to_json()

# Reconstruct from JSON
reconstructed = Object.from_dict(obj_dict, workspace)
```

#### Objects (`objects/objects.py`)

Collection class extending Python's `List`.

**Query Methods:**
- `get_detected_object(coordinate, label)` - Find by location
- `get_nearest_detected_object(coordinate)` - Nearest search
- `get_largest_detected_object()` - Size-based queries
- `get_detected_objects(location, coordinate, label)` - Spatial filtering

**Spatial Filters:**
```python
Location.LEFT_NEXT_TO   # y > coordinate[1]
Location.RIGHT_NEXT_TO  # y < coordinate[1]
Location.ABOVE          # x > coordinate[0]
Location.BELOW          # x < coordinate[0]
Location.CLOSE_TO       # distance <= 2cm
```

#### PoseObjectPNP (`objects/pose_object.py`)

File source: [Niryo Robotics](https://niryorobotics.github.io/pyniryo/v1.2.1/api/api.html#pyniryo.api.objects.PoseObject)

6-DOF pose representation.

**Components:**
- Position: `(x, y, z)` in meters
- Orientation: `(roll, pitch, yaw)` in radians

**Features:**
- Arithmetic operations (`+`, `-`)
- Transformation matrices (4×4 homogeneous)
- Quaternion conversion
- Approximate equality checks

### 5. Workspace Layer

#### Workspace (`workspaces/workspace.py`)

Abstract base class defining a robot workspace.

**Responsibilities:**
- Camera-to-world coordinate transformation
- Workspace boundary definition
- Visibility detection
- Observation pose management

**Key Concept - Coordinate Transformation:**
```
Image Coordinates (pixels)
    ↓ normalize
Relative Coordinates [0,1]
    ↓ transform_camera2world_coords()
World Coordinates (meters)
```

#### NiryoWorkspace (`workspaces/niryo_workspace.py`)

Niryo-specific workspace implementation.

**Features:**
- Uses Niryo's built-in vision system
- Supports multiple workspace definitions
- Automatic corner detection
- Predefined observation poses

**Workspace Corners:**
- `xy_ul_wc()` - Upper left
- `xy_ur_wc()` - Upper right
- `xy_ll_wc()` - Lower left
- `xy_lr_wc()` - Lower right
- `xy_center_wc()` - Center point

#### Workspaces (`workspaces/workspaces.py`)

Collection managing multiple workspaces.

**Features:**
- Workspace lookup by ID
- Visible workspace detection
- Home workspace management

## Coordinate Systems

### Three Coordinate Systems

1. **Image Coordinates (Pixels)**
   - Origin: Top-left corner
   - Units: Pixels
   - Range: `u ∈ [0, width]`, `v ∈ [0, height]`

2. **Relative Coordinates**
   - Origin: Top-left corner
   - Units: Normalized [0, 1]
   - Range: `u_rel, v_rel ∈ [0, 1]`
   - Used for workspace-independent calculations

3. **World Coordinates (Robot Base Frame)**
   - Origin: Robot base
   - Units: Meters
   - Niryo axes:
     - `x`: Forward (away from base)
     - `y`: Right (when facing robot)
     - `z`: Up

Camera Coordinate System:

```
  (u_min, v_min)
         ────────────
        │            │
  v_rel │            │
        │            │
        └──────────── (u_max, v_max)
            → u_rel

```

World Coordinate System:

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

### Transformation Chain

```
Image (u, v)
    ↓ divide by image dimensions
Relative (u_rel, v_rel)
    ↓ Workspace.transform_camera2world_coords()
    ↓ (uses Niryo's get_target_pose_from_rel())
World (x, y, z) + orientation (roll, pitch, yaw)
```

### Example Transformation

```python
# Object at pixel (320, 240) in 640x480 image
u, v = 320, 240
u_rel = 320 / 640 = 0.5  # Center horizontally
v_rel = 240 / 480 = 0.5  # Center vertically

# Transform to world coordinates
pose = workspace.transform_camera2world_coords(
    "niryo_ws", u_rel=0.5, v_rel=0.5, yaw=0.0
)
# Result: pose.x ≈ 0.25, pose.y ≈ 0.0, pose.z ≈ 0.01
```

## Configuration

### Workspace Configuration

Workspaces are defined in `niryo_workspace.py`:

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

## Error Handling

### Object Not Found

```python
obj = objects.get_detected_object([0.2, 0.0], label="nonexistent")
if obj is None:
    print("Object not found")
    # Handle missing object
```

## Extension Points

### Adding New Workspace

1. Add ID to `NiryoWorkspace._set_observation_pose()`
2. Define observation pose
3. No code changes needed elsewhere

### Custom Object Queries

```python
class MyObjects(Objects):
    def get_objects_in_region(self, x_min, x_max, y_min, y_max):
        return Objects(
            obj for obj in self
            if x_min <= obj.x_com() <= x_max
            and y_min <= obj.y_com() <= y_max
        )
```

## Summary

The `robot_workspace` architecture provides:

✅ **Modular Design** - Clear separation of concerns
✅ **Flexible Workspaces** - Multiple workspace support
✅ **Rich Object Representation** - Full spatial information
