# Robot Workspace - Examples

Comprehensive examples demonstrating common use cases and workflows.

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Workspace Management](#workspace-management)
3. [Object Detection](#object-detection)
4. [Spatial Queries](#spatial-queries)
5. [Serialization](#serialization)
6. [Advanced Examples](#advanced-examples)
7. [Integration Examples](#integration-examples)

---

## Basic Examples

### Creating a Pose

```python
from robot_workspace import PoseObjectPNP

# Create a 6-DOF pose
pose = PoseObjectPNP(
    x=0.2,      # 20 cm forward
    y=0.1,      # 10 cm to the right
    z=0.3,      # 30 cm up
    roll=0.0,   # No roll
    pitch=1.57, # 90 degrees (gripper pointing down)
    yaw=0.0     # No yaw rotation
)

print(pose)
# Output:
# x = 0.2000, y = 0.1000, z = 0.3000
# roll = 0.000, pitch = 1.570, yaw = 0.000
```

### Pose Arithmetic

```python
from robot_workspace import PoseObjectPNP

# Initial pose
pose1 = PoseObjectPNP(x=0.2, y=0.1, z=0.3, roll=0.0, pitch=1.57, yaw=0.0)

# Offset
offset = PoseObjectPNP(x=0.05, y=0.02, z=0.0, roll=0.0, pitch=0.0, yaw=0.5)

# Add offset
new_pose = pose1 + offset
print(f"New position: ({new_pose.x}, {new_pose.y}, {new_pose.z})")
# Output: New position: (0.25, 0.12, 0.3)

# Subtract poses
difference = new_pose - pose1
print(f"Difference: ({difference.x}, {difference.y}, {difference.z})")
# Output: Difference: (0.05, 0.02, 0.0)
```

### Pose Transformations

```python
# Convert to list
pose_list = pose.to_list()
print(pose_list)
# [0.2, 0.1, 0.3, 0.0, 1.57, 0.0]

# Get XY coordinates only
xy = pose.xy_coordinate()
print(xy)
# [0.2, 0.1]

# Convert to transformation matrix
matrix = pose.to_transformation_matrix()
print(matrix.shape)
# (4, 4)

# Get quaternion representation
quaternion = pose.quaternion
print(quaternion)
# [qx, qy, qz, qw]
```

---

## Workspace Management

### Creating a Workspace with Mock Environment

```python
from unittest.mock import Mock
from robot_workspace import NiryoWorkspaces, PoseObjectPNP

# Create a mock environment (for demo/testing without real robot)
def create_mock_environment(use_simulation=True):
    env = Mock()
    env.use_simulation.return_value = use_simulation
    env.verbose.return_value = False

    # Mock coordinate transformation
    def mock_get_target_pose(ws_id, u_rel, v_rel, yaw):
        x = 0.4 - u_rel * 0.3    # x: 0.4 to 0.1
        y = 0.15 - v_rel * 0.3   # y: 0.15 to -0.15
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    env.get_robot_target_pose_from_rel = mock_get_target_pose
    return env

# Create workspace collection
mock_env = create_mock_environment(use_simulation=True)
workspaces = NiryoWorkspaces(mock_env, verbose=False)

# Get home workspace
workspace = workspaces.get_home_workspace()
print(f"Workspace ID: {workspace.id()}")
print(f"Dimensions: {workspace.width_m():.3f}m × {workspace.height_m():.3f}m")
```

### Coordinate Transformation

```python
# Transform relative image coordinates to world coordinates
world_pose = workspace.transform_camera2world_coords(
    workspace_id=workspace.id(),
    u_rel=0.5,  # Center of image (horizontal)
    v_rel=0.5,  # Center of image (vertical)
    yaw=0.0     # No rotation
)

print(f"World coordinates: ({world_pose.x:.3f}, {world_pose.y:.3f}, {world_pose.z:.3f})")
# Output: World coordinates: (0.250, 0.000, 0.050)
```

### Multiple Workspaces

```python
from robot_workspace import NiryoWorkspaces

# Create workspaces (automatically loads configured workspaces)
workspaces = NiryoWorkspaces(environment, verbose=False)

print(f"Number of workspaces: {len(workspaces)}")
print(f"Workspace IDs: {workspaces.get_workspace_ids()}")

# Access specific workspaces
left_ws = workspaces.get_workspace_left()
right_ws = workspaces.get_workspace_right()

# Get observation pose for each
left_obs = left_ws.observation_pose()
right_obs = right_ws.observation_pose()

print(f"Left workspace observation: {left_obs}")
print(f"Right workspace observation: {right_obs}")
```

### Checking Workspace Visibility

```python
# Get current camera pose (from robot)
current_pose = workspace.observation_pose()

# Check if workspace is visible
is_visible = workspace.is_visible(current_pose)
print(f"Workspace visible: {is_visible}")

# Find visible workspace from current pose
visible_ws = workspaces.get_visible_workspace(current_pose)
if visible_ws:
    print(f"Visible workspace: {visible_ws.id()}")
else:
    print("No workspace visible")
```

---

## Object Detection

### Creating Objects

```python
from robot_workspace import Object
import numpy as np

# Create object without segmentation mask
obj = Object(
    label="pencil",
    u_min=100, v_min=100,    # Bounding box top-left
    u_max=200, v_max=200,    # Bounding box bottom-right
    mask_8u=None,             # No segmentation mask
    workspace=workspace
)

# Create object with segmentation mask
mask = np.zeros((640, 480), dtype=np.uint8)
mask[100:200, 100:200] = 255  # White square

obj_with_mask = Object(
    label="cube",
    u_min=100, v_min=100,
    u_max=200, v_max=200,
    mask_8u=mask,
    workspace=workspace
)
```

### Accessing Object Properties

```python
# Label
print(f"Label: {obj.label()}")

# Position
print(f"Position (x, y): {obj.coordinate()}")
print(f"Center of mass: ({obj.x_com():.3f}, {obj.y_com():.3f})")
print(f"Center: ({obj.x_center():.3f}, {obj.y_center():.3f})")

# Dimensions
print(f"Width: {obj.width_m():.3f}m")
print(f"Height: {obj.height_m():.3f}m")
print(f"Size: {obj.size_m2() * 10000:.2f} cm²")

# Orientation
print(f"Gripper rotation: {obj.gripper_rotation():.3f} rad")
print(f"Gripper rotation: {np.degrees(obj.gripper_rotation()):.1f}°")

# Full poses
print(f"Pose (center): {obj.pose_center()}")
print(f"Pose (COM): {obj.pose_com()}")
```

### Updating Object Position

```python
# Update position after robot moves object
new_position = [0.3, 0.05]
obj.set_position(new_position)

# Or update full pose (includes rotation)
new_pose = obj.pose_com().copy_with_offsets(
    x_offset=0.1,
    y_offset=0.05,
    yaw_offset=0.5  # Rotate 0.5 radians
)
obj.set_pose_com(new_pose)

print(f"New position: {obj.coordinate()}")
```

---

## Spatial Queries

### Creating an Object Collection

```python
from robot_workspace import Objects

# Create multiple objects
obj1 = Object("pencil", 100, 100, 180, 140, None, workspace)
obj2 = Object("pen", 280, 200, 360, 260, None, workspace)
obj3 = Object("eraser", 450, 350, 510, 410, None, workspace)

# Create collection
objects = Objects([obj1, obj2, obj3])

print(f"Total objects: {len(objects)}")
print(f"Objects: {objects.get_detected_objects_as_comma_separated_string()}")
```

### Finding Objects by Location

```python
from robot_workspace import Location

# Reference point
reference = [0.25, 0.0]

# Find objects to the left
left_objects = objects.get_detected_objects(
    location=Location.LEFT_NEXT_TO,
    coordinate=reference
)
print(f"Objects to the left: {len(left_objects)}")

# Find objects to the right
right_objects = objects.get_detected_objects(
    location=Location.RIGHT_NEXT_TO,
    coordinate=reference
)
print(f"Objects to the right: {len(right_objects)}")

# Find objects above
above_objects = objects.get_detected_objects(
    location=Location.ABOVE,
    coordinate=reference
)
print(f"Objects above: {len(above_objects)}")

# Find objects below
below_objects = objects.get_detected_objects(
    location=Location.BELOW,
    coordinate=reference
)
print(f"Objects below: {len(below_objects)}")

# Find objects close to point (within 2cm)
close_objects = objects.get_detected_objects(
    location=Location.CLOSE_TO,
    coordinate=reference
)
print(f"Objects nearby: {len(close_objects)}")
```

### Finding Objects by Label

```python
# Find all pens
pens = objects.get_detected_objects(label="pen")
print(f"Found {len(pens)} pen(s)")

# Note: Label matching is substring-based
# "pen" will match both "pen" and "pencil"
pencils = objects.get_detected_objects(label="pencil")
print(f"Found {len(pencils)} pencil(s)")

# Combine filters
left_pens = objects.get_detected_objects(
    location=Location.LEFT_NEXT_TO,
    coordinate=[0.25, 0.0],
    label="pen"
)
```

### Finding Nearest Object

```python
# Find nearest object to a point
point = [0.25, 0.05]
nearest, distance = objects.get_nearest_detected_object(point)

print(f"Nearest object: {nearest.label()}")
print(f"Distance: {distance:.3f}m")

# Find nearest with specific label
nearest_pen, distance = objects.get_nearest_detected_object(
    point,
    label="pen"
)
```

### Size-Based Queries

```python
# Find largest object
largest, size = objects.get_largest_detected_object()
print(f"Largest: {largest.label()} ({size * 10000:.2f} cm²)")

# Find smallest object
smallest, size = objects.get_smallest_detected_object()
print(f"Smallest: {smallest.label()} ({size * 10000:.2f} cm²)")

# Get all objects sorted by size
sorted_ascending = objects.get_detected_objects_sorted(ascending=True)
sorted_descending = objects.get_detected_objects_sorted(ascending=False)

print("\nObjects by size (ascending):")
for obj in sorted_ascending:
    print(f"  {obj.label()}: {obj.size_m2() * 10000:.2f} cm²")
```

### Specific Object Lookup

```python
# Get object at specific coordinate with label
obj = objects.get_detected_object(
    coordinate=[0.2, 0.1],
    label="pencil"
)

if obj:
    print(f"Found: {obj.label()} at {obj.coordinate()}")
else:
    print("Object not found")
```

---

## Serialization

### Object Serialization

```python
# Serialize single object to dictionary
obj_dict = obj.to_dict()
print(f"Serialized keys: {list(obj_dict.keys())}")

# Serialize to JSON string
json_str = obj.to_json()
print(f"JSON length: {len(json_str)} characters")

# Deserialize from dictionary
reconstructed = Object.from_dict(obj_dict, workspace)
print(f"Reconstructed: {reconstructed.label()}")

# Deserialize from JSON string
reconstructed = Object.from_json(json_str, workspace)
print(f"Reconstructed: {reconstructed.label()}")
```

### Collection Serialization

```python
# Serialize collection to list of dictionaries
dict_list = Objects.objects_to_dict_list(objects)
print(f"Serialized {len(dict_list)} objects")

# Deserialize back to Objects collection
reconstructed_objects = Objects.dict_list_to_objects(dict_list, workspace)
print(f"Reconstructed {len(reconstructed_objects)} objects")

# Verify reconstruction
for orig, recon in zip(objects, reconstructed_objects):
    print(f"{orig.label()} → {recon.label()}: Match!")
```

### Serializable Query Results

```python
# Get serializable results directly
obj_dict = objects.get_detected_object(
    [0.2, 0.1],
    label="pencil",
    serializable=True
)

largest_dict, size = objects.get_largest_detected_object(serializable=True)

sorted_dicts = objects.get_detected_objects_sorted(serializable=True)

# Useful for sending over network or storing in database
import json
json_data = json.dumps(sorted_dicts)
```

---

## Advanced Examples

### LLM-Friendly Formatting

```python
# Format for LLM consumption (compact)
llm_str = obj.as_string_for_llm()
print(llm_str)
# Output: "- 'pencil' at world coordinates [0.20, 0.10] with a width of
#          0.05 meters, a height of 0.08 meters and a size of 40.00
#          square centimeters."

# Format for chat window
chat_str = obj.as_string_for_chat_window()
print(chat_str)
# Output: "Detected a new object: pencil at world coordinate (0.20, 0.10)
#          with orientation 0.5 rad and size 0.05 m x 0.08 m."

# Format entire collection
for obj in objects:
    print(obj.as_string_for_llm())
```

### Working with Segmentation Masks

```python
import cv2
import numpy as np

# Create object with mask
mask = np.zeros((640, 480), dtype=np.uint8)
cv2.circle(mask, (150, 150), 50, 255, -1)  # Filled circle

obj = Object(
    label="coin",
    u_min=100, v_min=100,
    u_max=200, v_max=200,
    mask_8u=mask,
    workspace=workspace
)

# Access contour information
contour = obj.largest_contour()
if contour is not None:
    area = cv2.contourArea(contour)
    print(f"Contour area: {area} pixels")
```

### Pose Comparison and Approximation

```python
pose1 = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
pose2 = PoseObjectPNP(1.05, 2.05, 3.05, 0.12, 0.22, 0.32)

# Exact equality
is_equal = pose1 == pose2
print(f"Exactly equal: {is_equal}")  # False

# Approximate equality (full pose)
is_approx = pose1.approx_eq(
    pose2,
    eps_position=0.1,      # 10cm tolerance
    eps_orientation=0.1    # ~5.7° tolerance
)
print(f"Approximately equal: {is_approx}")  # True

# Approximate equality (position only)
is_approx_xyz = pose1.approx_eq_xyz(pose2, eps=0.1)
print(f"Position approximately equal: {is_approx_xyz}")  # True
```

### Custom Object Filtering

```python
# Filter objects by custom criteria
def filter_by_size_range(objects, min_size, max_size):
    """Get objects within size range (in cm²)"""
    return Objects([
        obj for obj in objects
        if min_size <= obj.size_m2() * 10000 <= max_size
    ])

medium_objects = filter_by_size_range(objects, 20.0, 50.0)
print(f"Medium-sized objects: {len(medium_objects)}")

# Filter by distance from point
def filter_by_distance(objects, point, max_distance):
    """Get objects within distance from point"""
    return Objects([
        obj for obj in objects
        if np.sqrt((obj.x_com() - point[0])**2 +
                  (obj.y_com() - point[1])**2) <= max_distance
    ])

nearby = filter_by_distance(objects, [0.25, 0.05], 0.1)
print(f"Objects within 10cm: {len(nearby)}")
```

---

## Integration Examples

### Pick-and-Place Workflow

```python
from robot_workspace import NiryoWorkspaces, Objects, Location

# 1. Initialize workspace
workspaces = NiryoWorkspaces(environment)
workspace = workspaces.get_home_workspace()

# 2. Detect objects (from vision system)
detected_objects = Objects([obj1, obj2, obj3])

# 3. Find target object
target = detected_objects.get_detected_object(
    coordinate=[0.2, 0.1],
    label="cube"
)

if target:
    # 4. Get pickup pose
    pickup_pose = target.pose_com()
    pickup_rotation = target.gripper_rotation()

    print(f"Pickup at: ({pickup_pose.x:.3f}, {pickup_pose.y:.3f})")
    print(f"Rotation: {np.degrees(pickup_rotation):.1f}°")

    # 5. Execute pick (using robot API)
    # robot.pick(pickup_pose, pickup_rotation)

    # 6. Define placement location
    place_pose = PoseObjectPNP(0.3, -0.1, 0.05, 0.0, 1.57, 0.0)

    # 7. Execute place
    # robot.place(place_pose)

    # 8. Update object position in memory
    target.set_pose_com(place_pose)
```

### Multi-Workspace Object Transfer

```python
# Initialize multiple workspaces
workspaces = NiryoWorkspaces(environment)
left_ws = workspaces.get_workspace_left()
right_ws = workspaces.get_workspace_right()

# Objects in left workspace
left_objects = Objects([...])

# Objects in right workspace
right_objects = Objects([...])

# Find object to transfer
transfer_obj = left_objects.get_detected_object(
    coordinate=[0.2, 0.1],
    label="cube"
)

if transfer_obj:
    # Remove from left workspace memory
    left_objects.remove(transfer_obj)

    # Calculate new position in right workspace
    # (transform from left to right workspace coordinates)
    new_pose = right_ws.transform_camera2world_coords(
        right_ws.id(),
        u_rel=0.5,
        v_rel=0.5,
        yaw=0.0
    )

    # Update object's pose
    transfer_obj.set_pose_com(new_pose)

    # Add to right workspace memory
    right_objects.append(transfer_obj)

    print(f"Transferred {transfer_obj.label()} to right workspace")
```

### Sorting Objects by Size

```python
# Detect all objects
all_objects = Objects([...])

# Sort by size
sorted_objects = all_objects.get_detected_objects_sorted(ascending=True)

# Pick and place in order (smallest to largest)
for i, obj in enumerate(sorted_objects):
    pickup_pose = obj.pose_com()

    # Calculate placement in a row
    place_pose = PoseObjectPNP(
        x=0.3,
        y=-0.1 + i * 0.05,  # Space objects 5cm apart
        z=0.05,
        roll=0.0,
        pitch=1.57,
        yaw=obj.gripper_rotation()
    )

    print(f"Moving {obj.label()} from {obj.coordinate()} to row position {i}")

    # Execute pick and place
    # robot.pick(pickup_pose, obj.gripper_rotation())
    # robot.place(place_pose)

    # Update object position
    obj.set_pose_com(place_pose)
```

### Finding and Grouping Similar Objects

```python
# Find all objects with "pen" in the label
pen_like = Objects([
    obj for obj in all_objects
    if "pen" in obj.label().lower()
])

print(f"Found {len(pen_like)} pen-like objects:")
for obj in pen_like:
    print(f"  - {obj.label()} at {obj.coordinate()}")

# Group by size category
def categorize_by_size(obj):
    size_cm2 = obj.size_m2() * 10000
    if size_cm2 < 30:
        return "small"
    elif size_cm2 < 60:
        return "medium"
    else:
        return "large"

from collections import defaultdict
grouped = defaultdict(list)

for obj in all_objects:
    category = categorize_by_size(obj)
    grouped[category].append(obj)

for category, objs in grouped.items():
    print(f"{category.capitalize()}: {len(objs)} objects")
```

### Real-time Object Tracking

```python
import time

# Initial detection
previous_objects = detect_objects_from_camera(workspace)

# Track changes
while True:
    time.sleep(0.5)  # Check every 500ms

    # Detect current objects
    current_objects = detect_objects_from_camera(workspace)

    # Find new objects
    new_labels = set(obj.label() for obj in current_objects)
    old_labels = set(obj.label() for obj in previous_objects)

    added = new_labels - old_labels
    removed = old_labels - new_labels

    if added:
        print(f"New objects detected: {added}")
    if removed:
        print(f"Objects removed: {removed}")

    # Track position changes
    for curr_obj in current_objects:
        for prev_obj in previous_objects:
            if curr_obj.label() == prev_obj.label():
                distance = np.sqrt(
                    (curr_obj.x_com() - prev_obj.x_com())**2 +
                    (curr_obj.y_com() - prev_obj.y_com())**2
                )
                if distance > 0.01:  # Moved more than 1cm
                    print(f"{curr_obj.label()} moved {distance*100:.1f}cm")

    previous_objects = current_objects
```

---

## See Also

- [Installation Guide](INSTALL.md)
- [Architecture Documentation](README.md)
- [API Reference](api.md)
- [Contributing Guide](../CONTRIBUTING.md)

---

**Author**: Daniel Gaida
**Email**: daniel.gaida@th-koeln.de
**Repository**: https://github.com/dgaida/robot_workspace
