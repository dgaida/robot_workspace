# Robot Workspace - API Reference

Complete API documentation for the `robot_workspace` package.

## Table of Contents

- [Object Layer](#object-layer)
  - [PoseObjectPNP](#poseobjectpnp)
  - [Object](#object)
  - [Objects](#objects)
  - [Location](#location)
- [Workspace Layer](#workspace-layer)
  - [Workspace](#workspace)
  - [Workspaces](#workspaces)
  - [NiryoWorkspace](#niryoworkspace)
  - [NiryoWorkspaces](#niryoworkspaces)
  - [WidowXWorkspace](#widowxworkspace)
  - [WidowXWorkspaces](#widowxworkspaces)

---

## Object Layer

### PoseObjectPNP

Represents a 6-DOF pose (position + orientation) in 3D space.

#### Constructor

```python
PoseObjectPNP(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
```

**Parameters:**
- `x` (float): X-coordinate in meters
- `y` (float): Y-coordinate in meters
- `z` (float): Z-coordinate in meters
- `roll` (float): Roll angle in radians
- `pitch` (float): Pitch angle in radians
- `yaw` (float): Yaw angle in radians

#### Properties

```python
pose.x          # X-coordinate (float)
pose.y          # Y-coordinate (float)
pose.z          # Z-coordinate (float)
pose.roll       # Roll angle (float)
pose.pitch      # Pitch angle (float)
pose.yaw        # Yaw angle (float)
```

#### Methods

##### `to_list() -> List[float]`

Convert pose to list format `[x, y, z, roll, pitch, yaw]`.

```python
pose = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
pose_list = pose.to_list()
# [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
```

##### `to_transformation_matrix() -> np.ndarray`

Get 4×4 homogeneous transformation matrix.

```python
matrix = pose.to_transformation_matrix()
# [[R11, R12, R13, x],
#  [R21, R22, R23, y],
#  [R31, R32, R33, z],
#  [0,   0,   0,   1]]
```

##### `xy_coordinate() -> List[float]`

Get XY coordinates only.

```python
xy = pose.xy_coordinate()  # [x, y]
```

##### `copy_with_offsets(...) -> PoseObjectPNP`

Create a new pose with offsets applied.

```python
new_pose = pose.copy_with_offsets(
    x_offset=0.1,
    y_offset=0.05,
    z_offset=0.0,
    roll_offset=0.0,
    pitch_offset=0.0,
    yaw_offset=0.5
)
```

##### `approx_eq(other, eps_position=0.1, eps_orientation=0.1) -> bool`

Check approximate equality with tolerance.

```python
pose1 = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
pose2 = PoseObjectPNP(1.05, 2.05, 3.05, 0.12, 0.22, 0.32)

is_equal = pose1.approx_eq(pose2, eps_position=0.1, eps_orientation=0.1)
# True
```

##### `approx_eq_xyz(other, eps=0.1) -> bool`

Check approximate equality for position only.

```python
is_equal = pose1.approx_eq_xyz(pose2, eps=0.1)
```

#### Operators

```python
# Addition
pose3 = pose1 + pose2

# Subtraction
pose_diff = pose1 - pose2

# Equality
is_equal = pose1 == pose2
```

#### Static Methods

##### `euler_to_quaternion(roll, pitch, yaw) -> List[float]`

Convert Euler angles to quaternion `[qx, qy, qz, qw]`.

```python
quat = PoseObjectPNP.euler_to_quaternion(0.1, 0.2, 0.3)
```

##### `quaternion_to_euler_angle(qx, qy, qz, qw) -> Tuple[float, float, float]`

Convert quaternion to Euler angles `(roll, pitch, yaw)`.

```python
roll, pitch, yaw = PoseObjectPNP.quaternion_to_euler_angle(qx, qy, qz, qw)
```

---

### Object

Represents a detected object in the workspace.

#### Constructor

```python
Object(label, u_min, v_min, u_max, v_max, mask_8u, workspace, verbose=False)
```

**Parameters:**
- `label` (str): Object label (e.g., "pencil", "cube")
- `u_min` (int): Bounding box upper-left u-coordinate (pixels)
- `v_min` (int): Bounding box upper-left v-coordinate (pixels)
- `u_max` (int): Bounding box lower-right u-coordinate (pixels)
- `v_max` (int): Bounding box lower-right v-coordinate (pixels)
- `mask_8u` (np.ndarray or None): Segmentation mask (8-bit uint8)
- `workspace` (Workspace): Workspace where object is located
- `verbose` (bool): Enable verbose logging

#### Properties

##### Position Properties

```python
obj.label()              # str: Object label
obj.coordinate()         # List[float]: [x, y] world coordinates
obj.x_com()              # float: X-coordinate of center of mass
obj.y_com()              # float: Y-coordinate of center of mass
obj.xy_com()             # Tuple[float, float]: (x, y) center of mass
obj.x_center()           # float: X-coordinate of center
obj.y_center()           # float: Y-coordinate of center
obj.xy_center()          # Tuple[float, float]: (x, y) center
obj.pose_com()           # PoseObjectPNP: Full pose at center of mass
obj.pose_center()        # PoseObjectPNP: Full pose at center
```

##### Dimension Properties

```python
obj.width_m()            # float: Width in meters
obj.height_m()           # float: Height in meters
obj.shape_m()            # Tuple[float, float]: (width, height) in meters
obj.size_m2()            # float: Area in square meters
```

##### Orientation Property

```python
obj.gripper_rotation()   # float: Optimal gripper rotation (radians)
```

##### Other Properties

```python
obj.workspace()          # Workspace: Associated workspace
obj.get_workspace_id()   # str: Workspace ID
obj.largest_contour()    # np.ndarray: Largest contour from segmentation
obj.verbose()            # bool: Verbose setting
```

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert object to dictionary for serialization.

```python
obj_dict = obj.to_dict()
# {
#     "id": "abc123",
#     "label": "pencil",
#     "position": {...},
#     "dimensions": {...},
#     ...
# }
```

##### `to_json() -> str`

Convert object to JSON string.

```python
json_str = obj.to_json()
```

##### `from_dict(data, workspace) -> Object` (class method)

Reconstruct object from dictionary.

```python
obj = Object.from_dict(obj_dict, workspace)
```

##### `from_json(json_str, workspace) -> Object` (class method)

Reconstruct object from JSON string.

```python
obj = Object.from_json(json_str, workspace)
```

##### `set_pose_com(pose_com: PoseObjectPNP) -> None`

Update object's center of mass pose. Handles rotation and translation.

```python
new_pose = obj.pose_com().copy_with_offsets(x_offset=0.1, y_offset=0.05)
obj.set_pose_com(new_pose)
```

##### `set_position(xy_coordinate: List[float]) -> None`

Legacy method to update XY position (keeps Z and orientation).

```python
obj.set_position([0.3, 0.05])
```

##### `as_string_for_llm() -> str`

Format object for LLM consumption (compact).

```python
description = obj.as_string_for_llm()
# "- 'pencil' at world coordinates [0.20, 0.10] with a width of
#    0.05 meters, a height of 0.08 meters and a size of 40.00
#    square centimeters."
```

##### `as_string_for_chat_window() -> str`

Format object for chat display.

```python
description = obj.as_string_for_chat_window()
# "Detected a new object: pencil at world coordinate (0.20, 0.10)
#  with orientation 0.5 rad and size 0.05 m x 0.08 m."
```

##### `generate_object_id() -> str`

Generate unique identifier for object.

```python
obj_id = obj.generate_object_id()  # "abc123de"
```

#### Static Methods

##### `calc_width_height(pose_ul, pose_lr) -> Tuple[float, float]`

Calculate width and height between two poses.

```python
width, height = Object.calc_width_height(pose_upper_left, pose_lower_right)
```

---

### Objects

Collection of Object instances with spatial query methods.

#### Constructor

```python
Objects(iterable=None, verbose=False)
```

**Parameters:**
- `iterable` (iterable of Object): Initial objects
- `verbose` (bool): Enable verbose logging

#### Methods

##### `get_detected_object(coordinate, label=None, serializable=False)`

Get specific object at/near coordinate.

**Parameters:**
- `coordinate` (List[float]): [x, y] world coordinates
- `label` (str, optional): Filter by label
- `serializable` (bool): Return dict instead of Object

**Returns:**
- `Object` or `Dict` or `None`

```python
obj = objects.get_detected_object([0.2, 0.1], label="pencil")
```

##### `get_detected_objects(location=Location.NONE, coordinate=None, label=None)`

Get objects with optional filters.

**Parameters:**
- `location` (Location or str): Spatial filter
- `coordinate` (List[float]): Reference coordinate for spatial filter
- `label` (str, optional): Filter by label

**Returns:**
- `Objects`: Filtered collection

**Location filters:**
- `Location.LEFT_NEXT_TO`: y > coordinate[1]
- `Location.RIGHT_NEXT_TO`: y < coordinate[1]
- `Location.ABOVE`: x > coordinate[0]
- `Location.BELOW`: x < coordinate[0]
- `Location.CLOSE_TO`: within 2cm radius

```python
# Get all pencils
pencils = objects.get_detected_objects(label="pencil")

# Get objects to the left
left_objs = objects.get_detected_objects(
    location=Location.LEFT_NEXT_TO,
    coordinate=[0.2, 0.0]
)

# Get objects close to a point
nearby = objects.get_detected_objects(
    location=Location.CLOSE_TO,
    coordinate=[0.25, 0.05]
)
```

##### `get_detected_objects_serializable(...)`

Same as `get_detected_objects` but returns `List[Dict]`.

##### `get_nearest_detected_object(coordinate, label=None)`

Find nearest object to coordinate.

**Returns:**
- `Tuple[Object, float]`: (nearest_object, distance)

```python
nearest, distance = objects.get_nearest_detected_object([0.2, 0.1])
print(f"Nearest: {nearest.label()} at {distance:.3f}m")
```

##### `get_largest_detected_object(serializable=False)`

Get largest object by area.

**Returns:**
- `Tuple[Object, float]` or `Tuple[Dict, float]`: (object, size_m2)

```python
largest, size = objects.get_largest_detected_object()
print(f"Largest: {largest.label()} ({size*10000:.2f} cm²)")
```

##### `get_smallest_detected_object(serializable=False)`

Get smallest object by area.

**Returns:**
- `Tuple[Object, float]` or `Tuple[Dict, float]`: (object, size_m2)

##### `get_detected_objects_sorted(ascending=True, serializable=False)`

Get objects sorted by size.

**Returns:**
- `Objects` or `List[Dict]`: Sorted collection

```python
sorted_objs = objects.get_detected_objects_sorted(ascending=True)
for obj in sorted_objs:
    print(f"{obj.label()}: {obj.size_m2()*10000:.2f} cm²")
```

##### `get_detected_objects_as_comma_separated_string() -> str`

Get comma-separated string of labels.

```python
labels_str = objects.get_detected_objects_as_comma_separated_string()
# "'pencil', 'pen', 'eraser'"
```

#### Static Methods

##### `objects_to_dict_list(objects) -> List[Dict]`

Convert Objects collection to list of dictionaries.

```python
dict_list = Objects.objects_to_dict_list(objects)
```

##### `dict_list_to_objects(dict_list, workspace) -> Objects`

Convert list of dictionaries back to Objects collection.

```python
objects = Objects.dict_list_to_objects(dict_list, workspace)
```

---

### Location

Enum defining spatial relationships.

#### Values

```python
Location.LEFT_NEXT_TO     # "left next to"
Location.RIGHT_NEXT_TO    # "right next to"
Location.ABOVE            # "above"
Location.BELOW            # "below"
Location.ON_TOP_OF        # "on top of"
Location.INSIDE           # "inside"
Location.CLOSE_TO         # "close to"
Location.NONE             # None
```

#### Static Methods

##### `convert_str2location(location) -> Location`

Convert string or Location to Location enum.

```python
loc = Location.convert_str2location("left next to")
# Location.LEFT_NEXT_TO

loc = Location.convert_str2location(Location.ABOVE)
# Location.ABOVE (identity)

loc = Location.convert_str2location(None)
# Location.NONE
```

---

## Workspace Layer

### Workspace

Abstract base class for robot workspaces.

#### Constructor

```python
Workspace(workspace_id, verbose=False)
```

**Parameters:**
- `workspace_id` (str): Unique workspace identifier
- `verbose` (bool): Enable verbose logging

#### Properties

```python
workspace.id()               # str: Workspace ID
workspace.width_m()          # float: Width in meters
workspace.height_m()         # float: Height in meters
workspace.img_shape()        # Tuple[int, int, int]: Image dimensions
workspace.observation_pose() # PoseObjectPNP: Observation pose
workspace.xy_ul_wc()         # PoseObjectPNP: Upper-left corner
workspace.xy_ur_wc()         # PoseObjectPNP: Upper-right corner
workspace.xy_ll_wc()         # PoseObjectPNP: Lower-left corner
workspace.xy_lr_wc()         # PoseObjectPNP: Lower-right corner
workspace.xy_center_wc()     # PoseObjectPNP: Center
workspace.verbose()          # bool: Verbose setting
```

#### Methods

##### `transform_camera2world_coords(workspace_id, u_rel, v_rel, yaw=0.0)`

Transform relative image coordinates to world coordinates.

**Parameters:**
- `workspace_id` (str): Workspace ID
- `u_rel` (float): Horizontal coordinate [0, 1]
- `v_rel` (float): Vertical coordinate [0, 1]
- `yaw` (float): Object orientation (radians)

**Returns:**
- `PoseObjectPNP`: World pose

```python
pose = workspace.transform_camera2world_coords(
    workspace_id="niryo_ws",
    u_rel=0.5,  # Center horizontally
    v_rel=0.5,  # Center vertically
    yaw=0.0     # No rotation
)
```

##### `is_visible(camera_pose) -> bool`

Check if workspace is visible from camera pose.

```python
is_visible = workspace.is_visible(current_camera_pose)
```

##### `set_img_shape(img_shape: Tuple[int, int, int]) -> None`

Set image dimensions.

```python
workspace.set_img_shape((640, 480, 3))
```

---

### Workspaces

Collection of Workspace instances.

#### Constructor

```python
Workspaces(verbose=False)
```

#### Methods

##### `get_workspace(index) -> Workspace`

Get workspace by index.

```python
workspace = workspaces.get_workspace(0)
```

##### `get_workspace_by_id(id) -> Workspace or None`

Get workspace by ID.

```python
workspace = workspaces.get_workspace_by_id("niryo_ws")
```

##### `get_workspace_ids() -> List[str]`

Get all workspace IDs.

```python
ids = workspaces.get_workspace_ids()
# ["niryo_ws", "niryo_ws_right"]
```

##### `get_workspace_id(index) -> str`

Get workspace ID by index.

```python
ws_id = workspaces.get_workspace_id(0)
```

##### `get_home_workspace() -> Workspace`

Get home workspace (index 0).

```python
home = workspaces.get_home_workspace()
```

##### `get_workspace_home_id() -> str`

Get home workspace ID.

```python
home_id = workspaces.get_workspace_home_id()
```

##### `get_observation_pose(workspace_id) -> PoseObjectPNP`

Get observation pose for workspace.

```python
obs_pose = workspaces.get_observation_pose("niryo_ws")
```

##### `get_width_height_m(workspace_id) -> Tuple[float, float]`

Get workspace dimensions.

```python
width, height = workspaces.get_width_height_m("niryo_ws")
```

##### `get_img_shape(workspace_id) -> Tuple[int, int, int]`

Get image shape for workspace.

```python
shape = workspaces.get_img_shape("niryo_ws")
```

##### `get_visible_workspace(camera_pose) -> Workspace or None`

Get workspace visible from camera pose.

```python
visible = workspaces.get_visible_workspace(current_pose)
```

##### `append_workspace(workspace) -> None`

Add workspace to collection.

```python
workspaces.append_workspace(new_workspace)
```

---

### NiryoWorkspace

Concrete workspace implementation for Niryo Ned2 robot.

#### Constructor

```python
NiryoWorkspace(workspace_id, environment, verbose=False)
```

**Parameters:**
- `workspace_id` (str): Workspace ID (e.g., "niryo_ws", "gazebo_1")
- `environment`: Environment object with robot API
- `verbose` (bool): Enable verbose logging

#### Additional Properties

```python
workspace.environment()  # Environment object
```

#### Supported Workspace IDs

- `"niryo_ws"` - Main workspace (real robot)
- `"niryo_ws2"` - Second workspace (real robot)
- `"niryo_ws_left"` - Left workspace (multi-workspace)
- `"niryo_ws_right"` - Right workspace (multi-workspace)
- `"gazebo_1"` - First simulation workspace
- `"gazebo_2"` - Second simulation workspace

---

### NiryoWorkspaces

Collection of Niryo workspaces.

#### Constructor

```python
NiryoWorkspaces(environment, verbose=False)
```

Automatically initializes appropriate workspaces based on environment (real/simulation).

#### Additional Methods

##### `get_workspace_left() -> Workspace`

Get left workspace.

##### `get_workspace_right() -> Workspace or None`

Get right workspace (if available).

##### `get_workspace_left_id() -> str`

Get left workspace ID.

##### `get_workspace_right_id() -> str or None`

Get right workspace ID (if available).

---

### WidowXWorkspace

Concrete workspace implementation for WidowX 250 6DOF robot.

#### Constructor

```python
WidowXWorkspace(workspace_id, environment, verbose=False)
```

**Note:** Uses third-person camera view rather than gripper-mounted camera.

#### Supported Workspace IDs

- `"widowx_ws"` - Main workspace
- `"widowx_ws_main"` - Main workspace (alias)
- `"widowx_ws_left"` - Left workspace
- `"widowx_ws_right"` - Right workspace
- `"widowx_ws_extended"` - Extended reach workspace
- `"gazebo_widowx_1"` - First simulation workspace
- `"gazebo_widowx_2"` - Second simulation workspace

---

### WidowXWorkspaces

Collection of WidowX workspaces.

#### Constructor

```python
WidowXWorkspaces(environment, verbose=False)
```

#### Additional Methods

##### `get_workspace_main() -> Workspace`

Get main workspace.

##### `get_workspace_left() -> Workspace or None`

Get left workspace (if in multi-workspace setup).

##### `get_workspace_right() -> Workspace or None`

Get right workspace (if available).

---

## Usage Examples

### Basic Object Creation

```python
from robot_workspace import Object, PoseObjectPNP

# Create object
obj = Object(
    label="pencil",
    u_min=100, v_min=100,
    u_max=200, v_max=200,
    mask_8u=segmentation_mask,
    workspace=workspace
)

# Access properties
print(f"Position: {obj.coordinate()}")
print(f"Size: {obj.size_m2() * 10000:.2f} cm²")
print(f"Rotation: {obj.gripper_rotation():.3f} rad")
```

### Spatial Queries

```python
from robot_workspace import Objects, Location

objects = Objects([obj1, obj2, obj3])

# Find objects to the left of a point
left_objects = objects.get_detected_objects(
    location=Location.LEFT_NEXT_TO,
    coordinate=[0.25, 0.0]
)

# Find nearest object
nearest, distance = objects.get_nearest_detected_object([0.2, 0.1])

# Get largest object
largest, size = objects.get_largest_detected_object()
```

### Workspace Management

```python
from robot_workspace import NiryoWorkspaces

# Initialize workspaces
workspaces = NiryoWorkspaces(environment)

# Get home workspace
home_ws = workspaces.get_home_workspace()

# Transform coordinates
world_pose = home_ws.transform_camera2world_coords(
    workspace_id=home_ws.id(),
    u_rel=0.5,
    v_rel=0.5,
    yaw=0.0
)
```

### Serialization

```python
# Serialize object
obj_dict = obj.to_dict()
json_str = obj.to_json()

# Deserialize
reconstructed = Object.from_dict(obj_dict, workspace)
reconstructed = Object.from_json(json_str, workspace)

# Serialize collection
dict_list = Objects.objects_to_dict_list(objects)
reconstructed_objects = Objects.dict_list_to_objects(dict_list, workspace)
```

---

## See Also

- [Installation Guide](INSTALL.md)
- [Architecture Documentation](README.md)
- [Examples](examples.md)
- [Contributing Guide](../CONTRIBUTING.md)

---

**Author**: Daniel Gaida
**Email**: daniel.gaida@th-koeln.de
**Repository**: https://github.com/dgaida/robot_workspace
