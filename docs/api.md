# Robot Workspace - API Reference

## PoseObjectPNP

Represents a 6-DOF pose (position + orientation).

**Key Methods:**
- `to_list()` - Convert to [x, y, z, roll, pitch, yaw]
- `to_transformation_matrix()` - Get 4x4 transformation matrix
- `quaternion` - Get quaternion representation [qx, qy, qz, qw]
- `copy_with_offsets()` - Create new pose with offsets
- `approx_eq()` - Check approximate equality

## Object

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

## Objects

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

## Workspace

Abstract base class for robot workspaces.

**Key Methods:**
- `transform_camera2world_coords(workspace_id, u_rel, v_rel, yaw)` - Coordinate transformation
- `is_visible(camera_pose)` - Check if workspace is visible
- `observation_pose()` - Get optimal viewing pose

**Key Properties:**
- `id()` - Workspace identifier
- `width_m()` / `height_m()` - Dimensions in meters
- `xy_ul_wc()`, `xy_lr_wc()` - Corner coordinates
