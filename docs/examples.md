# Robot Workspace - Examples

Common use cases and example workflows.

## Managing Workspaces

```python
from unittest.mock import Mock
from robot_workspace.workspaces.niryo_workspaces import NiryoWorkspaces
from robot_workspace.objects.pose_object import PoseObjectPNP

# Create a mock environment (for demo/testing without real robot)
def create_mock_environment(use_simulation=True):
    env = Mock()
    env.use_simulation.return_value = use_simulation
    env.verbose.return_value = False

    # Mock coordinate transformation
    def mock_get_target_pose(ws_id, u_rel, v_rel, yaw):
        x = 0.4 - u_rel * 0.3
        y = 0.15 - v_rel * 0.3
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    env.get_robot_target_pose_from_rel = mock_get_target_pose
    return env

# Create workspace collection with mock environment
mock_env = create_mock_environment(use_simulation=True)
workspaces = NiryoWorkspaces(mock_env, verbose=False)

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

**Note**: For actual robot integration, you would replace the mock environment with a real `Environment` object from your robot control package (e.g., `pyniryo` for Niryo robots).

## Working with Objects

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

## Object Serialization

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

## LLM-Friendly Formatting

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

---

For more information:
- [API Reference](api.md) - Complete documentation
