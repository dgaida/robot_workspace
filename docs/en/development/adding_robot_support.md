# 🔧 Adding Robot Support

To add support for a new robot platform:

## 1. Create a Workspace Class

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

## 2. Create a Workspaces Collection

```python
from robot_workspace.workspaces.workspaces import Workspaces

class MyRobotWorkspaces(Workspaces):
    def __init__(self, environment, verbose: bool = False):
        super().__init__(verbose)
        workspace = MyRobotWorkspace("my_workspace_id", environment, verbose)
        self.append_workspace(workspace)
```

## 3. Integrate with Environment

Your `Environment` class should provide:
- `use_simulation()` - Returns True if in simulation mode
- `get_robot_target_pose_from_rel(workspace_id, u_rel, v_rel, yaw)` - Coordinate transformation

See [WidowXWorkspace](robot_workspace/workspaces/widowx_workspace.py) for a complete example.
