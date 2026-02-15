"""
Unit tests for multi-workspace operations.
Create this file at: robot_workspace/tests/workspaces/test_multi_workspace.py
"""

from unittest.mock import Mock

import pytest

from robot_workspace.objects.object import Object
from robot_workspace.objects.objects import Objects
from robot_workspace.objects.pose_object import PoseObjectPNP
from robot_workspace.workspaces.niryo_workspaces import NiryoWorkspaces


@pytest.fixture
def mock_environment():
    """Create a mock environment supporting multiple workspaces."""
    env = Mock()
    env.use_simulation.return_value = True
    env.verbose.return_value = False

    # Mock different coordinate transformations for different workspaces
    def mock_get_target_pose(ws_id, u_rel, v_rel, yaw):
        if ws_id == "gazebo_1":
            # Left workspace (gazebo_1)
            x = 0.4 - u_rel * 0.3
            y = 0.25 - v_rel * 0.3  # Shifted left
            return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)
        elif ws_id == "gazebo_2":
            # Right workspace (gazebo_2)
            x = 0.4 - u_rel * 0.3
            y = -0.05 - v_rel * 0.3  # Shifted right
            return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)
        else:
            # Default transformation
            x = 0.4 - u_rel * 0.3
            y = 0.15 - v_rel * 0.3
            return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    env.get_robot_target_pose_from_rel = mock_get_target_pose
    return env


class TestMultiWorkspaceCollection:
    """Test suite for multi-workspace collection."""

    def test_multiple_workspace_initialization(self, mock_environment):
        """Test that multiple workspaces are initialized in simulation."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)

        # Should have 2 workspaces in simulation
        assert len(workspaces) >= 2

        # Check workspace IDs (simulation workspaces)
        workspace_ids = workspaces.get_workspace_ids()
        assert "gazebo_1" in workspace_ids
        assert "gazebo_2" in workspace_ids

    def test_get_workspace_left(self, mock_environment):
        """Test getting left workspace in simulation."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)

        left_ws = workspaces.get_workspace_left()
        assert left_ws is not None
        assert left_ws.id() == "gazebo_1"

    def test_get_workspace_right(self, mock_environment):
        """Test getting right workspace in simulation."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)

        right_ws = workspaces.get_workspace_right()
        assert right_ws is not None
        assert right_ws.id() == "gazebo_2"

    def test_workspace_coordinate_systems_differ(self, mock_environment):
        """Test that different workspaces have different coordinate systems."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)

        left_ws = workspaces.get_workspace_left()
        right_ws = workspaces.get_workspace_right()

        # Transform same relative coordinates in both workspaces
        left_pose = left_ws.transform_camera2world_coords(left_ws.id(), 0.5, 0.5, 0.0)
        right_pose = right_ws.transform_camera2world_coords(right_ws.id(), 0.5, 0.5, 0.0)

        # Y-coordinates should differ (workspaces are offset)
        assert abs(left_pose.y - right_pose.y) > 0.09

    def test_observation_poses_differ(self, mock_environment):
        """Test that observation poses are different for each workspace."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)

        left_ws = workspaces.get_workspace_left()
        right_ws = workspaces.get_workspace_right()

        left_obs = left_ws.observation_pose()
        right_obs = right_ws.observation_pose()

        assert left_obs is not None
        assert right_obs is not None

        # Observation poses should differ
        assert not left_obs.approx_eq(right_obs, eps_position=0.05)


class TestWorkspaceMemoryManagement:
    """Test suite for per-workspace memory management."""

    def test_separate_workspace_memories(self):
        """Test that each workspace has separate memory."""
        # Create mock workspaces
        # env = Mock()
        ws_left = Mock()
        ws_left.id.return_value = "ws_left"
        ws_right = Mock()
        ws_right.id.return_value = "ws_right"

        # Simulate separate memories
        memories = {"ws_left": Objects(), "ws_right": Objects()}

        # Add object to left workspace
        obj_left = Mock()
        obj_left.label.return_value = "cube_left"
        obj_left.workspace.return_value = ws_left
        memories["ws_left"].append(obj_left)

        # Add object to right workspace
        obj_right = Mock()
        obj_right.label.return_value = "cube_right"
        obj_right.workspace.return_value = ws_right
        memories["ws_right"].append(obj_right)

        # Verify separation
        assert len(memories["ws_left"]) == 1
        assert len(memories["ws_right"]) == 1
        assert memories["ws_left"][0] != memories["ws_right"][0]

    def test_clear_specific_workspace_memory(self):
        """Test clearing memory for a specific workspace."""
        memories = {"ws_left": Objects([Mock(), Mock()]), "ws_right": Objects([Mock()])}

        # Clear left workspace
        memories["ws_left"].clear()

        # Verify only left was cleared
        assert len(memories["ws_left"]) == 0
        assert len(memories["ws_right"]) == 1

    def test_object_transfer_between_workspace_memories(self):
        """Test moving an object from one workspace memory to another."""
        # Setup
        ws_left = Mock()
        ws_left.id.return_value = "ws_left"
        ws_right = Mock()
        ws_right.id.return_value = "ws_right"

        memories = {"ws_left": Objects(), "ws_right": Objects()}

        # Create object in left workspace
        obj = Mock()
        obj.label.return_value = "cube"
        obj.x_com.return_value = 0.2
        obj.y_com.return_value = 0.05
        obj.workspace.return_value = ws_left

        memories["ws_left"].append(obj)

        # Transfer to right workspace
        transferred_obj = memories["ws_left"][0]
        memories["ws_left"].clear()

        # Update workspace reference
        transferred_obj.workspace.return_value = ws_right
        transferred_obj.x_com.return_value = 0.25
        transferred_obj.y_com.return_value = -0.05

        memories["ws_right"].append(transferred_obj)

        # Verify transfer
        assert len(memories["ws_left"]) == 0
        assert len(memories["ws_right"]) == 1
        assert memories["ws_right"][0].workspace().id() == "ws_right"


class TestCoordinateTransformations:
    """Test suite for coordinate transformations in multiple workspaces."""

    def test_left_workspace_coordinates(self, mock_environment):
        """Test coordinate transformation in left workspace (simulation)."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)
        left_ws = workspaces.get_workspace_left()

        # Center of workspace
        pose = left_ws.transform_camera2world_coords(left_ws.id(), 0.5, 0.5, 0.0)

        # Y should be positive or around 0.0 (left side)
        assert pose.y >= -0.05

    def test_right_workspace_coordinates(self, mock_environment):
        """Test coordinate transformation in right workspace (simulation)."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)
        right_ws = workspaces.get_workspace_right()

        # Center of workspace
        pose = right_ws.transform_camera2world_coords(right_ws.id(), 0.5, 0.5, 0.0)

        # Y should be negative or around 0.0 (right side)
        assert pose.y <= 0.05

    def test_workspace_corners(self, mock_environment):
        """Test that workspace corners are correctly calculated in simulation."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)

        for workspace in workspaces:
            # All corners should be defined
            assert workspace.xy_ul_wc() is not None
            assert workspace.xy_ur_wc() is not None
            assert workspace.xy_ll_wc() is not None
            assert workspace.xy_lr_wc() is not None

            # Upper-left should have highest x and y
            ul = workspace.xy_ul_wc()
            lr = workspace.xy_lr_wc()

            assert ul.x >= lr.x  # Upper has higher x
            assert ul.y >= lr.y  # Left has higher y


class TestObjectDetectionInMultipleWorkspaces:
    """Test suite for object detection across multiple workspaces."""

    def test_create_objects_in_different_workspaces(self, mock_environment):
        """Test creating objects in different workspaces (simulation)."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)

        left_ws = workspaces.get_workspace_left()
        right_ws = workspaces.get_workspace_right()

        # Set image shape for both workspaces
        left_ws.set_img_shape((640, 480, 3))
        right_ws.set_img_shape((640, 480, 3))

        # Create object in left workspace
        obj_left = Object("cube_left", 100, 100, 200, 200, None, left_ws)

        # Create object in right workspace
        obj_right = Object("cube_right", 150, 150, 250, 250, None, right_ws)

        # Verify workspace assignment
        assert obj_left.workspace() == left_ws
        assert obj_right.workspace() == right_ws

        # Verify coordinates differ due to different workspace transforms
        assert obj_left.x_com() != obj_right.x_com() or obj_left.y_com() != obj_right.y_com()

    def test_object_collections_per_workspace(self, mock_environment):
        """Test maintaining separate object collections per workspace (simulation)."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)

        left_ws = workspaces.get_workspace_left()
        right_ws = workspaces.get_workspace_right()

        left_ws.set_img_shape((640, 480, 3))
        right_ws.set_img_shape((640, 480, 3))

        # Create collections
        left_objects = Objects()
        right_objects = Objects()

        # Add objects to left workspace
        for i in range(3):
            obj = Object(f"obj_{i}", 100 + i * 50, 100, 200 + i * 50, 200, None, left_ws)
            left_objects.append(obj)

        # Add objects to right workspace
        for i in range(2):
            obj = Object(f"obj_{i}", 150 + i * 50, 150, 250 + i * 50, 250, None, right_ws)
            right_objects.append(obj)

        # Verify separation
        assert len(left_objects) == 3
        assert len(right_objects) == 2

        # All objects in each collection belong to correct workspace
        assert all(obj.workspace() == left_ws for obj in left_objects)
        assert all(obj.workspace() == right_ws for obj in right_objects)


@pytest.mark.integration
class TestMultiWorkspaceIntegration:
    """Integration tests for multi-workspace operations."""

    def test_complete_workflow(self, mock_environment):
        """Test complete multi-workspace workflow (simulation)."""
        # Initialize workspaces
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)

        left_ws = workspaces.get_workspace_left()
        right_ws = workspaces.get_workspace_right()

        left_ws.set_img_shape((640, 480, 3))
        right_ws.set_img_shape((640, 480, 3))

        # Create objects in left workspace
        left_objects = Objects()
        obj1 = Object("cube1", 100, 100, 200, 200, None, left_ws)
        obj2 = Object("cube2", 250, 100, 350, 200, None, left_ws)
        left_objects.append(obj1)
        left_objects.append(obj2)

        # Create objects in right workspace
        right_objects = Objects()
        obj3 = Object("cylinder1", 150, 150, 250, 250, None, right_ws)
        right_objects.append(obj3)

        # Verify initial state
        assert len(left_objects) == 2
        assert len(right_objects) == 1

        # Simulate transfer: move obj1 from left to right
        transferred_obj = left_objects[0]
        left_objects.remove(transferred_obj)

        # Update object's workspace reference (simulated)
        # In real implementation, this would update the object's internal state
        right_objects.append(transferred_obj)

        # Verify final state
        assert len(left_objects) == 1  # One object remains in left
        assert len(right_objects) == 2  # Two objects now in right

        # Verify the transferred object
        assert transferred_obj in right_objects
        assert transferred_obj not in left_objects


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
