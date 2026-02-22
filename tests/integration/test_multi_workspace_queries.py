"""
Integration tests for spatial queries across multiple workspaces.
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
            # Left workspace
            x = 0.4 - u_rel * 0.3
            y = 0.25 - v_rel * 0.3
            return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)
        elif ws_id == "gazebo_2":
            # Right workspace
            x = 0.4 - u_rel * 0.3
            y = -0.05 - v_rel * 0.3
            return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)
        return PoseObjectPNP(0, 0, 0, 0, 0, 0)

    env.get_robot_target_pose_from_rel = mock_get_target_pose
    return env


@pytest.mark.integration
class TestMultiWorkspaceQueries:
    """Integration tests for queries involving multiple workspaces."""

    def test_size_queries_across_two_workspaces(self, mock_environment):
        """Test get_largest/smallest_detected_object with objects in different workspaces."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)
        left_ws = workspaces.get_workspace_by_id("gazebo_1")
        right_ws = workspaces.get_workspace_by_id("gazebo_2")

        left_ws.set_img_shape((640, 480, 3))
        right_ws.set_img_shape((640, 480, 3))

        all_objects = Objects()

        # Large object in left workspace
        # Bbox: (100, 100) to (300, 300) -> 200x200 pixels
        obj_large = Object("large_cube", 100, 100, 300, 300, None, left_ws)
        all_objects.append(obj_large)

        # Small object in right workspace
        # Bbox: (200, 200) to (250, 250) -> 50x50 pixels
        obj_small = Object("small_sphere", 200, 200, 250, 250, None, right_ws)
        all_objects.append(obj_small)

        # Medium object in left workspace
        # Bbox: (150, 150) to (250, 250) -> 100x100 pixels
        obj_medium = Object("medium_cylinder", 150, 150, 250, 250, None, left_ws)
        all_objects.append(obj_medium)

        # 1. Test get_largest_detected_object
        largest, size_l = all_objects.get_largest_detected_object()
        assert largest == obj_large
        assert largest.label() == "large_cube"
        assert largest.workspace() == left_ws

        # 2. Test get_smallest_detected_object
        smallest, size_s = all_objects.get_smallest_detected_object()
        assert smallest == obj_small
        assert smallest.label() == "small_sphere"
        assert smallest.workspace() == right_ws

        # 3. Test sorting
        sorted_objs = all_objects.get_detected_objects_sorted(ascending=True)
        assert len(sorted_objs) == 3
        assert sorted_objs[0] == obj_small
        assert sorted_objs[1] == obj_medium
        assert sorted_objs[2] == obj_large

        # 4. Verify sizes are consistent
        assert size_l > size_s
        assert obj_large.size_m2() > obj_medium.size_m2() > obj_small.size_m2()

    def test_spatial_queries_across_workspaces(self, mock_environment):
        """Test spatial queries when objects are in different workspaces."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)
        left_ws = workspaces.get_workspace_by_id("gazebo_1")
        right_ws = workspaces.get_workspace_by_id("gazebo_2")

        left_ws.set_img_shape((640, 480, 3))
        right_ws.set_img_shape((640, 480, 3))

        all_objects = Objects()

        # Object in left workspace (y around 0.1)
        obj_left = Object("left_obj", 200, 240, 250, 290, None, left_ws)
        all_objects.append(obj_left)

        # Object in right workspace (y around -0.1)
        obj_right = Object("right_obj", 200, 240, 250, 290, None, right_ws)
        all_objects.append(obj_right)

        # Query relative to y=0.0
        # For Niryo, y increases to the left
        # left_obj should have y > 0
        # right_obj should have y < 0

        from robot_workspace.objects.object_api import Location

        # Objects "left next to" y=0.0 (meaning higher y)
        left_results = all_objects.get_detected_objects(Location.LEFT_NEXT_TO, coordinate=[0.25, 0.0])
        assert obj_left in left_results
        assert obj_right not in left_results

        # Objects "right next to" y=0.0 (meaning lower y)
        right_results = all_objects.get_detected_objects(Location.RIGHT_NEXT_TO, coordinate=[0.25, 0.0])
        assert obj_right in right_results
        assert obj_left not in right_results
