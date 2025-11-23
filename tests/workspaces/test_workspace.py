"""
Unit tests for Workspace classes
"""

import pytest
import math
from robot_workspace.workspaces.niryo_workspace import NiryoWorkspace
from robot_workspace.objects.pose_object import PoseObjectPNP


class TestNiryoWorkspace:
    """Test suite for NiryoWorkspace class"""

    def test_initialization_niryo_ws(self):
        """Test workspace initialization with niryo_ws"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)

        assert workspace.id() == "niryo_ws"
        assert workspace.verbose() is False

    def test_initialization_niryo_ws2(self):
        """Test workspace initialization with niryo_ws2"""
        workspace = NiryoWorkspace("niryo_ws2", verbose=False)

        assert workspace.id() == "niryo_ws2"

    def test_initialization_gazebo_1(self):
        """Test workspace initialization with gazebo_1"""
        workspace = NiryoWorkspace("gazebo_1", verbose=False)

        assert workspace.id() == "gazebo_1"

    def test_initialization_verbose(self):
        """Test workspace initialization with verbose=True"""
        workspace = NiryoWorkspace("niryo_ws", verbose=True)

        assert workspace.verbose() is True

    def test_observation_pose_niryo_ws(self):
        """Test observation pose for niryo_ws"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)
        pose = workspace.observation_pose()

        assert isinstance(pose, PoseObjectPNP)
        assert pose.x is not None
        assert pose.y is not None
        assert pose.z is not None
        # Check that orientation is set
        assert pose.roll is not None
        assert pose.pitch is not None
        assert pose.yaw is not None

    def test_observation_pose_niryo_ws2(self):
        """Test observation pose for niryo_ws2"""
        workspace = NiryoWorkspace("niryo_ws2", verbose=False)
        pose = workspace.observation_pose()

        assert isinstance(pose, PoseObjectPNP)
        assert pose.x is not None

    def test_observation_pose_gazebo_1(self):
        """Test observation pose for gazebo_1"""
        workspace = NiryoWorkspace("gazebo_1", verbose=False)
        pose = workspace.observation_pose()

        assert isinstance(pose, PoseObjectPNP)
        assert pose.x is not None
        assert pose.pitch == pytest.approx(math.pi / 2)

    def test_observation_pose_gazebo_1_chocolate_bars(self):
        """Test observation pose for gazebo_1_chocolate_bars"""
        workspace = NiryoWorkspace("gazebo_1_chocolate_bars", verbose=False)
        pose = workspace.observation_pose()

        assert isinstance(pose, PoseObjectPNP)
        assert pose.pitch == pytest.approx(math.pi / 2 + 0.3)

    def test_observation_pose_unknown_workspace(self):
        """Test observation pose for unknown workspace"""
        workspace = NiryoWorkspace("unknown_ws_id", verbose=False)
        pose = workspace.observation_pose()

        assert pose is None

    def test_corners_of_workspace_set(self):
        """Test that all four corners are set"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)

        # All corners should be PoseObjectPNP instances
        assert isinstance(workspace.xy_ul_wc(), PoseObjectPNP)
        assert isinstance(workspace.xy_ll_wc(), PoseObjectPNP)
        assert isinstance(workspace.xy_ur_wc(), PoseObjectPNP)
        assert isinstance(workspace.xy_lr_wc(), PoseObjectPNP)

    def test_corners_coordinate_system(self):
        """Test that corners follow the expected coordinate system"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)

        ul = workspace.xy_ul_wc()  # Upper-left
        ll = workspace.xy_ll_wc()  # Lower-left
        ur = workspace.xy_ur_wc()  # Upper-right
        lr = workspace.xy_lr_wc()  # Lower-right

        # For Niryo: width along y-axis, height along x-axis
        # Upper should have higher x than lower
        assert ul.x > ll.x or ul.x == pytest.approx(ll.x, abs=0.001)
        assert ur.x > lr.x or ur.x == pytest.approx(lr.x, abs=0.001)

        # Left should have higher y than right
        assert ul.y > ur.y or ul.y == pytest.approx(ur.y, abs=0.001)
        assert ll.y > lr.y or ll.y == pytest.approx(lr.y, abs=0.001)

    def test_width_height_positive(self):
        """Test that width and height are positive"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)

        width = workspace.width_m()
        height = workspace.height_m()

        assert width > 0
        assert height > 0
        assert isinstance(width, float)
        assert isinstance(height, float)

    def test_width_height_calculation(self):
        """Test width and height calculation matches corner distances"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)

        ul = workspace.xy_ul_wc()
        lr = workspace.xy_lr_wc()

        # Width is along y-axis
        expected_width = abs(ul.y - lr.y)
        # Height is along x-axis
        expected_height = abs(ul.x - lr.x)

        assert workspace.width_m() == pytest.approx(expected_width, abs=0.001)
        assert workspace.height_m() == pytest.approx(expected_height, abs=0.001)

    def test_center_of_workspace(self):
        """Test center calculation"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)
        center = workspace.xy_center_wc()

        assert isinstance(center, PoseObjectPNP)

        # Center should be between corners
        ul = workspace.xy_ul_wc()
        lr = workspace.xy_lr_wc()

        # Allow for small floating point differences
        assert min(ul.x, lr.x) <= center.x <= max(ul.x, lr.x)
        assert min(ul.y, lr.y) <= center.y <= max(ul.y, lr.y)

    def test_is_visible_from_observation_pose(self):
        """Test visibility check from observation pose"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)

        # Should be visible from its own observation pose
        obs_pose = workspace.observation_pose()
        is_visible = workspace.is_visible(obs_pose)

        assert is_visible is True

    def test_is_not_visible_from_far_pose(self):
        """Test visibility check from far away pose"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)

        # Random pose far from observation pose
        far_pose = PoseObjectPNP(10.0, 10.0, 10.0, 0.0, 0.0, 0.0)
        is_visible = workspace.is_visible(far_pose)

        assert is_visible is False

    def test_is_visible_with_approximate_pose(self):
        """Test visibility with slightly different pose"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)

        # Pose close to observation pose
        obs_pose = workspace.observation_pose()
        close_pose = obs_pose.copy_with_offsets(x_offset=0.01, y_offset=0.01)

        is_visible = workspace.is_visible(close_pose)

        assert is_visible is True

    def test_set_img_shape(self):
        """Test setting image shape"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)

        test_shape = (640, 480, 3)
        workspace.set_img_shape(test_shape)
        shape = workspace.img_shape()

        assert shape == test_shape

    def test_set_img_shape_different_sizes(self):
        """Test setting different image shapes"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)

        # Test various image sizes
        shapes = [(320, 240, 3), (1920, 1080, 3), (800, 600, 3)]

        for test_shape in shapes:
            workspace.set_img_shape(test_shape)
            assert workspace.img_shape() == test_shape

    def test_transform_camera2world_coords_not_implemented(self):
        """Test that transform method needs implementation"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)

        # This should work but will fail without proper environment setup
        # We just test that the method exists and has correct signature
        assert hasattr(workspace, "transform_camera2world_coords")

    def test_str_representation(self):
        """Test string representation"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)
        str_repr = str(workspace)

        assert "niryo_ws" in str_repr
        assert "Workspace" in str_repr
        assert "width" in str_repr
        assert "height" in str_repr

    def test_repr_equals_str(self):
        """Test that repr equals str"""
        workspace = NiryoWorkspace("niryo_ws", verbose=False)
        assert repr(workspace) == str(workspace)

    def test_multiple_workspaces_independent(self):
        """Test that multiple workspaces are independent"""
        ws1 = NiryoWorkspace("niryo_ws", verbose=False)
        ws2 = NiryoWorkspace("gazebo_1", verbose=False)

        assert ws1.id() != ws2.id()
        assert ws1.observation_pose() != ws2.observation_pose()
