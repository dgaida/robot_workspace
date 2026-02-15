"""
Unit tests for WidowXWorkspace and WidowXWorkspaces classes
"""

import math
from unittest.mock import Mock

import pytest

from robot_workspace.config import PoseConfig, WorkspaceConfig
from robot_workspace.objects.object import Object
from robot_workspace.objects.objects import Objects
from robot_workspace.objects.pose_object import PoseObjectPNP
from robot_workspace.workspaces.widowx_workspace import WidowXWorkspace
from robot_workspace.workspaces.widowx_workspaces import WidowXWorkspaces


@pytest.fixture
def widowx_workspace_config():
    """Create a default WidowX workspace configuration for testing"""
    return WorkspaceConfig(
        id="widowx_ws",
        observation_pose=PoseConfig(x=0.30, y=0.0, z=0.25, roll=0.0, pitch=0.0, yaw=0.0),
        image_shape=(1920, 1080, 3),
        robot_type="widowx",
    )


@pytest.fixture
def mock_widowx_environment():
    """Create a mock environment for WidowX robot"""
    env = Mock()
    env.use_simulation.return_value = False
    env.verbose.return_value = False

    # Mock for WidowX with third-person camera
    # Different from Niryo - doesn't use gripper camera
    def mock_get_target_pose(ws_id, u_rel, v_rel, yaw):
        # WidowX coordinate mapping
        # Third-person camera view
        x = 0.5 - u_rel * 0.4  # x: 0.5 to 0.1
        y = 0.2 - v_rel * 0.4  # y: 0.2 to -0.2
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    env.get_robot_target_pose_from_rel = mock_get_target_pose

    return env


class TestWidowXWorkspace:
    """Test suite for WidowXWorkspace class"""

    def test_initialization(self, mock_widowx_environment, widowx_workspace_config):
        """Test workspace initialization"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        assert workspace.id() == "widowx_ws"
        assert workspace.environment() == mock_widowx_environment

    def test_observation_pose_from_config(self, mock_widowx_environment):
        """Test observation pose comes from config"""
        custom_config = WorkspaceConfig(
            id="custom_ws",
            observation_pose=PoseConfig(x=1.0, y=2.0, z=3.0, roll=0, pitch=0, yaw=0),
            image_shape=(1920, 1080, 3),
        )
        workspace = WidowXWorkspace("custom_ws", mock_widowx_environment, config=custom_config)
        pose = workspace.observation_pose()

        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0

    def test_initialization_without_config_raises_error(self, mock_widowx_environment):
        """Test that initialization without config raises ValueError"""
        with pytest.raises(ValueError, match="No configuration provided"):
            WidowXWorkspace("widowx_ws", mock_widowx_environment)

    def test_transform_camera2world_coords(self, mock_widowx_environment, widowx_workspace_config):
        """Test camera to world coordinate transformation"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        pose = workspace.transform_camera2world_coords("widowx_ws", 0.5, 0.5, 0.0)

        assert isinstance(pose, PoseObjectPNP)
        assert pose.x is not None
        assert pose.y is not None
        assert pose.z == 0.05  # Default height
        assert pose.pitch == 1.57  # Pointing down

    def test_transform_with_yaw(self, mock_widowx_environment, widowx_workspace_config):
        """Test transformation with custom yaw"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        yaw = math.pi / 4
        pose = workspace.transform_camera2world_coords("widowx_ws", 0.5, 0.5, yaw)

        assert pose.yaw == yaw

    def test_corners_of_workspace(self, mock_widowx_environment, widowx_workspace_config):
        """Test that all four corners are set"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        assert workspace.xy_ul_wc() is not None
        assert workspace.xy_ll_wc() is not None
        assert workspace.xy_ur_wc() is not None
        assert workspace.xy_lr_wc() is not None

    def test_corners_coordinate_relationships(self, mock_widowx_environment, widowx_workspace_config):
        """Test coordinate relationships between corners"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        ul = workspace.xy_ul_wc()
        ll = workspace.xy_ll_wc()
        ur = workspace.xy_ur_wc()
        lr = workspace.xy_lr_wc()

        # Upper corners should have higher x than lower corners
        assert ul.x >= ll.x
        assert ur.x >= lr.x

        # Left corners should have higher y than right corners
        assert ul.y >= ur.y
        assert ll.y >= lr.y

    def test_width_height(self, mock_widowx_environment, widowx_workspace_config):
        """Test width and height calculation"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        width = workspace.width_m()
        height = workspace.height_m()

        assert width > 0
        assert height > 0

    def test_center_of_workspace(self, mock_widowx_environment, widowx_workspace_config):
        """Test center calculation"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)
        center = workspace.xy_center_wc()

        assert isinstance(center, PoseObjectPNP)

        # Center should be between corners
        ul = workspace.xy_ul_wc()
        lr = workspace.xy_lr_wc()

        assert lr.x <= center.x <= ul.x
        assert lr.y <= center.y <= ul.y

    def test_is_visible(self, mock_widowx_environment, widowx_workspace_config):
        """Test visibility check"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        # Get observation pose and check if visible from there
        obs_pose = workspace.observation_pose()
        is_visible = workspace.is_visible(obs_pose)

        # Should be visible from observation pose
        assert is_visible is True

    def test_is_not_visible(self, mock_widowx_environment, widowx_workspace_config):
        """Test visibility check with different pose"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        # Random dummy pose far from observation pose
        random_pose = PoseObjectPNP(10.0, 10.0, 10.0, 0.0, 0.0, 0.0)
        is_visible = workspace.is_visible(random_pose)

        assert is_visible is False

    def test_set_img_shape(self, mock_widowx_environment, widowx_workspace_config):
        """Test setting image shape"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        workspace.set_img_shape((1920, 1080, 3))  # HD resolution
        shape = workspace.img_shape()

        assert shape == (1920, 1080, 3)

    def test_str_representation(self, mock_widowx_environment, widowx_workspace_config):
        """Test string representation"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)
        str_repr = str(workspace)

        assert "widowx_ws" in str_repr
        assert "Workspace" in str_repr

    def test_repr_equals_str(self, mock_widowx_environment, widowx_workspace_config):
        """Test that repr equals str"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)
        assert repr(workspace) == str(workspace)

    def test_verbose_property(self, mock_widowx_environment, widowx_workspace_config):
        """Test verbose property"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, verbose=True, config=widowx_workspace_config)
        assert workspace.verbose() is True

    def test_workspace_comparison_with_niryo(self, mock_widowx_environment, widowx_workspace_config):
        """Test that WidowX workspace behaves differently from Niryo"""
        widowx_ws = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        from robot_workspace.config import PoseConfig as NiryoPoseConfig
        from robot_workspace.config import WorkspaceConfig as NiryoWorkspaceConfig
        from robot_workspace.workspaces.niryo_workspace import NiryoWorkspace
        niryo_config = NiryoWorkspaceConfig(
            id="niryo_ws",
            observation_pose=NiryoPoseConfig(x=0.173, y=-0.002, z=0.277, roll=-3.042, pitch=1.327, yaw=-3.027),
            image_shape=(640, 480, 3)
        )
        NiryoWorkspace("niryo_ws", mock_widowx_environment, config=niryo_config)

        # WidowX uses third-person camera, different observation pose
        obs_pose = widowx_ws.observation_pose()

        # WidowX observation pose has horizontal orientation (pitch=0)
        # Unlike Niryo which has downward pointing camera (pitch=1.57)
        assert obs_pose.pitch == 0.0
        assert obs_pose.roll == 0.0


class TestWidowXWorkspaces:
    """Test suite for WidowXWorkspaces class"""

    def test_initialization_real_robot(self, mock_widowx_environment):
        """Test initialization with real robot"""
        mock_widowx_environment.use_simulation.return_value = False

        workspaces = WidowXWorkspaces(mock_widowx_environment)

        # In config/widowx_config.yaml there are 5 real workspaces
        assert len(workspaces) == 5
        assert workspaces[0].id() == "widowx_ws"

    def test_initialization_simulation(self, mock_widowx_environment):
        """Test initialization with simulation"""
        mock_widowx_environment.use_simulation.return_value = True

        workspaces = WidowXWorkspaces(mock_widowx_environment)

        # In config/widowx_config.yaml there are 2 simulation workspaces
        assert len(workspaces) == 2
        assert workspaces[0].id() == "gazebo_widowx_1"
        assert workspaces[1].id() == "gazebo_widowx_2"

    def test_get_workspace_main(self, mock_widowx_environment):
        """Test getting main workspace"""
        workspaces = WidowXWorkspaces(mock_widowx_environment)

        main_ws = workspaces.get_workspace_main()
        assert main_ws is not None
        assert main_ws.id() == "widowx_ws"

    def test_get_workspace_left_none(self, mock_widowx_environment):
        """Test getting left workspace when not available"""
        mock_widowx_environment.use_simulation.return_value = False
        workspaces = WidowXWorkspaces(mock_widowx_environment)

        left_ws = workspaces.get_workspace_left()
        assert left_ws is None

    def test_get_workspace_right_exists(self, mock_widowx_environment):
        """Test getting right workspace"""
        mock_widowx_environment.use_simulation.return_value = False
        workspaces = WidowXWorkspaces(mock_widowx_environment)

        right_ws = workspaces.get_workspace_right()
        assert right_ws is not None

    def test_get_workspace_main_id(self, mock_widowx_environment):
        """Test getting main workspace ID"""
        workspaces = WidowXWorkspaces(mock_widowx_environment)

        main_id = workspaces.get_workspace_main_id()
        assert main_id == "widowx_ws"

    def test_get_workspace_left_id_none(self, mock_widowx_environment):
        """Test getting left workspace ID when not available"""
        mock_widowx_environment.use_simulation.return_value = False
        workspaces = WidowXWorkspaces(mock_widowx_environment)

        left_id = workspaces.get_workspace_left_id()
        assert left_id is None

    def test_get_workspace_right_id_exists(self, mock_widowx_environment):
        """Test getting right workspace ID"""
        mock_widowx_environment.use_simulation.return_value = False
        workspaces = WidowXWorkspaces(mock_widowx_environment)

        right_id = workspaces.get_workspace_right_id()
        assert right_id is not None

    def test_can_add_more_workspaces(self, mock_widowx_environment, widowx_workspace_config):
        """Test that additional workspaces can be added"""
        workspaces = WidowXWorkspaces(mock_widowx_environment)

        # Start with 5 workspaces
        assert len(workspaces) == 5

        # Add another workspace
        ws = WidowXWorkspace("custom_ws", mock_widowx_environment, config=widowx_workspace_config)
        workspaces.append_workspace(ws)

        assert len(workspaces) == 6

    def test_verbose_initialization(self, mock_widowx_environment):
        """Test initialization with verbose output"""
        workspaces = WidowXWorkspaces(mock_widowx_environment, verbose=True)

        assert workspaces.verbose() is True
        assert len(workspaces) >= 1


class TestWidowXWorkspaceIntegration:
    """Integration tests for WidowX workspace"""

    def test_object_creation_with_widowx_workspace(self, mock_widowx_environment, widowx_workspace_config):
        """Test creating objects in WidowX workspace"""
        workspaces = WidowXWorkspaces(mock_widowx_environment)
        workspace = workspaces.get_workspace_main()
        workspace.set_img_shape((1920, 1080, 3))

        # Create object
        obj = Object("cube", 500, 400, 600, 500, None, workspace)

        assert obj.label() == "cube"
        assert obj.workspace() == workspace

        # Verify coordinates are calculated
        x, y = obj.coordinate()
        assert x is not None
        assert y is not None

    def test_multiple_objects_in_widowx_workspace(self, mock_widowx_environment, widowx_workspace_config):
        """Test managing multiple objects"""
        workspaces = WidowXWorkspaces(mock_widowx_environment)
        workspace = workspaces.get_workspace_main()
        workspace.set_img_shape((1920, 1080, 3))

        objects = Objects()

        # Create multiple objects
        obj1 = Object("cylinder", 400, 300, 500, 400, None, workspace)
        obj2 = Object("sphere", 700, 500, 800, 600, None, workspace)
        obj3 = Object("cube", 1000, 700, 1100, 800, None, workspace)

        objects.append(obj1)
        objects.append(obj2)
        objects.append(obj3)

        assert len(objects) == 3

        # Test spatial queries
        largest, _ = objects.get_largest_detected_object()
        assert largest is not None

    def test_coordinate_transformation_accuracy(self, mock_widowx_environment, widowx_workspace_config):
        """Test coordinate transformation accuracy"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        # Test corner transformations
        ul = workspace.transform_camera2world_coords("widowx_ws", 0.0, 0.0, 0.0)
        lr = workspace.transform_camera2world_coords("widowx_ws", 1.0, 1.0, 0.0)

        # Upper-left should have higher x and y than lower-right
        assert ul.x > lr.x
        assert ul.y > lr.y

    def test_workspace_dimensions_calculation(self, mock_widowx_environment, widowx_workspace_config):
        """Test workspace dimensions are calculated correctly"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        width = workspace.width_m()
        height = workspace.height_m()

        # Verify dimensions are positive and reasonable
        assert 0.1 < width < 1.0
        assert 0.1 < height < 1.0

    def test_simulation_vs_real_workspaces(self, mock_widowx_environment):
        """Test differences between simulation and real workspaces"""
        # Real robot
        mock_widowx_environment.use_simulation.return_value = False
        real_workspaces = WidowXWorkspaces(mock_widowx_environment)

        # Simulation
        mock_widowx_environment.use_simulation.return_value = True
        sim_workspaces = WidowXWorkspaces(mock_widowx_environment)

        # Real robot has 5 workspaces
        assert len(real_workspaces) == 5
        assert len(sim_workspaces) == 2

        # Workspace IDs differ
        assert real_workspaces[0].id() == "widowx_ws"
        assert sim_workspaces[0].id() == "gazebo_widowx_1"


class TestWidowXWorkspaceEdgeCases:
    """Test edge cases and error handling"""

    def test_transform_at_boundaries(self, mock_widowx_environment, widowx_workspace_config):
        """Test transformation at workspace boundaries"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        # Test all four corners
        corners = [
            (0.0, 0.0),  # Upper-left
            (1.0, 0.0),  # Lower-left
            (0.0, 1.0),  # Upper-right
            (1.0, 1.0),  # Lower-right
        ]

        for u, v in corners:
            pose = workspace.transform_camera2world_coords("widowx_ws", u, v, 0.0)
            assert pose is not None
            assert isinstance(pose, PoseObjectPNP)

    def test_transform_outside_boundaries(self, mock_widowx_environment, widowx_workspace_config):
        """Test transformation with values outside [0, 1] range"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        # Should still work, but coordinates will be extrapolated
        pose = workspace.transform_camera2world_coords("widowx_ws", 1.5, 1.5, 0.0)
        assert pose is not None

    def test_very_small_workspace(self, mock_widowx_environment, widowx_workspace_config):
        """Test with very small workspace dimensions"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        # Width and height should still be positive
        assert workspace.width_m() > 0
        assert workspace.height_m() > 0

    def test_workspace_with_zero_yaw(self, mock_widowx_environment, widowx_workspace_config):
        """Test transformation with zero yaw"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        pose = workspace.transform_camera2world_coords("widowx_ws", 0.5, 0.5, 0.0)
        assert pose.yaw == 0.0

    def test_workspace_with_large_yaw(self, mock_widowx_environment, widowx_workspace_config):
        """Test transformation with large yaw angle"""
        workspace = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        large_yaw = 2 * math.pi
        pose = workspace.transform_camera2world_coords("widowx_ws", 0.5, 0.5, large_yaw)
        assert pose.yaw == large_yaw


class TestWidowXVsNiryoComparison:
    """Test differences between WidowX and Niryo workspaces"""

    def test_camera_setup_difference(self, mock_widowx_environment, widowx_workspace_config):
        """Test that WidowX and Niryo have different camera setups"""
        widowx_ws = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        # WidowX uses third-person camera
        obs_pose = widowx_ws.observation_pose()

        # WidowX observation pose is horizontal (for arm to be out of view)
        assert obs_pose.pitch == 0.0

        # Niryo would have pitch ≈ 1.57 (pointing down from gripper)

    def test_observation_pose_philosophy(self, mock_widowx_environment, widowx_workspace_config):
        """Test different observation pose philosophies"""
        widowx_ws = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        obs_pose = widowx_ws.observation_pose()

        # WidowX: arm retracted, camera has clear view
        # Observation pose is where arm doesn't obstruct camera
        assert obs_pose.x == 0.30  # Retracted position
        assert obs_pose.z == 0.25  # Moderate height

    def test_coordinate_system_consistency(self, mock_widowx_environment, widowx_workspace_config):
        """Test that coordinate systems are consistent"""
        widowx_ws = WidowXWorkspace("widowx_ws", mock_widowx_environment, config=widowx_workspace_config)

        # Test that transformations follow same conventions
        center = widowx_ws.transform_camera2world_coords("widowx_ws", 0.5, 0.5, 0.0)

        # Center should be in middle of workspace
        ul = widowx_ws.xy_ul_wc()
        lr = widowx_ws.xy_lr_wc()

        # Check center is between corners
        assert lr.x < center.x < ul.x
        assert lr.y < center.y < ul.y


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
