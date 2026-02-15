"""
Unit tests for Workspace classes
"""

from unittest.mock import Mock

import pytest

from robot_workspace.config import PoseConfig, WorkspaceConfig
from robot_workspace.objects.pose_object import PoseObjectPNP
from robot_workspace.workspaces.niryo_workspace import NiryoWorkspace
from robot_workspace.workspaces.niryo_workspaces import NiryoWorkspaces
from robot_workspace.workspaces.workspaces import Workspaces


@pytest.fixture
def workspace_config():
    """Create a default workspace configuration for testing"""
    return WorkspaceConfig(
        id="niryo_ws",
        observation_pose=PoseConfig(x=0.173, y=-0.002, z=0.277, roll=-3.042, pitch=1.327, yaw=-3.027),
        image_shape=(640, 480, 3),
        robot_type="niryo",
    )


@pytest.fixture
def mock_environment():
    """Create a mock environment"""
    env = Mock()
    env.use_simulation.return_value = False
    env.verbose.return_value = False

    # Mock robot target pose method - FIXED coordinate system
    # For Niryo: width goes along y-axis, height along x-axis
    # Upper-left corner should have higher x and higher y than lower-right
    def mock_get_target_pose(ws_id, u_rel, v_rel, yaw):
        # Map relative coords to world coords properly
        # u_rel: 0 (top) -> 1 (bottom), x should go from high to low
        # v_rel: 0 (left) -> 1 (right), y should go from high to low
        x = 0.4 - u_rel * 0.3  # x decreases as u increases (0.4 to 0.1)
        y = 0.15 - v_rel * 0.3  # y decreases as v increases (0.15 to -0.15)
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    env.get_robot_target_pose_from_rel = mock_get_target_pose

    return env


class TestNiryoWorkspace:
    """Test suite for NiryoWorkspace class"""

    def test_initialization(self, mock_environment, workspace_config):
        """Test workspace initialization"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)

        assert workspace.id() == "niryo_ws"
        assert workspace.environment() == mock_environment

    def test_observation_pose_from_config(self, mock_environment):
        """Test observation pose comes from config"""
        custom_config = WorkspaceConfig(
            id="custom_ws",
            observation_pose=PoseConfig(x=1.0, y=2.0, z=3.0, roll=0, pitch=0, yaw=0),
            image_shape=(640, 480, 3),
        )
        workspace = NiryoWorkspace("custom_ws", mock_environment, config=custom_config)
        pose = workspace.observation_pose()

        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0

    def test_initialization_without_config_raises_error(self, mock_environment):
        """Test that initialization without config raises ValueError"""
        with pytest.raises(ValueError, match="No configuration provided"):
            NiryoWorkspace("niryo_ws", mock_environment)

    def test_transform_camera2world_coords(self, mock_environment, workspace_config):
        """Test camera to world coordinate transformation"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)

        pose = workspace.transform_camera2world_coords("niryo_ws", 0.5, 0.5, 0.0)

        assert isinstance(pose, PoseObjectPNP)
        assert pose.x is not None
        assert pose.y is not None

    def test_corners_of_workspace(self, mock_environment, workspace_config):
        """Test that all four corners are set"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)

        assert workspace.xy_ul_wc() is not None
        assert workspace.xy_ll_wc() is not None
        assert workspace.xy_ur_wc() is not None
        assert workspace.xy_lr_wc() is not None

    def test_width_height(self, mock_environment, workspace_config):
        """Test width and height calculation"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)

        width = workspace.width_m()
        height = workspace.height_m()

        assert width > 0
        assert height > 0

    def test_center_of_workspace(self, mock_environment, workspace_config):
        """Test center calculation"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)
        center = workspace.xy_center_wc()

        assert isinstance(center, PoseObjectPNP)

    def test_is_visible(self, mock_environment, workspace_config):
        """Test visibility check"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)

        # Get observation pose and check if visible from there
        obs_pose = workspace.observation_pose()
        is_visible = workspace.is_visible(obs_pose)

        # Should be visible from observation pose
        assert is_visible is True

    def test_is_not_visible(self, mock_environment, workspace_config):
        """Test visibility check with different pose"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)

        # Random pose far from observation pose
        random_pose = PoseObjectPNP(10.0, 10.0, 10.0, 0.0, 0.0, 0.0)
        is_visible = workspace.is_visible(random_pose)

        assert is_visible is False

    def test_set_img_shape(self, mock_environment, workspace_config):
        """Test setting image shape"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)

        workspace.set_img_shape((640, 480, 3))
        shape = workspace.img_shape()

        assert shape == (640, 480, 3)

    def test_str_representation(self, mock_environment, workspace_config):
        """Test string representation"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)
        str_repr = str(workspace)

        assert "niryo_ws" in str_repr
        assert "Workspace" in str_repr

    def test_repr_equals_str(self, mock_environment, workspace_config):
        """Test that repr equals str"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)
        assert repr(workspace) == str(workspace)

    def test_verbose_property(self, mock_environment, workspace_config):
        """Test verbose property"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, verbose=True, config=workspace_config)
        assert workspace.verbose() is True


class TestWorkspaces:
    """Test suite for Workspaces collection class"""

    def test_initialization(self, mock_environment):
        """Test workspaces collection initialization"""
        workspaces = Workspaces()

        assert len(workspaces) == 0

    def test_append_workspace(self, mock_environment, workspace_config):
        """Test appending workspace"""
        workspaces = Workspaces()
        ws = NiryoWorkspace("test_ws", mock_environment, config=workspace_config)

        workspaces.append_workspace(ws)

        assert len(workspaces) == 1
        assert workspaces[0] == ws

    def test_get_workspace(self, mock_environment, workspace_config):
        """Test getting workspace by index"""
        workspaces = Workspaces()
        ws1 = NiryoWorkspace("ws1", mock_environment, config=workspace_config)
        ws2 = NiryoWorkspace("ws2", mock_environment, config=workspace_config)

        workspaces.append_workspace(ws1)
        workspaces.append_workspace(ws2)

        assert workspaces.get_workspace(0) == ws1
        assert workspaces.get_workspace(1) == ws2

    def test_get_workspace_by_id(self, mock_environment, workspace_config):
        """Test getting workspace by ID"""
        workspaces = Workspaces()
        ws1 = NiryoWorkspace("ws1", mock_environment, config=workspace_config)
        ws2 = NiryoWorkspace("ws2", mock_environment, config=workspace_config)

        workspaces.append_workspace(ws1)
        workspaces.append_workspace(ws2)

        assert workspaces.get_workspace_by_id("ws1") == ws1
        assert workspaces.get_workspace_by_id("ws2") == ws2
        assert workspaces.get_workspace_by_id("nonexistent") is None

    def test_get_workspace_ids(self, mock_environment, workspace_config):
        """Test getting all workspace IDs"""
        workspaces = Workspaces()
        ws1 = NiryoWorkspace("ws1", mock_environment, config=workspace_config)
        ws2 = NiryoWorkspace("ws2", mock_environment, config=workspace_config)

        workspaces.append_workspace(ws1)
        workspaces.append_workspace(ws2)

        ids = workspaces.get_workspace_ids()

        assert len(ids) == 2
        assert "ws1" in ids
        assert "ws2" in ids

    def test_get_workspace_id(self, mock_environment, workspace_config):
        """Test getting workspace ID by index"""
        workspaces = Workspaces()
        ws = NiryoWorkspace("test_ws", mock_environment, config=workspace_config)

        workspaces.append_workspace(ws)

        assert workspaces.get_workspace_id(0) == "test_ws"

    def test_get_workspace_home_id(self, mock_environment, workspace_config):
        """Test getting home workspace ID"""
        workspaces = Workspaces()
        ws = NiryoWorkspace("home_ws", mock_environment, config=workspace_config)

        workspaces.append_workspace(ws)

        assert workspaces.get_workspace_home_id() == "home_ws"

    def test_get_home_workspace(self, mock_environment, workspace_config):
        """Test getting home workspace"""
        workspaces = Workspaces()
        ws = NiryoWorkspace("home_ws", mock_environment, config=workspace_config)

        workspaces.append_workspace(ws)

        assert workspaces.get_home_workspace() == ws

    def test_get_observation_pose(self, mock_environment, workspace_config):
        """Test getting observation pose by workspace ID"""
        workspaces = Workspaces()
        ws = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)

        workspaces.append_workspace(ws)

        pose = workspaces.get_observation_pose("niryo_ws")

        assert isinstance(pose, PoseObjectPNP)

    def test_get_width_height_m(self, mock_environment, workspace_config):
        """Test getting workspace dimensions"""
        workspaces = Workspaces()
        ws = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)

        workspaces.append_workspace(ws)

        width, height = workspaces.get_width_height_m("niryo_ws")

        assert width > 0
        assert height > 0

    def test_get_img_shape(self, mock_environment, workspace_config):
        """Test getting image shape"""
        workspaces = Workspaces()
        ws = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)
        ws.set_img_shape((640, 480, 3))

        workspaces.append_workspace(ws)

        shape = workspaces.get_img_shape("niryo_ws")

        assert shape == (640, 480, 3)

    def test_get_visible_workspace(self, mock_environment, workspace_config):
        """Test getting visible workspace"""
        workspaces = Workspaces()
        ws1 = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)
        ws2 = NiryoWorkspace("gazebo_1", mock_environment, config=workspace_config)

        workspaces.append_workspace(ws1)
        workspaces.append_workspace(ws2)

        # Use observation pose of first workspace
        obs_pose = ws1.observation_pose()
        visible = workspaces.get_visible_workspace(obs_pose)

        assert visible == ws1

    def test_get_visible_workspace_none(self, mock_environment, workspace_config):
        """Test when no workspace is visible"""
        workspaces = Workspaces()
        ws = NiryoWorkspace("niryo_ws", mock_environment, config=workspace_config)

        workspaces.append_workspace(ws)

        # Use a pose that's not the observation pose
        random_pose = PoseObjectPNP(10.0, 10.0, 10.0, 0.0, 0.0, 0.0)
        visible = workspaces.get_visible_workspace(random_pose)

        assert visible is None

    def test_verbose_property(self, mock_environment):
        """Test verbose property"""
        workspaces = Workspaces(verbose=True)
        assert workspaces.verbose() is True


class TestNiryoWorkspaces:
    """Test suite for NiryoWorkspaces class"""

    def test_initialization_real_robot(self, mock_environment):
        """Test initialization with real robot"""
        mock_environment.use_simulation.return_value = False

        workspaces = NiryoWorkspaces(mock_environment)

        # In config/niryo_config.yaml there are 4 real workspaces
        assert len(workspaces) == 4
        assert workspaces[0].id() == "niryo_ws"

    def test_initialization_simulation(self, mock_environment):
        """Test initialization with simulation"""
        mock_environment.use_simulation.return_value = True

        workspaces = NiryoWorkspaces(mock_environment)

        # In config/niryo_config.yaml there are 3 simulation workspaces
        assert len(workspaces) == 3
        assert workspaces[0].id() == "gazebo_1"
        assert workspaces[1].id() == "gazebo_2"

    def test_inherits_from_workspaces(self, mock_environment):
        """Test that NiryoWorkspaces inherits from Workspaces"""
        workspaces = NiryoWorkspaces(mock_environment)

        assert isinstance(workspaces, Workspaces)

    def test_can_add_more_workspaces(self, mock_environment, workspace_config):
        """Test that additional workspaces can be added"""
        workspaces = NiryoWorkspaces(mock_environment)

        # Start with 4 workspaces (real robot from config)
        initial_count = len(workspaces)
        assert initial_count == 4

        # Add another workspace
        ws = NiryoWorkspace("custom_ws", mock_environment, config=workspace_config)
        workspaces.append_workspace(ws)

        # Should now have 5 workspaces
        assert len(workspaces) == 5

    def test_can_add_more_workspaces_simulation(self, mock_environment, workspace_config):
        """Test that additional workspaces can be added in simulation"""
        mock_environment.use_simulation.return_value = True
        workspaces = NiryoWorkspaces(mock_environment)

        # Start with 3 workspaces (simulation from config)
        initial_count = len(workspaces)
        assert initial_count == 3

        # Add another workspace
        ws = NiryoWorkspace("custom_ws", mock_environment, config=workspace_config)
        workspaces.append_workspace(ws)

        # Should now have 4 workspaces
        assert len(workspaces) == 4

    def test_get_workspace_left_simulation(self, mock_environment):
        """Test getting left workspace in simulation"""
        mock_environment.use_simulation.return_value = True
        workspaces = NiryoWorkspaces(mock_environment)

        left_ws = workspaces.get_workspace_left()
        assert left_ws is not None
        assert left_ws.id() == "gazebo_1"

    def test_get_workspace_right_simulation(self, mock_environment):
        """Test getting right workspace in simulation"""
        mock_environment.use_simulation.return_value = True
        workspaces = NiryoWorkspaces(mock_environment)

        right_ws = workspaces.get_workspace_right()
        assert right_ws is not None
        assert right_ws.id() == "gazebo_2"

    def test_get_workspace_left_real_robot(self, mock_environment):
        """Test getting left workspace on real robot"""
        mock_environment.use_simulation.return_value = False
        workspaces = NiryoWorkspaces(mock_environment)

        left_ws = workspaces.get_workspace_left()
        assert left_ws is not None
        assert left_ws.id() == "niryo_ws"

    def test_get_workspace_right_real_robot(self, mock_environment):
        """Test getting right workspace on real robot"""
        mock_environment.use_simulation.return_value = False
        workspaces = NiryoWorkspaces(mock_environment)

        right_ws = workspaces.get_workspace_right()
        assert right_ws is not None
        assert right_ws.id() == "niryo_ws2"
