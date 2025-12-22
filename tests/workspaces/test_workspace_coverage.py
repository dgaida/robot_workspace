"""
Additional tests for workspace classes to increase coverage
Add these tests to existing test files or create new ones
"""

import os
import pytest
from unittest.mock import Mock
from robot_workspace.workspaces.niryo_workspace import NiryoWorkspace
from robot_workspace.workspaces.niryo_workspaces import NiryoWorkspaces
from robot_workspace.workspaces.widowx_workspace import WidowXWorkspace
from robot_workspace.workspaces.widowx_workspaces import WidowXWorkspaces
from robot_workspace.objects.pose_object import PoseObjectPNP
from robot_workspace.config import ConfigManager


@pytest.fixture
def mock_environment():
    """Create a mock environment"""
    env = Mock()
    env.use_simulation.return_value = False
    env.verbose.return_value = False

    def mock_transform(ws_id, u_rel, v_rel, yaw):
        x = 0.4 - u_rel * 0.3
        y = 0.15 - v_rel * 0.3
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    env.get_robot_target_pose_from_rel = mock_transform
    return env


class TestNiryoWorkspaceFromConfig:
    """Test NiryoWorkspace.from_config class method"""

    def test_from_config_basic(self, mock_environment):
        """Test creating workspace from config"""
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Config file not found")

        config_mgr = ConfigManager()
        config_mgr.load_from_yaml(config_path)

        ws_config = config_mgr.get_workspace_config("niryo_ws")
        workspace = NiryoWorkspace.from_config(ws_config, mock_environment)

        assert workspace.id() == "niryo_ws"
        assert workspace.observation_pose() is not None

    def test_from_config_with_verbose(self, mock_environment):
        """Test from_config with verbose enabled"""
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Config file not found")

        config_mgr = ConfigManager()
        config_mgr.load_from_yaml(config_path)

        ws_config = config_mgr.get_workspace_config("niryo_ws")
        workspace = NiryoWorkspace.from_config(ws_config, mock_environment, verbose=True)

        assert workspace.verbose() is True

    def test_from_config_overrides_observation_pose(self, mock_environment):
        """Test that from_config overrides observation pose"""
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Config file not found")

        config_mgr = ConfigManager()
        config_mgr.load_from_yaml(config_path)

        ws_config = config_mgr.get_workspace_config("niryo_ws")
        workspace = NiryoWorkspace.from_config(ws_config, mock_environment)

        obs_pose = workspace.observation_pose()
        assert obs_pose.x == ws_config.observation_pose.x
        assert obs_pose.y == ws_config.observation_pose.y

    def test_from_config_overrides_image_shape(self, mock_environment):
        """Test that from_config overrides image shape"""
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Config file not found")

        config_mgr = ConfigManager()
        config_mgr.load_from_yaml(config_path)

        ws_config = config_mgr.get_workspace_config("niryo_ws")
        workspace = NiryoWorkspace.from_config(ws_config, mock_environment)

        assert workspace.img_shape() == ws_config.image_shape


class TestNiryoWorkspacesWithConfig:
    """Test NiryoWorkspaces initialization with config file"""

    def test_init_from_config_file(self, mock_environment):
        """Test initialization from config file"""
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Config file not found")

        workspaces = NiryoWorkspaces(mock_environment, verbose=False, config_path=config_path)

        assert len(workspaces) > 0

    def test_init_from_config_file_verbose(self, mock_environment):
        """Test initialization from config with verbose"""
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Config file not found")

        workspaces = NiryoWorkspaces(mock_environment, verbose=True, config_path=config_path)

        assert len(workspaces) > 0

    def test_init_from_config_simulation(self, mock_environment):
        """Test initialization from config in simulation mode"""
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Config file not found")

        mock_environment.use_simulation.return_value = True

        workspaces = NiryoWorkspaces(mock_environment, config_path=config_path)

        # Should load simulation workspaces
        workspace_ids = workspaces.get_workspace_ids()
        assert any("gazebo" in ws_id for ws_id in workspace_ids)

    def test_get_workspace_left_id_none(self, mock_environment):
        """Test get_workspace_left_id when right workspace doesn't exist"""
        mock_environment.use_simulation.return_value = False

        workspaces = NiryoWorkspaces(mock_environment)

        # With single workspace setup, right should be None
        right_id = workspaces.get_workspace_right_id()
        assert right_id is None

    def test_get_workspace_right_none(self, mock_environment):
        """Test get_workspace_right when it doesn't exist"""
        mock_environment.use_simulation.return_value = False

        workspaces = NiryoWorkspaces(mock_environment)

        right_ws = workspaces.get_workspace_right()
        assert right_ws is None


class TestWidowXWorkspaceEdgeCases:
    """Test WidowX workspace edge cases for coverage"""

    def test_transform_without_environment_fallback(self, mock_environment):
        """Test transform falls back when environment doesn't have method"""
        workspace = WidowXWorkspace("widowx_ws", mock_environment)

        # Remove the environment mock
        workspace._environment = None

        # Should use fallback interpolation
        pose = workspace.transform_camera2world_coords("widowx_ws", 0.5, 0.5, 0.0)

        assert pose is not None
        assert isinstance(pose, PoseObjectPNP)

    def test_transform_with_corners_already_set(self, mock_environment):
        """Test transform when corners are already initialized"""
        workspace = WidowXWorkspace("widowx_ws", mock_environment)

        # Corners should be set during initialization
        assert workspace._xy_ul_wc is not None
        assert workspace._xy_lr_wc is not None

        # Transform should work using corners
        pose = workspace.transform_camera2world_coords("widowx_ws", 0.5, 0.5, 0.0)
        assert pose is not None

    def test_all_observation_poses_covered(self, mock_environment):
        """Test all observation pose cases"""
        workspace_ids = [
            "widowx_ws",
            "widowx_ws_main",
            "widowx_ws_left",
            "widowx_ws_right",
            "gazebo_widowx_1",
            "gazebo_widowx_2",
            "widowx_ws_extended",
            "unknown_ws",
        ]

        for ws_id in workspace_ids:
            workspace = WidowXWorkspace(ws_id, mock_environment)
            pose = workspace.observation_pose()

            if ws_id == "unknown_ws":
                assert pose is None
            else:
                assert pose is not None

    def test_widowx_workspaces_left_and_right_getters(self, mock_environment):
        """Test WidowX workspace left/right ID getters"""
        mock_environment.use_simulation.return_value = True

        workspaces = WidowXWorkspaces(mock_environment)

        # In simulation, first workspace is left
        left_id = workspaces.get_workspace_left_id()
        assert left_id == "gazebo_widowx_1"

        # Second workspace is right
        right_id = workspaces.get_workspace_right_id()
        assert right_id == "gazebo_widowx_2"


class TestNiryoWorkspaceAllObservationPoses:
    """Test all Niryo observation pose cases"""

    def test_niryo_ws2_observation_pose(self, mock_environment):
        """Test niryo_ws2 observation pose"""
        workspace = NiryoWorkspace("niryo_ws2", mock_environment)
        pose = workspace.observation_pose()

        assert pose is not None
        assert pose.x == pytest.approx(0.173, abs=0.01)

    def test_niryo_ws_left_observation_pose(self, mock_environment):
        """Test niryo_ws_left observation pose"""
        workspace = NiryoWorkspace("niryo_ws_left", mock_environment)
        pose = workspace.observation_pose()

        assert pose is not None
        assert pose.y == 0.10  # Shifted left

    def test_niryo_ws_right_observation_pose(self, mock_environment):
        """Test niryo_ws_right observation pose"""
        workspace = NiryoWorkspace("niryo_ws_right", mock_environment)
        pose = workspace.observation_pose()

        assert pose is not None
        assert pose.y == -0.10  # Shifted right

    def test_gazebo_2_observation_pose(self, mock_environment):
        """Test gazebo_2 observation pose"""
        workspace = NiryoWorkspace("gazebo_2", mock_environment)
        pose = workspace.observation_pose()

        assert pose is not None
        assert pose.y == 0.15  # Second workspace

    def test_gazebo_1_chocolate_bars_observation_pose(self, mock_environment):
        """Test gazebo_1_chocolate_bars observation pose"""
        workspace = NiryoWorkspace("gazebo_1_chocolate_bars", mock_environment)
        pose = workspace.observation_pose()

        assert pose is not None
        assert pose.roll == 0.0
        assert pose.yaw == 0.0


class TestWorkspaceVerboseOutput:
    """Test verbose output in workspace operations"""

    def test_niryo_workspace_verbose_initialization(self, mock_environment):
        """Test Niryo workspace with verbose initialization"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, verbose=True)

        assert workspace.verbose() is True
        # Should have logged during initialization

    def test_niryo_workspace_verbose_transform(self, mock_environment):
        """Test verbose output during transformation"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, verbose=True)

        # This should log debug output
        pose = workspace.transform_camera2world_coords("niryo_ws", 0.5, 0.5, 0.0)

        assert pose is not None

    def test_widowx_workspace_verbose(self, mock_environment):
        """Test WidowX workspace verbose output"""
        workspace = WidowXWorkspace("widowx_ws", mock_environment, verbose=True)

        assert workspace.verbose() is True

        # Transform with verbose
        pose = workspace.transform_camera2world_coords("widowx_ws", 0.5, 0.5, 0.0)
        assert pose is not None


class TestConfigManagerEdgeCases:
    """Additional config manager tests for coverage"""

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML file"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:\n  - broken\n  malformed")
            temp_path = f.name

        try:
            manager = ConfigManager()
            # Should raise YAML error
            with pytest.raises(Exception):  # yaml.YAMLError or similar
                manager.load_from_yaml(temp_path)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
