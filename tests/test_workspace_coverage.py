"""
Additional tests for workspace classes to increase coverage
"""

import os
from unittest.mock import Mock

import pytest
import yaml

from robot_workspace.config import ConfigManager, PoseConfig, WorkspaceConfig
from robot_workspace.objects.pose_object import PoseObjectPNP
from robot_workspace.workspaces.niryo_workspace import NiryoWorkspace
from robot_workspace.workspaces.niryo_workspaces import NiryoWorkspaces
from robot_workspace.workspaces.widowx_workspace import WidowXWorkspace


@pytest.fixture
def mock_environment():
    """Create a mock environment"""
    env = Mock()
    env.use_simulation.return_value = False
    env.verbose.return_value = False

    def mock_transform(ws_id, u_rel, v_rel, yaw=0.0):
        x = 0.4 - u_rel * 0.3
        y = 0.15 - v_rel * 0.3
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    env.get_robot_target_pose_from_rel = mock_transform
    return env

@pytest.fixture
def niryo_ws_config():
    return WorkspaceConfig(
        id="niryo_ws",
        observation_pose=PoseConfig(x=0.173, y=-0.002, z=0.277, roll=-3.042, pitch=1.327, yaw=-3.027),
        image_shape=(640, 480, 3)
    )

@pytest.fixture
def widowx_ws_config():
    return WorkspaceConfig(
        id="widowx_ws",
        observation_pose=PoseConfig(x=0.3, y=0.0, z=0.25, roll=0.0, pitch=0.0, yaw=0.0),
        image_shape=(1920, 1080, 3)
    )


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


class TestWidowXWorkspaceEdgeCases:
    """Test WidowX workspace edge cases for coverage"""

    def test_transform_without_environment_fallback(self, mock_environment, widowx_ws_config):
        """Test transform falls back when environment doesn't have method"""
        workspace = WidowXWorkspace("widowx_ws", mock_environment, config=widowx_ws_config)

        # Remove the environment mock
        workspace._environment = None # type: ignore

        # Should use fallback interpolation
        pose = workspace.transform_camera2world_coords("widowx_ws", 0.5, 0.5, 0.0)

        assert pose is not None
        assert isinstance(pose, PoseObjectPNP)

    def test_transform_with_corners_already_set(self, mock_environment, widowx_ws_config):
        """Test transform when corners are already initialized"""
        workspace = WidowXWorkspace("widowx_ws", mock_environment, config=widowx_ws_config)

        # Corners should be set during initialization
        assert workspace.xy_ul_wc() is not None
        assert workspace.xy_lr_wc() is not None

        # Transform should work using corners
        pose = workspace.transform_camera2world_coords("widowx_ws", 0.5, 0.5, 0.0)
        assert pose is not None


class TestWorkspaceVerboseOutput:
    """Test verbose output in workspace operations"""

    def test_niryo_workspace_verbose_initialization(self, mock_environment, niryo_ws_config):
        """Test Niryo workspace with verbose initialization"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, verbose=True, config=niryo_ws_config)

        assert workspace.verbose() is True
        # Should have logged during initialization

    def test_niryo_workspace_verbose_transform(self, mock_environment, niryo_ws_config):
        """Test verbose output during transformation"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, verbose=True, config=niryo_ws_config)

        # This should log debug output
        pose = workspace.transform_camera2world_coords("niryo_ws", 0.5, 0.5, 0.0)

        assert pose is not None

    def test_widowx_workspace_verbose(self, mock_environment, widowx_ws_config):
        """Test WidowX workspace verbose output"""
        workspace = WidowXWorkspace("widowx_ws", mock_environment, verbose=True, config=widowx_ws_config)

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
            with pytest.raises(yaml.YAMLError):  # yaml.YAMLError or similar
                manager.load_from_yaml(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
