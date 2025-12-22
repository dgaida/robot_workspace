"""
Unit tests for config.py (Configuration management)
Create this file at: tests/test_config.py
"""

import pytest
import tempfile
import os
import yaml
from robot_workspace.config import (
    PoseConfig,
    WorkspaceConfig,
    RobotConfig,
    ConfigManager,
)


class TestPoseConfig:
    """Test suite for PoseConfig class"""

    def test_default_initialization(self):
        """Test PoseConfig with default values"""
        pose = PoseConfig()

        assert pose.x == 0.0
        assert pose.y == 0.0
        assert pose.z == 0.0
        assert pose.roll == 0.0
        assert pose.pitch == 0.0
        assert pose.yaw == 0.0

    def test_custom_initialization(self):
        """Test PoseConfig with custom values"""
        pose = PoseConfig(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)

        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.roll == 0.1
        assert pose.pitch == 0.2
        assert pose.yaw == 0.3

    def test_to_dict(self):
        """Test conversion to dictionary"""
        pose = PoseConfig(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        pose_dict = pose.to_dict()

        assert pose_dict == {"x": 1.0, "y": 2.0, "z": 3.0, "roll": 0.1, "pitch": 0.2, "yaw": 0.3}


class TestWorkspaceConfig:
    """Test suite for WorkspaceConfig class"""

    def test_from_dict_basic(self):
        """Test creating WorkspaceConfig from dictionary"""
        data = {
            "id": "test_ws",
            "observation_pose": {"x": 0.173, "y": -0.002, "z": 0.277, "roll": -3.042, "pitch": 1.327, "yaw": -3.027},
            "image_shape": [640, 480, 3],
        }

        config = WorkspaceConfig.from_dict(data)

        assert config.id == "test_ws"
        assert config.observation_pose.x == 0.173
        assert config.image_shape == (640, 480, 3)

    def test_from_dict_with_tuple_image_shape(self):
        """Test from_dict with tuple image_shape"""
        data = {
            "id": "test_ws",
            "observation_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "image_shape": (640, 480, 3),  # Already a tuple
        }

        config = WorkspaceConfig.from_dict(data)

        assert config.image_shape == (640, 480, 3)
        assert isinstance(config.image_shape, tuple)

    def test_from_dict_with_corners(self):
        """Test from_dict with corner definitions"""
        data = {
            "id": "test_ws",
            "observation_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "corners": {
                "ul": {"x": 0.4, "y": 0.15, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "lr": {"x": 0.1, "y": -0.15, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            },
        }

        config = WorkspaceConfig.from_dict(data)

        assert config.corners is not None
        assert "ul" in config.corners
        assert "lr" in config.corners
        assert config.corners["ul"].x == 0.4
        assert config.corners["lr"].x == 0.1

    def test_from_dict_with_robot_type(self):
        """Test from_dict with custom robot_type"""
        data = {
            "id": "test_ws",
            "observation_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "robot_type": "widowx",
        }

        config = WorkspaceConfig.from_dict(data)

        assert config.robot_type == "widowx"

    def test_from_dict_default_robot_type(self):
        """Test from_dict uses default robot_type"""
        data = {"id": "test_ws", "observation_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}}

        config = WorkspaceConfig.from_dict(data)

        assert config.robot_type == "niryo"  # Default


class TestRobotConfig:
    """Test suite for RobotConfig class"""

    def test_from_dict_basic(self):
        """Test creating RobotConfig from dictionary"""
        data = {
            "name": "niryo",
            "workspaces": [
                {
                    "id": "niryo_ws",
                    "observation_pose": {"x": 0.173, "y": -0.002, "z": 0.277, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                }
            ],
        }

        config = RobotConfig.from_dict(data)

        assert config.name == "niryo"
        assert len(config.workspaces) == 1
        assert config.workspaces[0].id == "niryo_ws"

    def test_from_dict_with_simulation_workspaces(self):
        """Test from_dict with simulation workspaces"""
        data = {
            "name": "niryo",
            "workspaces": [],
            "simulation_workspaces": [
                {
                    "id": "gazebo_1",
                    "observation_pose": {"x": 0.18, "y": 0.0, "z": 0.36, "roll": 0.0, "pitch": 1.5708, "yaw": 0.0},
                }
            ],
        }

        config = RobotConfig.from_dict(data)

        assert len(config.simulation_workspaces) == 1
        assert config.simulation_workspaces[0].id == "gazebo_1"

    def test_from_dict_with_default_workspace_id(self):
        """Test from_dict with default_workspace_id"""
        data = {"name": "niryo", "default_workspace_id": "niryo_ws2", "workspaces": []}

        config = RobotConfig.from_dict(data)

        assert config.default_workspace_id == "niryo_ws2"


class TestConfigManager:
    """Test suite for ConfigManager class"""

    def test_initialization(self):
        """Test ConfigManager initialization"""
        manager = ConfigManager()

        assert manager._robot_configs == {}
        assert manager._workspace_configs == {}

    def test_load_from_yaml_niryo(self):
        """Test loading Niryo configuration from YAML"""
        # Use actual config file from the project
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Niryo config file not found")

        manager = ConfigManager()
        manager.load_from_yaml(config_path)

        # Verify robot config loaded
        robot_config = manager.get_robot_config("niryo")
        assert robot_config is not None
        assert robot_config.name == "niryo"

        # Verify workspaces indexed
        workspace_ids = manager.list_workspace_ids("niryo")
        assert len(workspace_ids) > 0

    def test_load_from_yaml_widowx(self):
        """Test loading WidowX configuration from YAML"""
        config_path = "config/widowx_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("WidowX config file not found")

        manager = ConfigManager()
        manager.load_from_yaml(config_path)

        robot_config = manager.get_robot_config("widowx")
        assert robot_config is not None
        assert robot_config.name == "widowx"

    def test_load_from_yaml_file_not_found(self):
        """Test loading from non-existent file raises error"""
        manager = ConfigManager()

        with pytest.raises(FileNotFoundError):
            manager.load_from_yaml("/nonexistent/path/config.yaml")

    def test_load_from_yaml_custom_config(self):
        """Test loading custom YAML configuration"""
        # Create temporary YAML file
        config_data = {
            "robots": {
                "test_robot": {
                    "name": "test_robot",
                    "default_workspace_id": "test_ws",
                    "workspaces": [
                        {
                            "id": "test_ws",
                            "robot_type": "test",
                            "observation_pose": {"x": 0.3, "y": 0.0, "z": 0.25, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                            "image_shape": [640, 480, 3],
                        }
                    ],
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            manager = ConfigManager()
            manager.load_from_yaml(temp_path)

            robot_config = manager.get_robot_config("test_robot")
            assert robot_config is not None
            assert robot_config.name == "test_robot"
            assert robot_config.default_workspace_id == "test_ws"

            workspace_config = manager.get_workspace_config("test_ws")
            assert workspace_config is not None
            assert workspace_config.id == "test_ws"
        finally:
            os.unlink(temp_path)

    def test_get_robot_config_not_found(self):
        """Test getting non-existent robot config"""
        manager = ConfigManager()

        robot_config = manager.get_robot_config("nonexistent")

        assert robot_config is None

    def test_get_workspace_config_not_found(self):
        """Test getting non-existent workspace config"""
        manager = ConfigManager()

        workspace_config = manager.get_workspace_config("nonexistent")

        assert workspace_config is None

    def test_get_workspace_configs_not_found(self):
        """Test getting workspace configs for non-existent robot"""
        manager = ConfigManager()

        configs = manager.get_workspace_configs("nonexistent")

        assert configs == []

    def test_get_workspace_configs_simulation(self):
        """Test getting simulation workspace configs"""
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Niryo config file not found")

        manager = ConfigManager()
        manager.load_from_yaml(config_path)

        sim_configs = manager.get_workspace_configs("niryo", simulation=True)

        assert len(sim_configs) > 0
        # Simulation workspaces should have 'gazebo' in their IDs
        assert any("gazebo" in ws.id for ws in sim_configs)

    def test_get_workspace_configs_real(self):
        """Test getting real workspace configs"""
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Niryo config file not found")

        manager = ConfigManager()
        manager.load_from_yaml(config_path)

        real_configs = manager.get_workspace_configs("niryo", simulation=False)

        assert len(real_configs) > 0

    def test_list_workspace_ids_all(self):
        """Test listing all workspace IDs"""
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Niryo config file not found")

        manager = ConfigManager()
        manager.load_from_yaml(config_path)

        all_ids = manager.list_workspace_ids()

        assert len(all_ids) > 0
        assert isinstance(all_ids, list)

    def test_list_workspace_ids_for_robot(self):
        """Test listing workspace IDs for specific robot"""
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Niryo config file not found")

        manager = ConfigManager()
        manager.load_from_yaml(config_path)

        niryo_ids = manager.list_workspace_ids("niryo")

        assert len(niryo_ids) > 0
        # Should include both real and simulation workspaces

    def test_list_workspace_ids_for_nonexistent_robot(self):
        """Test listing workspace IDs for non-existent robot"""
        manager = ConfigManager()

        ids = manager.list_workspace_ids("nonexistent")

        assert ids == []


class TestConfigIntegration:
    """Integration tests for configuration system"""

    def test_complete_config_workflow(self):
        """Test complete configuration loading and usage workflow"""
        config_path = "config/niryo_config.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Niryo config file not found")

        # Load configuration
        manager = ConfigManager()
        manager.load_from_yaml(config_path)

        # Get robot configuration
        robot_config = manager.get_robot_config("niryo")
        assert robot_config is not None

        # Get default workspace
        default_ws_id = robot_config.default_workspace_id
        assert default_ws_id is not None

        # Get workspace configuration
        ws_config = manager.get_workspace_config(default_ws_id)
        assert ws_config is not None
        assert ws_config.id == default_ws_id

        # Verify observation pose
        assert ws_config.observation_pose is not None
        assert isinstance(ws_config.observation_pose, PoseConfig)

        # Verify image shape
        assert ws_config.image_shape is not None
        assert len(ws_config.image_shape) == 3

    def test_multiple_robots_config(self):
        """Test loading configuration with multiple robots"""
        config_data = {
            "robots": {
                "robot1": {
                    "name": "robot1",
                    "workspaces": [
                        {
                            "id": "ws1",
                            "observation_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                        }
                    ],
                },
                "robot2": {
                    "name": "robot2",
                    "workspaces": [
                        {
                            "id": "ws2",
                            "observation_pose": {"x": 1.0, "y": 1.0, "z": 1.0, "roll": 0.1, "pitch": 0.2, "yaw": 0.3},
                        }
                    ],
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            manager = ConfigManager()
            manager.load_from_yaml(temp_path)

            # Verify both robots loaded
            robot1 = manager.get_robot_config("robot1")
            robot2 = manager.get_robot_config("robot2")

            assert robot1 is not None
            assert robot2 is not None

            # Verify workspaces are separate
            ws1 = manager.get_workspace_config("ws1")
            ws2 = manager.get_workspace_config("ws2")

            assert ws1 is not None
            assert ws2 is not None
            assert ws1.observation_pose.x == 0.0
            assert ws2.observation_pose.x == 1.0
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
