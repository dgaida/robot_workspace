"""
Additional unit tests for ConfigManager to increase coverage.
"""

from robot_workspace.config import ConfigManager


def test_list_workspace_ids_none():
    """Test list_workspace_ids with robot_name=None."""
    config_mgr = ConfigManager()
    # Mock some workspace configs
    config_mgr._workspace_configs = {"ws1": None, "ws2": None}

    ids = config_mgr.list_workspace_ids(robot_name=None)
    assert "ws1" in ids
    assert "ws2" in ids
    assert len(ids) == 2


def test_list_workspace_ids_with_robot():
    """Test list_workspace_ids with a specific robot."""
    config_mgr = ConfigManager()

    # Use the correct path to config file
    config_mgr.load_from_yaml("robot_workspace/config/niryo_config.yaml")

    ids = config_mgr.list_workspace_ids(robot_name="niryo")
    assert len(ids) > 0
    assert "niryo_ws" in ids


def test_list_workspace_ids_non_existent_robot():
    """Test list_workspace_ids with a non-existent robot."""
    config_mgr = ConfigManager()
    ids = config_mgr.list_workspace_ids(robot_name="non_existent")
    assert ids == []
