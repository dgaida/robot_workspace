"""
Pytest configuration and shared fixtures for robot_environment tests
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from robot_workspace.objects.pose_object import PoseObjectPNP

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MockEnvironmentBuilder:
    """Builder for creating test environments with sensible defaults."""

    def __init__(self):
        self._simulation = False
        self._verbose = False
        self._transform_fn = self._default_niryo_transform

    def with_simulation(self, enabled: bool = True):
        self._simulation = enabled
        return self

    def with_verbose(self, enabled: bool = True):
        self._verbose = enabled
        return self

    def with_widowx_transform(self):
        self._transform_fn = self._default_widowx_transform
        return self

    def with_custom_transform(self, fn):
        self._transform_fn = fn
        return self

    def build(self) -> Mock:
        env = Mock()
        env.use_simulation.return_value = self._simulation
        env.verbose.return_value = self._verbose
        env.get_robot_target_pose_from_rel = self._transform_fn
        return env

    @staticmethod
    def _default_niryo_transform(ws_id, u_rel, v_rel, yaw=0.0):
        # Map relative coords to world coords properly for Niryo
        x = 0.4 - u_rel * 0.3
        y = 0.15 - v_rel * 0.3
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    @staticmethod
    def _default_widowx_transform(ws_id, u_rel, v_rel, yaw=0.0):
        # Map relative coords to world coords properly for WidowX
        x = 0.5 - u_rel * 0.4
        y = 0.2 - v_rel * 0.4
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)


@pytest.fixture
def mock_env_builder():
    return MockEnvironmentBuilder()


# Mark configuration for slow tests
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_robot: marks tests that require real robot hardware")
    config.addinivalue_line("markers", "requires_redis: marks tests that require Redis server")


# Skip tests based on markers
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers"""
    skip_slow = pytest.mark.skip(reason="slow test, use -m slow to run")
    skip_integration = pytest.mark.skip(reason="integration test")
    skip_robot = pytest.mark.skip(reason="requires robot hardware")
    skip_redis = pytest.mark.skip(reason="requires Redis server")

    for item in items:
        if "slow" in item.keywords and not config.getoption("-m") == "slow":
            item.add_marker(skip_slow)
        if "integration" in item.keywords:
            # Skip integration tests by default unless explicitly requested
            if not config.getoption("-m") or "integration" not in config.getoption("-m"):
                item.add_marker(skip_integration)
        if "requires_robot" in item.keywords:
            item.add_marker(skip_robot)
        if "requires_redis" in item.keywords:
            item.add_marker(skip_redis)
