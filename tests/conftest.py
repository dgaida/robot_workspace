"""
Pytest configuration and shared fixtures for robot_environment tests
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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
