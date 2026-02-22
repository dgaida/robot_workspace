"""
Additional unit tests for workspaces to increase coverage.
"""

from unittest.mock import Mock, patch

import pytest

from robot_workspace.config import PoseConfig, WorkspaceConfig
from robot_workspace.objects.pose_object import PoseObjectPNP
from robot_workspace.workspaces.niryo_workspace import NiryoWorkspace
from robot_workspace.workspaces.niryo_workspaces import NiryoWorkspaces
from robot_workspace.workspaces.widowx_workspace import WidowXWorkspace
from robot_workspace.workspaces.widowx_workspaces import WidowXWorkspaces
from robot_workspace.workspaces.workspace import Workspace
from robot_workspace.workspaces.workspaces import Workspaces


class ConcreteWorkspace(Workspace):
    """Concrete implementation of Workspace for testing abstract class methods."""

    def __init__(self, workspace_id, verbose=False):
        # We need to set these BEFORE super().__init__ calls methods that use them,
        # but Workspace.__init__ calls them directly.
        # So we must override __init__ or ensure the methods handle None.
        self._id = workspace_id
        self._verbose = verbose
        self._logger = Mock()
        self._observation_pose = PoseObjectPNP(0, 0, 0, 0, 0, 0)
        self._xy_ul_wc = PoseObjectPNP(1, 1, 0, 0, 0, 0)
        self._xy_ll_wc = PoseObjectPNP(1, 0, 0, 0, 0, 0)
        self._xy_ur_wc = PoseObjectPNP(0, 1, 0, 0, 0, 0)
        self._xy_lr_wc = PoseObjectPNP(0, 0, 0, 0, 0, 0)
        super().__init__(workspace_id, verbose)

    def _set_4corners_of_workspace(self) -> None:
        pass

    def _set_observation_pose(self) -> None:
        pass

    def transform_camera2world_coords(self, workspace_id, u_rel, v_rel, yaw=0.0):
        return PoseObjectPNP(0, 0, 0, 0, 0, 0)


@pytest.fixture
def mock_env():
    env = Mock()
    env.use_simulation.return_value = False
    env.verbose.return_value = False
    env.get_robot_target_pose_from_rel.return_value = PoseObjectPNP(0.1, 0.2, 0.3, 0, 0, 0)
    return env


@pytest.fixture
def widowx_config():
    return WorkspaceConfig(id="test_ws", observation_pose=PoseConfig(x=0.3, y=0, z=0.25), image_shape=(640, 480, 3))


def test_niryo_workspaces_edge_cases(mock_env):
    """Test NiryoWorkspaces missing lines."""
    workspaces = NiryoWorkspaces(mock_env, verbose=True)

    # Test get_workspace_left_id
    assert workspaces.get_workspace_left_id() == "niryo_ws"

    # Test get_workspace_right_id (exists)
    assert workspaces.get_workspace_right_id() == "niryo_ws2"

    # Test with only one workspace
    with patch.object(Workspaces, "__len__", return_value=1):
        assert workspaces.get_workspace_right() is None
        assert workspaces.get_workspace_right_id() is None


def test_niryo_workspace_missing_obs_pose(mock_env):
    """Test NiryoWorkspace with missing observation pose in config."""
    config = WorkspaceConfig(id="test_ws", observation_pose=None)
    ws = NiryoWorkspace.__new__(NiryoWorkspace)
    ws._id = "test_ws"
    ws._config = config

    with pytest.raises(ValueError, match="No observation pose defined in config"):
        ws._set_observation_pose()


def test_widowx_workspace_fallback(mock_env, widowx_config):
    """Test WidowXWorkspace fallback transformation."""
    ws = WidowXWorkspace("test_ws", mock_env, config=widowx_config)

    # Force fallback by removing environment or setting it to None
    ws._environment = None

    # Fallback without corners
    ws._xy_ul_wc = None
    ws._xy_lr_wc = None
    pose = ws.transform_camera2world_coords("test_ws", 0.5, 0.5)
    assert pose.x == 0.5 - 0.5 * 0.4

    # Fallback with corners
    ws._xy_ul_wc = PoseObjectPNP(1.0, 1.0, 0, 0, 0, 0)
    ws._xy_lr_wc = PoseObjectPNP(0.0, 0.0, 0, 0, 0, 0)
    pose = ws.transform_camera2world_coords("test_ws", 0.5, 0.5)
    assert pose.x == 0.5
    assert pose.y == 0.5

    # Test verbose fallback
    ws._verbose = True
    import logging

    ws._logger = logging.getLogger("robot_workspace")
    with patch.object(ws._logger, "debug") as mock_debug:
        ws.transform_camera2world_coords("test_ws", 0.5, 0.5)
        # Should be called for input and output (fallback)
        assert mock_debug.call_count >= 2


def test_widowx_workspace_missing_obs_pose(mock_env):
    """Test WidowXWorkspace with missing observation pose in config."""
    config = WorkspaceConfig(id="test_ws", observation_pose=None)
    # We can't use constructor because it calls _set_observation_pose
    ws = WidowXWorkspace.__new__(WidowXWorkspace)
    ws._id = "test_ws"
    ws._config = config

    with pytest.raises(ValueError, match="No observation pose defined in config"):
        ws._set_observation_pose()


def test_widowx_workspaces_more_coverage(mock_env):
    """Test WidowXWorkspaces missing lines."""
    workspaces = WidowXWorkspaces(mock_env)

    # Test get_workspace_left when it matches ID
    mock_ws = Mock()
    mock_ws.id.return_value = "widowx_ws_left"
    # We need to ensure get_workspace_id(0) returns one of the strings
    with patch.object(workspaces, "get_workspace_id", return_value="widowx_ws_left"), patch.object(
        workspaces, "get_workspace", return_value=mock_ws
    ):
        assert workspaces.get_workspace_left() == mock_ws
        assert workspaces.get_workspace_left_id() == "widowx_ws_left"

    # Test get_workspace_right (None case)
    with patch.object(Workspaces, "__len__", return_value=1):
        assert workspaces.get_workspace_right() is None
        assert workspaces.get_workspace_right_id() is None


def test_workspace_base_missing_obs_pose():
    """Test Workspace base class with None observation pose."""
    ws = ConcreteWorkspace("test_ws")
    ws._observation_pose = None

    assert ws.is_visible(PoseObjectPNP(0, 0, 0, 0, 0, 0)) is False

    # Test verbose is_visible
    ws._verbose = True
    import logging

    ws._logger = logging.getLogger("robot_workspace")
    with patch.object(ws._logger, "debug") as mock_debug:
        ws.is_visible(PoseObjectPNP(0, 0, 0, 0, 0, 0))
        mock_debug.assert_called()


def test_workspace_calc_center_error():
    """Test _calc_center_of_workspace error case."""
    ws = ConcreteWorkspace("test_ws")
    ws._xy_ll_wc = None  # Trigger error

    with pytest.raises(ValueError, match="Workspace corners must be set"):
        ws._calc_center_of_workspace()


def test_workspaces_get_width_height_not_found():
    """Test Workspaces.get_width_height_m for non-existent workspace."""
    workspaces = Workspaces()
    assert workspaces.get_width_height_m("non_existent") == (0.0, 0.0)
