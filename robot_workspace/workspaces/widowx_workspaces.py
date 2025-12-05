# class defining a list of WidowXWorkspace class
# final, apart from that more workspaces can be added
# Documentation and type definitions are almost final (chatgpt might be able to improve it).

from .workspaces import Workspaces
from .widowx_workspace import WidowXWorkspace
from typing import TYPE_CHECKING, Optional
import logging

if TYPE_CHECKING:
    from .workspace import Workspace

"""
TODO: APPLY SAME CHANGES AS niryo_workspaces.py:
1. Add 'config_path' parameter to __init__
2. Add _init_default() method
3. Add _init_from_config() method
"""


class WidowXWorkspaces(Workspaces):
    """
    Collection of WidowXWorkspace instances supporting multiple workspaces.

    The WidowX 250 6DOF robot typically uses a third-person camera setup
    (e.g., Intel RealSense) to observe the workspace, unlike the Niryo's
    gripper-mounted camera.
    """

    def __init__(self, environment, verbose: bool = False):
        """
        Adds list of WidowXWorkspace to the list of Workspaces.

        Args:
            environment: Environment object
            verbose: Enable verbose output
        """
        super().__init__(verbose)
        self._logger = logging.getLogger("robot_workspace")

        if not environment.use_simulation():
            # Real robot - can define multiple workspaces
            # WidowX typically has a single main workspace in front of the robot
            workspace_ids = ["widowx_ws_main"]

            # Optional: Add additional workspaces for multi-workspace setups
            # workspace_ids = ["widowx_ws_left", "widowx_ws_right"]
        else:
            # Simulation - can also have multiple workspaces
            workspace_ids = ["gazebo_widowx_1", "gazebo_widowx_2"]

        # Add all defined workspaces
        for workspace_id in workspace_ids:
            workspace = WidowXWorkspace(workspace_id, environment, verbose)
            super().append_workspace(workspace)

        if verbose:
            self._logger.info(f"Initialized {len(self)} WidowX workspaces: {self.get_workspace_ids()}")

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    def get_workspace_main(self) -> "Workspace":
        """Get the main workspace (index 0)."""
        return self.get_workspace(0)

    def get_workspace_left(self) -> Optional["Workspace"]:
        """
        Get the left workspace (index 0 in multi-workspace setup).

        Returns:
            Workspace or None if not available
        """
        if len(self) > 0 and self.get_workspace_id(0) in ["widowx_ws_left", "gazebo_widowx_1"]:
            return self.get_workspace(0)
        return None

    def get_workspace_right(self) -> Optional["Workspace"]:
        """
        Get the right workspace (index 1 in multi-workspace setup).

        Returns:
            Workspace or None if not available
        """
        if len(self) > 1:
            return self.get_workspace(1)
        return None

    def get_workspace_main_id(self) -> str:
        """Get the main workspace ID."""
        return self.get_workspace_id(0)

    def get_workspace_left_id(self) -> Optional[str]:
        """
        Get the left workspace ID.

        Returns:
            Workspace ID or None if not available
        """
        if len(self) > 0 and self.get_workspace_id(0) in ["widowx_ws_left", "gazebo_widowx_1"]:
            return self.get_workspace_id(0)
        return None

    def get_workspace_right_id(self) -> Optional[str]:
        """
        Get the right workspace ID.

        Returns:
            Workspace ID or None if not available
        """
        if len(self) > 1:
            return self.get_workspace_id(1)
        return None

    # *** PUBLIC methods ***

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    # *** PRIVATE variables ***
    _logger = None
