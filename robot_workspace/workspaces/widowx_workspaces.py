"""Collection of WidowX workspaces."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .widowx_workspace import WidowXWorkspace

# class defining a list of WidowXWorkspace class
# final, apart from that more workspaces can be added
# Documentation and type definitions are almost final (chatgpt might be able to improve it).
from .workspaces import Workspaces

if TYPE_CHECKING:
    from ..protocols import EnvironmentProtocol
    from .workspace import Workspace


class WidowXWorkspaces(Workspaces):
    """
    Collection of WidowXWorkspace instances supporting multiple workspaces.

    The WidowX 250 6DOF robot typically uses a third-person camera setup
    (e.g., Intel RealSense) to observe the workspace, unlike the Niryo's
    gripper-mounted camera.
    """

    def __init__(
        self, environment: EnvironmentProtocol, verbose: bool = False, config_path: str = "config/widowx_config.yaml"
    ) -> None:
        """
        Initialize WidowXWorkspaces collection from configuration.

        Args:
            environment: Environment object
            verbose: Enable verbose output
            config_path: Path to YAML configuration file. Defaults to 'config/widowx_config.yaml'.
        """
        super().__init__(verbose)
        self._logger = logging.getLogger("robot_workspace")

        self._init_from_config(environment, config_path, verbose)

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    def get_workspace_main(self) -> Workspace:
        """Get the main workspace (index 0)."""
        return self.get_workspace(0)

    def get_workspace_left(self) -> Workspace | None:
        """
        Get the left workspace (index 0 in multi-workspace setup).

        Returns:
            Workspace or None if not available
        """
        if len(self) > 0 and self.get_workspace_id(0) in ["widowx_ws_left", "gazebo_widowx_1"]:
            return self.get_workspace(0)
        return None

    def get_workspace_right(self) -> Workspace | None:
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

    def get_workspace_left_id(self) -> str | None:
        """
        Get the left workspace ID.

        Returns:
            Workspace ID or None if not available
        """
        if len(self) > 0 and self.get_workspace_id(0) in ["widowx_ws_left", "gazebo_widowx_1"]:
            return self.get_workspace_id(0)
        return None

    def get_workspace_right_id(self) -> str | None:
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

    def _init_from_config(self, environment: EnvironmentProtocol, config_path: str, verbose: bool) -> None:
        """Initialize workspaces from configuration file."""
        from ..config import ConfigManager

        config_mgr = ConfigManager()
        config_mgr.load_from_yaml(config_path)

        # Get workspace configs based on simulation mode
        workspace_configs = config_mgr.get_workspace_configs("widowx", simulation=environment.use_simulation())

        for ws_config in workspace_configs:
            workspace = WidowXWorkspace.from_config(ws_config, environment, verbose)
            super().append_workspace(workspace)

        if verbose:
            self._logger.info(f"Initialized {len(self)} WidowX workspaces from config: {self.get_workspace_ids()}")

    # *** PUBLIC properties ***

    # *** PRIVATE variables ***
    _logger: logging.Logger = None
