# class defining a list of NiryoWorkspace class
# final, apart from that more workspaces can be added
# Documentation and type definitions are almost final (chatgpt might be able to improve it).

from .workspaces import Workspaces
from .niryo_workspace import NiryoWorkspace
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .workspace import Workspace


class NiryoWorkspaces(Workspaces):
    """
    Collection of NiryoWorkspace instances supporting multiple workspaces.
    """

    def __init__(self, environment, verbose: bool = False, config_path: Optional[str] = None):
        """
        Initialize NiryoWorkspaces collection.

        Args:
            environment: Environment object
            verbose: Enable verbose output
            config_path: Optional path to YAML configuration file.
                        If provided, workspaces are loaded from config.
                        If None, uses default hardcoded workspaces.

        Example:
            >>> # Using default hardcoded configuration
            >>> workspaces = NiryoWorkspaces(environment)

            >>> # Using YAML configuration file
            >>> workspaces = NiryoWorkspaces(
            ...     environment,
            ...     config_path='config/niryo_config.yaml'
            ... )
        """
        super().__init__(verbose)

        if config_path:
            self._init_from_config(environment, config_path, verbose)
        else:
            self._init_default(environment, verbose)

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    def get_workspace_left(self) -> "Workspace":
        """Get the left workspace (index 0)."""
        return self.get_workspace(0)

    def get_workspace_right(self) -> Optional["Workspace"]:
        """Get the right workspace (index 1)."""
        if len(self) > 1:
            return self.get_workspace(1)
        return None

    def get_workspace_left_id(self) -> str:
        """Get the left workspace ID."""
        return self.get_workspace_id(0)

    def get_workspace_right_id(self) -> Optional[str]:
        """Get the right workspace ID."""
        if len(self) > 1:
            return self.get_workspace_id(1)
        return None

    # *** PUBLIC methods ***

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    def _init_default(self, environment, verbose: bool):
        """Initialize with default hardcoded workspaces."""
        if not environment.use_simulation():
            workspace_ids = ["niryo_ws2", "niryo_ws_right"]
        else:
            workspace_ids = ["gazebo_1", "gazebo_2"]

        for workspace_id in workspace_ids:
            workspace = NiryoWorkspace(workspace_id, environment, verbose)
            super().append_workspace(workspace)

        if verbose:
            print(f"Initialized {len(self)} workspaces: {self.get_workspace_ids()}")

    def _init_from_config(self, environment, config_path: str, verbose: bool):
        """Initialize workspaces from configuration file."""
        from ..config import ConfigManager

        config_mgr = ConfigManager()
        config_mgr.load_from_yaml(config_path)

        # Get workspace configs based on simulation mode
        workspace_configs = config_mgr.get_workspace_configs("niryo", simulation=environment.use_simulation())

        for ws_config in workspace_configs:
            workspace = NiryoWorkspace.from_config(ws_config, environment, verbose)
            super().append_workspace(workspace)

        if verbose:
            print(f"Initialized {len(self)} workspaces from config: " f"{self.get_workspace_ids()}")

    # *** PUBLIC properties ***

    # *** PRIVATE variables ***
