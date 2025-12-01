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

    def __init__(self, environment, verbose: bool = False):
        """
        Adds list of NiryoWorkspace to the list of Workspaces.

        Args:
            environment: Environment object
            verbose: Enable verbose output
        """
        super().__init__(verbose)

        if not environment.use_simulation():
            # Real robot - can define multiple workspaces
            workspace_ids = ["niryo_ws2", "niryo_ws_right"]  # Two workspaces
        else:
            # Simulation - can also have multiple workspaces
            workspace_ids = ["gazebo_1", "gazebo_2"]

        # Add all defined workspaces
        for workspace_id in workspace_ids:
            workspace = NiryoWorkspace(workspace_id, environment, verbose)
            super().append_workspace(workspace)

        if verbose:
            print(f"Initialized {len(self)} workspaces: {self.get_workspace_ids()}")

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

    # *** PUBLIC properties ***

    # *** PRIVATE variables ***
