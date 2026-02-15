from __future__ import annotations

# class defining a list of Workspace class
# TODO: get_visible_workspace
# Documentation and type definitions are almost final (chatgpt might be able to improve it).
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..objects.pose_object import PoseObjectPNP
    from ..workspaces.workspace import Workspace


class Workspaces(list["Workspace"]):
    """
    A collection of Workspace instances.

    This class extends the standard Python list and provides helper methods
    to manage multiple robot workspaces.
    """

    # *** CONSTRUCTORS ***
    def __init__(self, verbose: bool = False) -> None:
        """
        Creates an empty list of Workspace class.

        Args:
            verbose (bool): If True, enables verbose logging.
        """
        super().__init__()

        self._verbose = verbose

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    def get_workspace_home_id(self) -> str:
        """
        Returns the ID of the home workspace (at index 0).

        Returns:
            str: ID of the home workspace.
        """
        return self.get_workspace_id(0)

    def get_home_workspace(self) -> Workspace:
        """
        Returns the home workspace instance (at index 0).

        Returns:
            Workspace: The home workspace instance.
        """
        return self.get_workspace(0)

    def get_workspace(self, index: int) -> Workspace:
        """
        Returns the workspace at the specified index.

        Args:
            index (int): 0-based index.

        Returns:
            Workspace: The workspace at the given index.
        """
        return self[index]

    def get_workspace_by_id(self, id: str) -> Workspace | None:
        """
        Finds a workspace by its unique ID.

        Args:
            id (str): Workspace ID to look for.

        Returns:
            Workspace | None: The matching workspace instance, or None if not found.
        """
        for workspace in self:
            if workspace.id() == id:
                return workspace

        return None

    def get_workspace_ids(self) -> list[str]:
        """
        Returns a list of IDs for all managed workspaces.

        Returns:
            list[str]: List of workspace IDs.
        """
        ids = [workspace.id() for workspace in self]

        return ids

    def get_workspace_id(self, index: int) -> str:
        """
        Returns the ID of the workspace at the given index.

        Args:
            index (int): 0-based index.

        Returns:
            str: ID of the workspace.
        """
        return self[index].id()

    def get_observation_pose(self, workspace_id: str) -> PoseObjectPNP | None:
        """
        Returns the observation pose for the specified workspace.

        Args:
            workspace_id (str): Workspace ID.

        Returns:
            PoseObjectPNP | None: The observation pose, or None if workspace not found.
        """
        workspace = self.get_workspace_by_id(workspace_id)
        return workspace.observation_pose() if workspace else None

    def get_width_height_m(self, workspace_id: str) -> tuple[float, float]:
        """
        Returns the physical dimensions of the specified workspace.

        Args:
            workspace_id (str): Workspace ID.

        Returns:
            tuple[float, float]: (width, height) in meters.
        """
        workspace = self.get_workspace_by_id(workspace_id)
        if workspace:
            return workspace.width_m(), workspace.height_m()
        return 0.0, 0.0

    def get_img_shape(self, workspace_id: str) -> tuple[int, int, int] | None:
        """
        Returns the image shape for the specified workspace.

        Args:
            workspace_id (str): Workspace ID.

        Returns:
            tuple[int, int, int] | None: (height, width, channels) or None.
        """
        workspace = self.get_workspace_by_id(workspace_id)
        return workspace.img_shape() if workspace else None

    # *** PUBLIC methods ***

    def get_visible_workspace(self, camera_pose: PoseObjectPNP) -> Workspace | None:
        """
        Identifies which workspace is currently visible from the given camera pose.

        Args:
            camera_pose (PoseObjectPNP): The current pose of the camera.

        Returns:
            Workspace | None: The visible workspace, or None if none match.
        """
        for workspace in self:
            # the is_visible checks which workspace is visible.
            if workspace.is_visible(camera_pose):
                return workspace

        return None

    def append_workspace(self, workspace: Workspace) -> None:
        """
        Adds a new workspace to the collection.

        Args:
            workspace (Workspace): The workspace instance to add.
        """
        self.append(workspace)

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    # def environment(self) -> "Environment":
    #     return self._environment

    def verbose(self) -> bool:
        """
        Returns whether verbose logging is enabled.

        Returns:
            bool: True if verbose, else False.
        """
        return self._verbose

    # *** PRIVATE variables ***

    # environment this workspace belongs to
    # _environment = None

    _verbose = False
