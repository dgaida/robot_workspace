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
    class defining a list of Workspace class

    """

    # *** CONSTRUCTORS ***
    def __init__(self, verbose: bool = False) -> None:
        """
        Creates an empty list of Workspace class

        Args:
            verbose (bool): If True, enables verbose logging.
        """
        super().__init__()

        self._verbose = verbose

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    def get_workspace_home_id(self) -> str:
        """
        Returns the ID of the workspace at index 0.

        Returns:
            the ID of the workspace at index 0.
        """
        return self.get_workspace_id(0)

    def get_home_workspace(self) -> Workspace:
        """
        Returns the workspace at index 0.

        Returns:
            the workspace at index 0.
        """
        return self.get_workspace(0)

    def get_workspace(self, index: int) -> Workspace:
        """
        Return the workspace at the given position index in the list of workspaces.

        Args:
            index: 0-based index in the list of workspaces.

        Returns:

        """
        return self[index]

    def get_workspace_by_id(self, id: str) -> Workspace | None:
        """
        Return the Workspace object with the given id, if existent, else None is returned.

        Args:
            id: workspace ID

        Returns:
            Workspace or None, if no workspace with the given id exists.
        """
        for workspace in self:
            if workspace.id() == id:
                return workspace

        return None

    def get_workspace_ids(self) -> list[str]:
        """
        Returns a list of ids of all workspaces.

        Returns:
            list of ids of all workspaces.
        """
        ids = [workspace.id() for workspace in self]

        return ids

    def get_workspace_id(self, index: int) -> str:
        """
        Return the id of the workspace at the given position index in the list of workspaces.

        Args:
            index: 0-based index in the list of workspaces.

        Returns:
            str: id of the workspace at the given position index in the list of workspaces.
        """
        return self[index].id()

    def get_observation_pose(self, workspace_id: str) -> PoseObjectPNP | None:
        """
        Return the observation pose of the given workspace id

        Args:
            workspace_id: id of the workspace

        Returns:
            PoseObjectPNP: observation pose of the gripper where it can observe the workspace given by workspace_id
        """
        workspace = self.get_workspace_by_id(workspace_id)
        return workspace.observation_pose() if workspace else None

    def get_width_height_m(self, workspace_id: str) -> tuple[float, float]:
        """
        Return the width and height in m of the workspace with the given workspace id

        Args:
            workspace_id: id of the workspace

        Returns:
            width, height: width and height of the workspace in meters
        """
        workspace = self.get_workspace_by_id(workspace_id)
        if workspace:
            return workspace.width_m(), workspace.height_m()
        return 0.0, 0.0

    def get_img_shape(self, workspace_id: str) -> tuple[int, int, int] | None:
        """
        Return the shape of the image of the workspace with the given workspace id

        Args:
            workspace_id: id of the workspace

        Returns:
            shape of image of workspace in pixels
        """
        workspace = self.get_workspace_by_id(workspace_id)
        return workspace.img_shape() if workspace else None

    # *** PUBLIC methods ***

    # TODO: this is not yet the final implementation
    def get_visible_workspace(self, camera_pose: PoseObjectPNP) -> Workspace | None:
        for workspace in self:
            # the is_visible checks which workspace is visible.
            if workspace.is_visible(camera_pose):
                return workspace

        return None

    def append_workspace(self, workspace: Workspace) -> None:
        """
        Appends the given workspace to the list of workspaces.

        Args:
            workspace: some Workspace object
        """
        self.append(workspace)

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    # def environment(self) -> "Environment":
    #     return self._environment

    def verbose(self) -> bool:
        return self._verbose

    # *** PRIVATE variables ***

    # environment this workspace belongs to
    # _environment = None

    _verbose = False
