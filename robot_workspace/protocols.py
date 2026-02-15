"""
Protocols defining interfaces for core components to avoid circular dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .objects.pose_object import PoseObjectPNP


class EnvironmentProtocol(Protocol):
    """Interface for robot environment implementations."""

    def use_simulation(self) -> bool:
        """Returns whether the environment is in simulation mode."""
        ...

    def verbose(self) -> bool:
        """Returns whether verbose logging is enabled."""
        ...

    def get_robot_target_pose_from_rel(
        self, workspace_id: str, u_rel: float, v_rel: float, yaw: float = 0.0
    ) -> PoseObjectPNP:
        """
        Calculates the target pose from relative coordinates.

        Args:
            workspace_id (str): Workspace identifier.
            u_rel (float): Horizontal relative coordinate.
            v_rel (float): Vertical relative coordinate.
            yaw (float): Target yaw orientation.

        Returns:
            PoseObjectPNP: Calculated world pose.
        """
        ...


class WorkspaceProtocol(Protocol):
    """Interface for workspace implementations."""

    def id(self) -> str:
        """Returns the workspace ID."""
        ...

    def img_shape(self) -> tuple[int, int, int]:
        """Returns the image shape of the workspace."""
        ...

    def transform_camera2world_coords(
        self, workspace_id: str, u_rel: float, v_rel: float, yaw: float = 0.0
    ) -> PoseObjectPNP:
        """
        Transforms camera relative coordinates to world coordinates.

        Args:
            workspace_id (str): Workspace identifier.
            u_rel (float): Horizontal relative coordinate.
            v_rel (float): Vertical relative coordinate.
            yaw (float): Orientation at the point.

        Returns:
            PoseObjectPNP: Calculated world pose.
        """
        ...

    def xy_ul_wc(self) -> PoseObjectPNP:
        """Returns the upper-left corner in world coordinates."""
        ...

    def xy_lr_wc(self) -> PoseObjectPNP:
        """Returns the lower-right corner in world coordinates."""
        ...
