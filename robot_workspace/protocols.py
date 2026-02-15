from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .objects.pose_object import PoseObjectPNP


class EnvironmentProtocol(Protocol):
    """Interface for robot environment implementations."""

    def use_simulation(self) -> bool: ...
    def verbose(self) -> bool: ...
    def get_robot_target_pose_from_rel(
        self, workspace_id: str, u_rel: float, v_rel: float, yaw: float = 0.0
    ) -> PoseObjectPNP: ...


class WorkspaceProtocol(Protocol):
    """Interface for workspace implementations."""

    def id(self) -> str: ...
    def img_shape(self) -> tuple[int, int, int]: ...
    def transform_camera2world_coords(
        self, workspace_id: str, u_rel: float, v_rel: float, yaw: float = 0.0
    ) -> PoseObjectPNP: ...
    def xy_ul_wc(self) -> PoseObjectPNP: ...
    def xy_lr_wc(self) -> PoseObjectPNP: ...
