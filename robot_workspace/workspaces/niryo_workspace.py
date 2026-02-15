from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Defines a workspace where the pick-and-place robot Niryo Ned2 can pick or place objects from.
# Not final, but works
# Documentation and type definitions are almost final (chatgpt might be able to improve it).
from ..common.logger import log_start_end_cls
from ..objects.pose_object import PoseObjectPNP
from .workspace import Workspace

if TYPE_CHECKING:
    from ..config import WorkspaceConfig
    from ..protocols import EnvironmentProtocol


class NiryoWorkspace(Workspace):
    """
    Implementation of Workspace for the Niryo Ned2 robot.

    This class provides specific coordinate transformations and poses
    for the Niryo Ned2 robotic arm and its mounted camera.
    """

    # *** CONSTRUCTORS ***
    def __init__(
        self,
        workspace_id: str,
        environment: EnvironmentProtocol,
        verbose: bool = False,
        config: WorkspaceConfig | None = None,
    ) -> None:
        """
        Initializes the NiryoWorkspace.

        Args:
            workspace_id (str): Unique ID of the workspace.
            environment (EnvironmentProtocol): Object providing robot environment access.
            verbose (bool): Whether to enable verbose output.
            config (WorkspaceConfig, optional): Optional workspace configuration.
        """
        self._environment = environment
        self._config = config
        self._logger = logging.getLogger("robot_workspace")

        super().__init__(workspace_id, verbose)

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    @classmethod
    def from_config(cls, config: WorkspaceConfig, environment: EnvironmentProtocol, verbose: bool = False) -> NiryoWorkspace:
        """
        Creates a NiryoWorkspace instance from a configuration object.

        Args:
            config (WorkspaceConfig): Configuration instance.
            environment (EnvironmentProtocol): Environment object.
            verbose (bool): Whether to enable verbose output.

        Returns:
            NiryoWorkspace: The initialized workspace instance.
        """
        workspace = cls(config.id, environment, verbose, config)

        # Override observation pose from config if available
        if config.observation_pose:
            workspace._observation_pose = PoseObjectPNP(
                x=config.observation_pose.x,
                y=config.observation_pose.y,
                z=config.observation_pose.z,
                roll=config.observation_pose.roll,
                pitch=config.observation_pose.pitch,
                yaw=config.observation_pose.yaw,
            )

        # Override image shape from config if available
        if config.image_shape:
            workspace.set_img_shape(config.image_shape)

        return workspace

    def transform_camera2world_coords(self, workspace_id: str, u_rel: float, v_rel: float, yaw: float = 0.0) -> PoseObjectPNP:
        """
        Transforms relative image coordinates to Niryo world coordinates.

        Args:
            workspace_id (str): ID of the workspace.
            u_rel (float): Normalized horizontal coordinate [0, 1].
            v_rel (float): Normalized vertical coordinate [0, 1].
            yaw (float): Orientation of the object.

        Returns:
            PoseObjectPNP: Corresponding pose in world coordinates.
        """
        if self.verbose():
            self._logger.debug(
                f"transform_camera2world_coords input - workspace_id: {workspace_id}, u_rel: {u_rel}, v_rel: {v_rel}, yaw: {yaw}"
            )

        obj_coords = self._environment.get_robot_target_pose_from_rel(workspace_id, u_rel, v_rel, yaw)

        if self.verbose():
            self._logger.debug(f"transform_camera2world_coords output: {obj_coords}")

        return obj_coords

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    @log_start_end_cls()
    def _set_4corners_of_workspace(self) -> None:
        """Sets the four corners of the workspace in world coordinates."""
        self._xy_ul_wc = self.transform_camera2world_coords(self._id, 0.0, 0.0)
        self._xy_ll_wc = self.transform_camera2world_coords(self._id, 0.0, 1.0)
        self._xy_ur_wc = self.transform_camera2world_coords(self._id, 1.0, 0.0)
        self._xy_lr_wc = self.transform_camera2world_coords(self._id, 1.0, 1.0)

        if self.verbose():
            self._logger.debug(f"Workspace corners - UL: {self._xy_ul_wc}, LL: {self._xy_ll_wc}")
            self._logger.debug(f"Workspace corners - UR: {self._xy_ur_wc}, LR: {self._xy_lr_wc}")

    @log_start_end_cls()
    def _set_observation_pose(self) -> None:
        """Sets the observation pose using configuration data."""
        if not self._config:
            raise ValueError(
                f"No configuration provided for workspace '{self._id}'. " "Initialize with config_path or from_config()."
            )

        if not self._config.observation_pose:
            raise ValueError(f"No observation pose defined in config for workspace '{self._id}'")

        self._observation_pose = PoseObjectPNP(
            x=self._config.observation_pose.x,
            y=self._config.observation_pose.y,
            z=self._config.observation_pose.z,
            roll=self._config.observation_pose.roll,
            pitch=self._config.observation_pose.pitch,
            yaw=self._config.observation_pose.yaw,
        )

    # *** PRIVATE STATIC/CLASS methods ***

    # *** PUBLIC properties ***

    def environment(self) -> EnvironmentProtocol:
        """Returns the environment associated with this workspace."""
        return self._environment

    # *** PRIVATE variables ***

    _environment: EnvironmentProtocol = None
    _logger: logging.Logger = None
