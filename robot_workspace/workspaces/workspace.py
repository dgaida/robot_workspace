from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# abstract class Workspace for the pnp_robot_genai package
# Not yet final
# TODO in is_visible
# Documentation and type definitions are almost final (chatgpt might be able to improve it).
from ..common.logger import log_start_end_cls
from ..objects.object import Object
from ..objects.pose_object import PoseObjectPNP

if TYPE_CHECKING:
    # from ..environment import Environment
    from ..objects.pose_object import PoseObjectPNP


class Workspace(ABC):
    """
    Abstract base class representing a robotic workspace.

    A workspace defines a region where a robot can pick and place objects.
    It handles coordinate transformations between camera images and world coordinates.

    Attributes:
        _id (str): Unique identifier for the workspace.
        _verbose (bool): If True, enables verbose logging.
        _logger (logging.Logger): Logger instance.
        _observation_pose (PoseObjectPNP): Optimal pose for the camera to observe the workspace.
        _xy_ul_wc (PoseObjectPNP): Upper-left corner in world coordinates.
        _xy_ll_wc (PoseObjectPNP): Lower-left corner in world coordinates.
        _xy_ur_wc (PoseObjectPNP): Upper-right corner in world coordinates.
        _xy_lr_wc (PoseObjectPNP): Lower-right corner in world coordinates.
        _xy_center_wc (PoseObjectPNP): Center of the workspace in world coordinates.
        _width_m (float): Width of the workspace in meters.
        _height_m (float): Height of the workspace in meters.
        _img_shape (tuple[int, int, int]): Shape of the camera image (height, width, channels).
    """

    # *** CONSTRUCTORS ***
    @log_start_end_cls()
    def __init__(self, workspace_id: str, verbose: bool = False) -> None:
        """
        Initializes the workspace.

        Args:
            workspace_id (str): Unique ID of the workspace.
            verbose (bool): Whether to enable verbose logging.
        """
        self._id = workspace_id
        self._verbose = verbose
        self._logger = logging.getLogger("robot_workspace")

        self._set_observation_pose()

        self._set_4corners_of_workspace()

        self._calc_width_height()

        self._calc_center_of_workspace()

    def __str__(self) -> str:
        size = f"width = {self._width_m:.2f}, height = {self._height_m:.2f}"
        return "Workspace " + self.id() + "\n" + size + "\n" + str(self._xy_center_wc)

    def __repr__(self) -> str:
        return self.__str__()

    # *** PUBLIC GET methods ***

    @log_start_end_cls()
    def is_visible(self, camera_pose: PoseObjectPNP) -> bool:
        """
        Checks whether the workspace is visible from the given camera pose.

        Args:
            camera_pose (PoseObjectPNP): The current pose of the camera.

        Returns:
            bool: True if the workspace is considered visible, False otherwise.
        """
        if self.verbose() and self._logger:
            self._logger.debug(f"is_visible check: camera={camera_pose}, obs={self._observation_pose}")

        if self._observation_pose is None:
            return False

        return camera_pose.approx_eq_xyz(self._observation_pose)

    # *** PUBLIC methods ***

    def set_img_shape(self, img_shape: tuple[int, int, int]) -> None:
        """
        Sets the image shape for the workspace.

        Args:
            img_shape (tuple[int, int, int]): (height, width, channels).
        """
        self._img_shape = img_shape

    @abstractmethod
    def transform_camera2world_coords(self, workspace_id: str, u_rel: float, v_rel: float, yaw: float = 0.0) -> PoseObjectPNP:
        """
        Transforms relative image coordinates to world coordinates.

        Args:
            workspace_id (str): ID of the workspace.
            u_rel (float): Normalized horizontal coordinate [0, 1].
            v_rel (float): Normalized vertical coordinate [0, 1].
            yaw (float): Orientation of the object at the given point.

        Returns:
            PoseObjectPNP: Corresponding pose in world coordinates.
        """
        pass

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    @abstractmethod
    def _set_4corners_of_workspace(self) -> None:
        """Sets the four corners of the workspace in world coordinates."""
        pass

    @abstractmethod
    def _set_observation_pose(self) -> None:
        """Sets the optimal observation pose for the workspace."""
        pass

    @log_start_end_cls()
    def _calc_width_height(self) -> None:
        """Calculates the physical width and height of the workspace."""
        self._width_m, self._height_m = Object.calc_width_height(self._xy_ul_wc, self._xy_lr_wc)

    @log_start_end_cls()
    def _calc_center_of_workspace(self) -> None:
        """Calculates the center point of the workspace."""
        if self._xy_ll_wc is None or self._xy_ul_wc is None or self._xy_lr_wc is None:
            raise ValueError("Workspace corners must be set before calculating center.")

        dx = self._xy_ll_wc.x - self._xy_ul_wc.x
        dy = self._xy_ll_wc.y - self._xy_lr_wc.y

        self._xy_center_wc = self._xy_lr_wc.copy_with_offsets(-dx / 2.0, dy / 2.0)

        if self.verbose() and self._logger:
            self._logger.debug(f"_calc_center_of_workspace: {self._xy_center_wc}, {self._xy_ll_wc.x}")

    # *** PRIVATE STATIC/CLASS methods ***

    # *** PUBLIC properties ***

    def id(self) -> str:
        """Returns the workspace ID."""
        return self._id

    # def environment(self) -> "Environment":
    #     return self._environment

    def xy_ul_wc(self) -> PoseObjectPNP:
        """Returns the upper-left corner in world coordinates."""
        return self._xy_ul_wc

    def xy_ll_wc(self) -> PoseObjectPNP:
        """Returns the lower-left corner in world coordinates."""
        return self._xy_ll_wc

    def xy_ur_wc(self) -> PoseObjectPNP:
        """Returns the upper-right corner in world coordinates."""
        return self._xy_ur_wc

    def xy_lr_wc(self) -> PoseObjectPNP:
        """Returns the lower-right corner in world coordinates."""
        return self._xy_lr_wc

    def xy_center_wc(self) -> PoseObjectPNP:
        """Returns the center of the workspace in world coordinates."""
        return self._xy_center_wc

    def width_m(self) -> float:
        """Returns the width of the workspace in meters."""
        return self._width_m

    def height_m(self) -> float:
        """Returns the height of the workspace in meters."""
        return self._height_m

    def img_shape(self) -> tuple[int, int, int]:
        """Returns the image shape of the workspace."""
        return self._img_shape

    def observation_pose(self) -> PoseObjectPNP:
        """Returns the optimal observation pose."""
        return self._observation_pose

    def verbose(self) -> bool:
        """Returns whether verbose logging is enabled."""
        return self._verbose

    # *** PRIVATE variables ***

    _id: str = ""
    _xy_ul_wc: PoseObjectPNP = None
    _xy_ll_wc: PoseObjectPNP = None
    _xy_ur_wc: PoseObjectPNP = None
    _xy_lr_wc: PoseObjectPNP = None
    _xy_center_wc: PoseObjectPNP = None
    _width_m: float = 0.0
    _height_m: float = 0.0
    _img_shape: tuple[int, int, int] = None
    _observation_pose: PoseObjectPNP = None
    _verbose: bool = False
    _logger: logging.Logger = None
