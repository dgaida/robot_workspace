# Defines a workspace where the pick-and-place robot WidowX 250 6DOF can pick or place objects from.
# Documentation and type definitions are almost final (chatgpt might be able to improve it).

from ..common.logger import log_start_end_cls

from .workspace import Workspace
from ..objects.pose_object import PoseObjectPNP
import logging

"""
TODO: APPLY SAME CHANGES AS niryo_workspace.py:
1. Add 'config' parameter to __init__
2. Add from_config() class method
3. Add config check at beginning of _set_observation_pose()
"""


class WidowXWorkspace(Workspace):
    """
    A workspace where the pick-and-place robot WidowX 250 6DOF can pick or place objects from.

    The WidowX robot typically uses a third-person camera view (e.g., Intel RealSense)
    rather than a gripper-mounted camera like the Niryo.
    """

    # *** CONSTRUCTORS ***
    def __init__(self, workspace_id: str, environment, verbose: bool = False):
        """
        Inits the workspace.

        Args:
            workspace_id: id of the workspace
            environment: object of the Environment class
            verbose: enable verbose output
        """
        self._environment = environment
        self._logger = logging.getLogger("robot_workspace")

        super().__init__(workspace_id, verbose)

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    def transform_camera2world_coords(
        self, workspace_id: str, u_rel: float, v_rel: float, yaw: float = 0.0
    ) -> "PoseObjectPNP":
        """
        Given relative image coordinates [u_rel, v_rel] and optionally an orientation of the point (yaw),
        calculate the corresponding pose in world coordinates.

        For WidowX with a third-person camera, this transformation maps from the camera's view
        to the robot's base frame. The workspace is defined by calibrated corner markers or
        predefined bounds.

        Args:
            workspace_id: id of the workspace
            u_rel: horizontal coordinate in image of workspace, normalized between 0 and 1
            v_rel: vertical coordinate in image of workspace, normalized between 0 and 1
            yaw: orientation of an object at the pixel coordinates [u_rel, v_rel].

        Returns:
            PoseObjectPNP: Pose of the point in world coordinates of the robot.
        """
        if self.verbose():
            self._logger.debug(
                f"transform_camera2world_coords input - workspace_id: {workspace_id}, u_rel: {u_rel}, v_rel: {v_rel}, yaw: {yaw}"
            )

        # For WidowX, we need to handle coordinate transformation differently
        # since it uses a third-person camera rather than gripper-mounted camera

        # Get workspace bounds in world coordinates
        # Map relative coordinates to world coordinates using linear interpolation
        x_min = self._xy_lr_wc.x  # Lower right (farther from robot)
        x_max = self._xy_ul_wc.x  # Upper left (closer to robot)
        y_min = self._xy_lr_wc.y  # Lower right (rightmost)
        y_max = self._xy_ul_wc.y  # Upper left (leftmost)

        # Linear interpolation
        x = x_max - u_rel * (x_max - x_min)  # u_rel: 0 (top) -> 1 (bottom)
        y = y_max - v_rel * (y_max - y_min)  # v_rel: 0 (left) -> 1 (right)

        # Default height above workspace
        z = 0.05  # 5cm above workspace surface

        # Default orientation for WidowX gripper
        # Roll and pitch point gripper downward, yaw from parameter
        roll = 0.0
        pitch = 1.57  # ~90 degrees (pointing down)

        obj_coords = PoseObjectPNP(x, y, z, roll, pitch, yaw)

        if self.verbose():
            self._logger.debug(f"transform_camera2world_coords output: {obj_coords}")

        return obj_coords

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    @log_start_end_cls()
    def _set_4corners_of_workspace(self) -> None:
        """
        Sets the 4 corners of the workspace using the transform method.

        For WidowX with a third-person camera, these corners are typically
        calibrated based on the camera's view of the workspace.
        """
        self._xy_ul_wc = self.transform_camera2world_coords(self._id, 0.0, 0.0)
        self._xy_ll_wc = self.transform_camera2world_coords(self._id, 1.0, 0.0)
        self._xy_ur_wc = self.transform_camera2world_coords(self._id, 0.0, 1.0)
        self._xy_lr_wc = self.transform_camera2world_coords(self._id, 1.0, 1.0)

        if self.verbose():
            self._logger.debug(f"Workspace corners - UL: {self._xy_ul_wc}, LL: {self._xy_ll_wc}")
            self._logger.debug(f"Workspace corners - UR: {self._xy_ur_wc}, LR: {self._xy_lr_wc}")

    @log_start_end_cls()
    def _set_observation_pose(self) -> None:
        """
        Set the variable _observation_pose for the given workspace.

        For WidowX, the observation pose is typically a retracted position
        where the robot arm is out of the camera's view, allowing clear
        observation of the workspace.

        Supports multiple workspace configurations.
        """
        if self._id == "widowx_ws" or self._id == "widowx_ws_main":
            # Main workspace - robot retracted to home position
            # WidowX 250 typical home pose values
            self._observation_pose = PoseObjectPNP(
                x=0.30,  # 30cm forward from base
                y=0.0,  # Centered
                z=0.25,  # 25cm above base
                roll=0.0,
                pitch=0.0,  # Horizontal orientation for observation
                yaw=0.0,
            )

        # Left workspace (for multi-workspace setup)
        elif self._id == "widowx_ws_left":
            self._observation_pose = PoseObjectPNP(
                x=0.30,
                y=0.15,  # Shifted to the left
                z=0.25,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
            )

        # Right workspace (for multi-workspace setup)
        elif self._id == "widowx_ws_right":
            self._observation_pose = PoseObjectPNP(
                x=0.30,
                y=-0.15,  # Shifted to the right
                z=0.25,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
            )

        # Simulation workspaces
        elif self._id == "gazebo_widowx_1":
            self._observation_pose = PoseObjectPNP(
                x=0.30,
                y=0.0,
                z=0.30,  # Slightly higher in simulation
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
            )

        elif self._id == "gazebo_widowx_2":
            self._observation_pose = PoseObjectPNP(
                x=0.30,
                y=0.15,  # Second workspace in simulation
                z=0.30,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
            )

        # Extended reach workspace (farther from base)
        elif self._id == "widowx_ws_extended":
            self._observation_pose = PoseObjectPNP(
                x=0.40,  # 40cm forward (near max reach)
                y=0.0,
                z=0.20,  # Lower height for extended reach
                roll=0.0,
                pitch=0.3,  # Slight downward tilt
                yaw=0.0,
            )

        else:
            self._observation_pose = None
            if self._verbose:
                self._logger.warning(f"No observation pose defined for workspace '{self._id}'")

    # *** PRIVATE STATIC/CLASS methods ***

    # *** PUBLIC properties ***

    def environment(self):
        return self._environment

    # *** PRIVATE variables ***

    _environment = None
    _logger = None
