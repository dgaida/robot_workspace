# Defines a workspace where the pick-and-place robot Niryo Ned2 can pick or place objects from.
# Not final, but works
# Documentation and type definitions are almost final (chatgpt might be able to improve it).

from ..common.logger import log_start_end_cls

from .workspace import Workspace
from ..objects.pose_object import PoseObjectPNP

# from typing import TYPE_CHECKING
import math

# if TYPE_CHECKING:
#     from ..environment import Environment


class NiryoWorkspace(Workspace):
    """
    A workspace where the pick-and-place robot Niryo Ned2 can pick or place objects from.

    """

    # *** CONSTRUCTORS ***
    def __init__(self, workspace_id: str, environment, verbose: bool = False):  # : "Environment"
        """
        Inits the workspace.

        Args:
            workspace_id: id of the workspace
            environment: object of the Environment class
            verbose:
        """
        self._environment = environment

        super().__init__(workspace_id, verbose)

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    # @log_start_end_cls()
    def transform_camera2world_coords(
        self, workspace_id: str, u_rel: float, v_rel: float, yaw: float = 0.0
    ) -> "PoseObjectPNP":
        """
        Given relative image coordinates [u_rel, v_rel] and optionally an orientation of the point (yaw),
        calculate the corresponding pose in world coordinates. The parameter yaw is useful, if we want to pick at the
        given coordinate an object that has the given orientation. For this method to work, it is important that
        only the workspace of the robot is visible in the image and nothing else. At least for the Niryo robot
        this is important. This means, (u_rel, v_rel) = (0, 0), is the upper left corner of the workspace.

        Args:
            workspace_id: id of the workspace
            u_rel: horizontal coordinate in image of workspace, normalized between 0 and 1
            v_rel: vertical coordinate in image of workspace, normalized between 0 and 1
            yaw: orientation of an object at the pixel coordinates [u_rel, v_rel].

        Returns:
            PoseObjectPNP: Pose of the point in world coordinates of the robot.
        """
        if self.verbose():
            print(workspace_id, u_rel, v_rel, yaw)

        obj_coords = self._environment.get_robot_target_pose_from_rel(workspace_id, u_rel, v_rel, yaw)

        if self.verbose():
            print("transform_camera2world_coords:", obj_coords)

        return obj_coords

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    @log_start_end_cls()
    def _set_4corners_of_workspace(self) -> None:
        """
        sets the 4 corners of the workspace, being _xy_ul_wc, ...
        """
        self._xy_ul_wc = self.transform_camera2world_coords(self._id, 0.0, 0.0)
        self._xy_ll_wc = self.transform_camera2world_coords(self._id, 0.0, 1.0)
        self._xy_ur_wc = self.transform_camera2world_coords(self._id, 1.0, 0.0)
        self._xy_lr_wc = self.transform_camera2world_coords(self._id, 1.0, 1.0)

        if self.verbose():
            print(self._xy_ul_wc, self._xy_ll_wc)
            print(self._xy_ur_wc, self._xy_lr_wc)

    @log_start_end_cls()
    def _set_observation_pose(self) -> None:
        """
        Set the variable _observation_pose for the given workspace. An observation pose is a pose of the gripper
        where the gripper hovers over the workspace. For the niryo robot this is a gripper pose in which the
        gripper mounted camera can observe the complete workspace.

        Supports multiple workspace configurations.
        """
        if self._id == "niryo_ws" or self._id == "niryo_ws2":
            self._observation_pose = PoseObjectPNP(  # position for the robot to watch the workspace in the real world
                x=0.173 - 0.0,
                y=-0.002,
                z=0.247 + 0.03,
                roll=-3.042,
                pitch=1.327 - 0.0,
                yaw=-3.027,
            )
        # Left workspace (for multi-workspace setup)
        elif self._id == "niryo_ws_left":
            self._observation_pose = PoseObjectPNP(
                x=0.173,
                y=0.10,  # Shifted to the left
                z=0.277,
                roll=-3.042,
                pitch=1.327,
                yaw=-3.027,
            )

        # Right workspace (for multi-workspace setup)
        elif self._id == "niryo_ws_right":
            self._observation_pose = PoseObjectPNP(
                x=0.173,
                y=-0.10,  # Shifted to the right
                z=0.277,
                roll=-3.042,
                pitch=1.327,
                yaw=-3.027,
            )

        # Gazebo simulation workspaces
        elif self._id == "gazebo_1":
            self._observation_pose = PoseObjectPNP(
                x=0.18,
                y=0.0,
                z=0.36,
                roll=2.4,
                pitch=math.pi / 2,
                yaw=2.4,
            )

        elif self._id == "gazebo_2":
            self._observation_pose = PoseObjectPNP(
                x=0.18,
                y=0.15,  # Second workspace in simulation
                z=0.36,
                roll=2.4,
                pitch=math.pi / 2,
                yaw=2.4,
            )

        elif self._id == "gazebo_1_chocolate_bars":
            self._observation_pose = PoseObjectPNP(
                x=0.18,
                y=0,
                z=0.36,
                roll=0.0,
                pitch=math.pi / 2 + 0.3,
                yaw=0.0,
            )

        else:
            self._observation_pose = None
            if self._verbose:
                print(f"Warning: No observation pose defined for workspace '{self._id}'")

    # *** PRIVATE STATIC/CLASS methods ***

    # *** PUBLIC properties ***

    def environment(self):
        return self._environment

    # *** PRIVATE variables ***

    _environment = None
