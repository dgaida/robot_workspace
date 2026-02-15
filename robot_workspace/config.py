"""
Configuration management for robot workspaces.

Provides configuration loading from YAML files for workspace parameters,
observation poses, and other robot-specific settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore


@dataclass
class PoseConfig:
    """Configuration for a pose (position + orientation)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {"x": self.x, "y": self.y, "z": self.z, "roll": self.roll, "pitch": self.pitch, "yaw": self.yaw}


@dataclass
class WorkspaceConfig:
    """Configuration for a single workspace."""

    id: str
    observation_pose: PoseConfig
    image_shape: tuple[int, int, int] = (640, 480, 3)
    corners: dict[str, PoseConfig] | None = None
    robot_type: str = "niryo"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceConfig:
        """
        Create WorkspaceConfig from dictionary.

        Args:
            data: Dictionary with workspace configuration

        Returns:
            WorkspaceConfig instance

        Example:
            >>> data = {
            ...     'id': 'niryo_ws',
            ...     'observation_pose': {'x': 0.173, 'y': -0.002, ...},
            ...     'image_shape': [640, 480, 3]
            ... }
            >>> config = WorkspaceConfig.from_dict(data)
        """
        # Parse observation pose
        obs_pose_data = data.get("observation_pose", {})
        observation_pose = PoseConfig(**obs_pose_data)

        # Parse image shape
        img_shape = data.get("image_shape", [640, 480, 3])
        image_shape = tuple(img_shape) if isinstance(img_shape, list) else img_shape

        # Parse corners if present
        corners = None
        if "corners" in data:
            corners = {key: PoseConfig(**pose_data) for key, pose_data in data["corners"].items()}

        return cls(
            id=data["id"],
            observation_pose=observation_pose,
            image_shape=image_shape,
            corners=corners,
            robot_type=data.get("robot_type", "niryo"),
        )


@dataclass
class RobotConfig:
    """Configuration for robot-specific parameters."""

    name: str
    workspaces: list[WorkspaceConfig] = field(default_factory=list)
    simulation_workspaces: list[WorkspaceConfig] = field(default_factory=list)
    default_workspace_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotConfig:
        """
        Create RobotConfig from dictionary.

        Args:
            data: Dictionary with robot configuration

        Returns:
            RobotConfig instance
        """
        workspaces = [WorkspaceConfig.from_dict(ws_data) for ws_data in data.get("workspaces", [])]

        sim_workspaces = [WorkspaceConfig.from_dict(ws_data) for ws_data in data.get("simulation_workspaces", [])]

        return cls(
            name=data["name"],
            workspaces=workspaces,
            simulation_workspaces=sim_workspaces,
            default_workspace_id=data.get("default_workspace_id"),
        )


class ConfigManager:
    """
    Manages configuration loading and access for robot workspaces.

    Example:
        >>> config_mgr = ConfigManager()
        >>> config_mgr.load_from_yaml('config/niryo_config.yaml')
        >>> workspace_config = config_mgr.get_workspace_config('niryo_ws')
    """

    def __init__(self) -> None:
        self._robot_configs: dict[str, RobotConfig] = {}
        self._workspace_configs: dict[str, WorkspaceConfig] = {}

    def load_from_yaml(self, path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Load robot configurations
        for robot_name, robot_data in data.get("robots", {}).items():
            robot_data["name"] = robot_name
            robot_config = RobotConfig.from_dict(robot_data)
            self._robot_configs[robot_name] = robot_config

            # Index workspaces for quick lookup
            for ws_config in robot_config.workspaces:
                self._workspace_configs[ws_config.id] = ws_config
            for ws_config in robot_config.simulation_workspaces:
                self._workspace_configs[ws_config.id] = ws_config

    def get_robot_config(self, robot_name: str) -> RobotConfig | None:
        """Get configuration for a specific robot."""
        return self._robot_configs.get(robot_name)

    def get_workspace_config(self, workspace_id: str) -> WorkspaceConfig | None:
        """Get configuration for a specific workspace."""
        return self._workspace_configs.get(workspace_id)

    def get_workspace_configs(self, robot_name: str, simulation: bool = False) -> list[WorkspaceConfig]:
        """
        Get all workspace configurations for a robot.

        Args:
            robot_name: Name of the robot
            simulation: If True, return simulation workspaces

        Returns:
            List of WorkspaceConfig instances
        """
        robot_config = self.get_robot_config(robot_name)
        if robot_config is None:
            return []

        if simulation:
            return robot_config.simulation_workspaces
        return robot_config.workspaces

    def list_workspace_ids(self, robot_name: str | None = None) -> list[str]:
        """
        List all available workspace IDs.

        Args:
            robot_name: If provided, only return workspaces for this robot

        Returns:
            List of workspace IDs
        """
        if robot_name is None:
            return list(self._workspace_configs.keys())

        robot_config = self.get_robot_config(robot_name)
        if robot_config is None:
            return []

        workspace_ids = [ws.id for ws in robot_config.workspaces]
        workspace_ids.extend([ws.id for ws in robot_config.simulation_workspaces])
        return workspace_ids
