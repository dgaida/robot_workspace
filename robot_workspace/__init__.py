from __future__ import annotations

"""
Robot Environment - A framework for robotic pick-and-place operations with vision.

This package provides a comprehensive system for controlling robotic arms with
integrated computer vision, workspace management, and manipulation capabilities.
"""

from .config import ConfigManager, PoseConfig, RobotConfig, WorkspaceConfig
from .objects.object import Object
from .objects.object_api import Location
from .objects.objects import Objects
from .objects.pose_object import PoseObjectPNP
from .workspaces.niryo_workspace import NiryoWorkspace
from .workspaces.niryo_workspaces import NiryoWorkspaces
from .workspaces.widowx_workspace import WidowXWorkspace
from .workspaces.widowx_workspaces import WidowXWorkspaces
from .workspaces.workspace import Workspace
from .workspaces.workspaces import Workspaces

__version__ = "0.1.0"
__author__ = "Daniel Gaida"

__all__ = [
    "ConfigManager",
    "Location",
    "NiryoWorkspace",
    "NiryoWorkspaces",
    "Object",
    "Objects",
    "PoseConfig",
    "PoseObjectPNP",
    "RobotConfig",
    "WidowXWorkspace",
    "WidowXWorkspaces",
    "Workspace",
    "WorkspaceConfig",
    "Workspaces",
]
