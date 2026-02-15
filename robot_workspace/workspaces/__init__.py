"""
Workspaces package for the robot_environment system.

This package provides workspace management functionality for pick-and-place robots,
including abstract workspace definitions and concrete implementations for specific robots.

Classes:
    Workspace: Abstract base class for robot workspaces
    Workspaces: Collection class for managing multiple workspaces
    NiryoWorkspace: Concrete workspace implementation for Niryo Ned2 robot
    NiryoWorkspaces: Collection of NiryoWorkspace instances

Example:
    from robot_environment.workspaces import NiryoWorkspaces, NiryoWorkspace

    # Create a collection of workspaces for Niryo robot
    workspaces = NiryoWorkspaces(environment, verbose=True)

    # Get a specific workspace by ID
    workspace = workspaces.get_workspace_by_id("niryo_ws")
"""

from __future__ import annotations

from .niryo_workspace import NiryoWorkspace
from .niryo_workspaces import NiryoWorkspaces
from .widowx_workspace import WidowXWorkspace
from .widowx_workspaces import WidowXWorkspaces
from .workspace import Workspace
from .workspaces import Workspaces

__all__ = ["NiryoWorkspace", "NiryoWorkspaces", "WidowXWorkspace", "WidowXWorkspaces", "Workspace", "Workspaces"]

__version__ = "0.1.0"
__author__ = "Daniel Gaida"
