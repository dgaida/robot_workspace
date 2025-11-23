"""
Robot Environment - A framework for robotic pick-and-place operations with vision.

This package provides a comprehensive system for controlling robotic arms with
integrated computer vision, workspace management, and manipulation capabilities.
"""

from .objects.object import Object
from .objects.objects import Objects
from .objects.pose_object import PoseObjectPNP

__version__ = "0.1.0"
__author__ = "Daniel Gaida"

__all__ = [
    "Object",
    "Objects",
    "PoseObjectPNP",
]
