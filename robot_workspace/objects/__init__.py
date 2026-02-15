"""
Object detection and representation module.

Provides classes for representing detected objects with their physical
properties, positions, and segmentation information.
"""

from __future__ import annotations

from .object import Object
from .object_api import Location, ObjectAPI
from .objects import Objects
from .pose_object import PoseObjectPNP

__all__ = [
    "Location",
    "Object",
    "ObjectAPI",
    "Objects",
    "PoseObjectPNP",
]
