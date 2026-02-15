from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

# Objects class that is a List of Object's
# Documentation and type definitions are final
from ..common.logger import log_start_end_cls
from .object import Object
from .object_api import Location

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..workspaces.workspace import Workspace


class Objects(list[Object]):
    """
    A class representing a list of Object instances.

    Objects are typically stored in a workspace and represent physical entities detected by vision.
    This class provides several spatial query methods to find objects based on coordinates, labels, or size.
    """

    # *** CONSTRUCTORS ***
    def __init__(self, iterable: Iterable[Object] | None = None, verbose: bool = False) -> None:
        """
        Initializes the Objects instance.

        Args:
            iterable (Iterable[Object], optional): An iterable of Object instances. Defaults to an empty list.
            verbose (bool): If True, enables verbose logging.
        """
        if iterable is None:
            iterable = []
        super().__init__(iterable)

        self._verbose = verbose

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    @log_start_end_cls()
    def get_detected_object(
        self, coordinate: list[float], label: str | None = None, serializable: bool = False
    ) -> Object | dict[str, Any] | None:
        """
        Retrieves a detected object at or near a specified world coordinate, optionally filtering by label.

        Checks for objects that are within a 2-centimeter radius of the specified coordinate.
        If multiple objects meet the criteria, the first one found is returned.

        Args:
            coordinate (list[float]): A 2D coordinate in world units [x, y].
            label (str, optional): An optional filter for the object's label.
            serializable (bool): If True, returns a dictionary representation instead of an Object instance.

        Returns:
            Object | dict[str, Any] | None: The first object detected near the given coordinate, or None if not found.
        """
        detected_objects = self.get_detected_objects(Location.CLOSE_TO, coordinate, label)

        if detected_objects:
            res: Object | dict[str, Any] = detected_objects[0].to_dict() if serializable else detected_objects[0]
            return res
        else:
            return None

    def get_detected_objects(
        self,
        location: Location | str = Location.NONE,
        coordinate: list[float] | None = None,
        label: str | None = None,
    ) -> Objects:
        """
        Returns a list of objects filtered by spatial location, coordinate, and label.

        Args:
            location (Location | str): Spatial filter. Values can be "left next to", "right next to",
                "above", "below", "close to", or Location enum equivalents.
            coordinate (list[float], optional): (x, y) coordinate in meters used for spatial filtering.
                Required if 'location' is not NONE.
            label (str, optional): Filter by object label (substring match).

        Returns:
            Objects: A collection of filtered objects.

        Raises:
            ValueError: If coordinate is missing but required for the specified location filter.
        """
        detected_objects = self

        if label is not None:
            detected_objects = Objects(obj for obj in self if label in obj.label())

        location = Location.convert_str2location(location)

        if location is Location.NONE:
            return detected_objects

        if coordinate is None:
            raise ValueError(f"Coordinate must be provided for location filter: {location}")

        if location == Location.LEFT_NEXT_TO:
            return Objects(obj for obj in detected_objects if obj.y_com() > coordinate[1])
        elif location == Location.RIGHT_NEXT_TO:
            return Objects(obj for obj in detected_objects if obj.y_com() < coordinate[1])
        elif location == Location.ABOVE:
            return Objects(obj for obj in detected_objects if obj.x_com() > coordinate[0])
        elif location == Location.BELOW:
            return Objects(obj for obj in detected_objects if obj.x_com() < coordinate[0])
        elif location == Location.CLOSE_TO:
            return Objects(
                obj
                for obj in detected_objects
                if np.sqrt((obj.x_com() - coordinate[0]) ** 2 + (obj.y_com() - coordinate[1]) ** 2) <= 0.02
            )
        else:
            print("Error in get_detected_objects: Unknown Location:", location)
            return Objects()

    def get_detected_objects_serializable(
        self,
        location: Location | str = Location.NONE,
        coordinate: list[float] | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Similar to get_detected_objects but returns a list of dictionaries.

        Args:
            location (Location | str): Spatial filter.
            coordinate (list[float], optional): Reference (x, y) coordinate.
            label (str, optional): Filter by object label.

        Returns:
            list[dict[str, Any]]: List of dictionary representations of the filtered objects.
        """
        detected_objects = self.get_detected_objects(location, coordinate, label)

        objects = Objects.objects_to_dict_list(detected_objects)

        return objects

    def get_nearest_detected_object(self, coordinate: list[float], label: str | None = None) -> tuple[Object | None, float]:
        """
        Finds the object nearest to a specified coordinate.

        Args:
            coordinate (list[float]): Target (x, y) coordinate.
            label (str, optional): If specified, only consider objects with this label.

        Returns:
            tuple[Object | None, float]: (nearest_object, distance_in_meters).
        """
        nearest_object = None
        min_distance = float("inf")

        for obj in self:
            if label is None or obj.label() == label:
                # Calculate Euclidean distance
                distance = math.sqrt((obj.x_com() - coordinate[0]) ** 2 + (obj.y_com() - coordinate[1]) ** 2)
                # print(distance, min_distance)
                if distance < min_distance:
                    min_distance = distance
                    nearest_object = obj

        return nearest_object, min_distance

    def get_detected_objects_as_comma_separated_string(self) -> str:
        """
        Returns the labels of all detected objects as a comma-separated string.

        Returns:
            str: Comma-separated labels.
        """
        return f"""{', '.join(f"'{item.label()}'" for item in self)}"""

    def get_largest_detected_object(self, serializable: bool = False) -> tuple[Object, float] | tuple[dict[str, Any], float]:
        """
        Identifies the largest object by area.

        Args:
            serializable (bool): If True, returns a dictionary instead of an Object.

        Returns:
            tuple: (largest_object, area_m2).
        """
        largest_object = max(self, key=lambda obj: obj.size_m2())

        size = largest_object.size_m2()

        if serializable:
            return largest_object.to_dict(), size
        else:
            return largest_object, size

    def get_smallest_detected_object(self, serializable: bool = False) -> tuple[Object, float] | tuple[dict[str, Any], float]:
        """
        Identifies the smallest object by area.

        Args:
            serializable (bool): If True, returns a dictionary instead of an Object.

        Returns:
            tuple: (smallest_object, area_m2).
        """
        smallest_object = min(self, key=lambda obj: obj.size_m2())

        size = smallest_object.size_m2()

        if serializable:
            return smallest_object.to_dict(), size
        else:
            return smallest_object, size

    def get_detected_objects_sorted(
        self, ascending: bool = True, serializable: bool = False
    ) -> Objects | list[dict[str, Any]]:
        """
        Returns objects sorted by their size.

        Args:
            ascending (bool): Sorting order. Defaults to True.
            serializable (bool): If True, returns a list of dictionaries.

        Returns:
            Objects | list[dict[str, Any]]: Sorted collection of objects.
        """
        sorted_objs = Objects(sorted(self, key=lambda obj: obj.size_m2(), reverse=not ascending))

        if serializable:
            return Objects.objects_to_dict_list(sorted_objs)
        else:
            return sorted_objs

    # *** PUBLIC methods ***

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** HELPER METHODE FÜR REDIS PUBLISHER ***

    @staticmethod
    def objects_to_dict_list(objects: list[Object]) -> list[dict[str, Any]]:
        """
        Converts a list of Object instances to a list of dictionaries.

        Args:
            objects (list[Object]): List of Object instances.

        Returns:
            list[dict[str, Any]]: List of dictionary representations.
        """
        return [obj.to_dict() for obj in objects]

    @staticmethod
    def dict_list_to_objects(dict_list: list[dict[str, Any]], workspace: Workspace) -> Objects:
        """
        Reconstructs an Objects collection from a list of dictionaries.

        Args:
            dict_list (list[dict[str, Any]]): List of object dictionaries.
            workspace (Workspace): The workspace to associate with the objects.

        Returns:
            Objects: A collection of reconstructed objects.
        """
        objects = Objects()
        for obj_dict in dict_list:
            obj = Object.from_dict(obj_dict, workspace)
            if obj is not None:
                objects.append(obj)
        return objects

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    def verbose(self) -> bool:
        """
        Returns whether verbose logging is enabled.

        Returns:
            bool: True if verbose, else False.
        """
        return self._verbose

    # *** PRIVATE variables ***

    _verbose = False
