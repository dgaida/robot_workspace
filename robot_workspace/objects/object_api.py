# abstract Class defining an object. An object has pixel coordinates as well as world coordinates.
# Documentation and type definitions are final.

from abc import ABC, abstractmethod

from typing import List

from enum import Enum


class Location(Enum):
    """
    Class that defines Locations, needed in the class RobotAPI, Robot and AgentAPI, Agent
    """

    LEFT_NEXT_TO = "left next to"
    RIGHT_NEXT_TO = "right next to"
    ABOVE = "above"
    BELOW = "below"
    ON_TOP_OF = "on top of"
    INSIDE = "inside"
    CLOSE_TO = "close to"
    NONE = None

    @staticmethod
    def convert_str2location(location: Union["Location", str, None]) -> "Location":
        """
        Converts a string to a Location enum if it matches one of the Location values.
        If already a Location, returns it unchanged.

        Args:
            location (Union[Location, str]): A Location object or a string representing a location.

        Returns:
            Location: The corresponding Location object.
        """
        if isinstance(location, str):
            # Match string to enum value
            for loc in Location:
                if location == loc.value:
                    return loc
            raise ValueError(f"Invalid location string: {location}")
        elif isinstance(location, Location):
            return location
        elif location is None:
            return Location.NONE
        else:
            raise TypeError("Location must be either a string or a Location enum")


class ObjectAPI(ABC):
    """
    Abstract Class defining an object. An object has pixel coordinates as well as world coordinates.
    This class provides those methods of teh class Object that are provided to the LLM.

    """

    # *** CONSTRUCTORS ***
    def __init__(self):
        pass

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    @abstractmethod
    def label(self) -> str:
        """
        Returns the label of the object (e.g., "chocolate bar").

        Returns:
            str: the label of the object.
        """
        pass

    @abstractmethod
    def coordinate(self) -> List[float]:
        """
        Returns (x,y) world coordinates of the center of mass of the object, measured in meters.
        At these coordinates you can pick the object. The List contains two float values.

        Returns:
            List[float]: x,y world coordinates of the object. At these coordinates you can pick the object.
        """
        pass

    # *** PRIVATE variables ***
