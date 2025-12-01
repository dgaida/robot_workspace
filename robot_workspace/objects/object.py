# Class defining an object. An object has pixel coordinates as well as world coordinates.
# a TODO, but more or less finished
# Documentation and type definitions are final.

from ..common.logger import log_start_end_cls
from .object_api import ObjectAPI

from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union

if TYPE_CHECKING:
    from ..workspaces.workspace import Workspace
    from .pose_object import PoseObjectPNP

import numpy as np
import cv2
import math
import json
import time
import base64


class Object(ObjectAPI):
    """
    Class representing an object detected in a robot's workspace.

    Each object is characterized by its bounding box (in both pixel and world coordinates),
    segmentation mask (if available), size, and additional metadata. This class also provides
    methods to calculate dimensions, orientation, and other properties of the object.

    Attributes:
        - label: Label identifying the object (e.g., "chocolate bar").
        - workspace: Workspace in which the object resides.
        - segmentation mask: Optional mask representing the object's shape.
        - dimensions: Bounding box dimensions in both pixels and meters.
        - position: Object's position in relative image coordinates and world coordinates.
        - gripper_rotation: Suggested orientation for the robot gripper to pick up the object.
        - size: Area of the object in square meters.
    """

    # *** CONSTRUCTORS ***
    @log_start_end_cls()
    def __init__(
        self,
        label: str,
        u_min: int,
        v_min: int,
        u_max: int,
        v_max: int,
        mask_8u: np.ndarray,
        workspace: "Workspace",
        verbose: bool = False,
    ):
        """
        Initializes an Object instance.

        Args:
            label (str): Label of the object (e.g., "chocolate bar").
            u_min (int): Upper-left corner u-coordinate of the bounding box (in pixels).
            v_min (int): Upper-left corner v-coordinate of the bounding box (in pixels).
            u_max (int): Lower-right corner u-coordinate of the bounding box (in pixels).
            v_max (int): Lower-right corner v-coordinate of the bounding box (in pixels).
            mask_8u (np.ndarray): Optional segmentation mask of the object (8-bit, uint8).
            workspace (Workspace): Workspace instance where the object is located.
            verbose (bool): If True, enables verbose logging.

        Raises:
            ValueError: If the provided segmentation mask is not 8-bit unsigned.
        """
        super().__init__()

        self._label = label
        self._workspace = workspace
        self._verbose = verbose

        if self._workspace.img_shape() is None:
            raise ValueError("Object has no image shape. This probably means that the Niryo Workspace was not detected.")

        # Store original mask for rotation
        self._original_mask_8u = mask_8u.copy() if mask_8u is not None else None

        # Initialize the object with the common initialization logic
        self._init_object_properties(u_min, v_min, u_max, v_max, mask_8u)

    def _init_object_properties(self, u_min: int, v_min: int, u_max: int, v_max: int, mask_8u: np.ndarray) -> None:
        """
        Common initialization logic for object properties.
        This method is called by __init__ and can be reused by set_pose_com.

        Args:
            u_min (int): Upper-left corner u-coordinate of the bounding box (in pixels).
            v_min (int): Upper-left corner v-coordinate of the bounding box (in pixels).
            u_max (int): Lower-right corner u-coordinate of the bounding box (in pixels).
            v_max (int): Lower-right corner v-coordinate of the bounding box (in pixels).
            mask_8u (np.ndarray): Optional segmentation mask of the object (8-bit, uint8).
        """
        self._u_rel_min, self._v_rel_min = self._calc_rel_coordinates(u_min, v_min)
        self._u_rel_max, self._v_rel_max = self._calc_rel_coordinates(u_max, v_max)

        self._calc_width_height_m()

        self._width = self._u_rel_max - self._u_rel_min
        self._height = self._v_rel_max - self._v_rel_min

        if mask_8u is not None:
            self._calc_largest_contour(mask_8u)

            gripper_rotation, center = self._calc_gripper_orientation_from_segmentation_mask()
            width, height = self._rotated_bounding_box()

            u_0 = center[0]
            v_0 = center[1]

            self._gripper_rotation = gripper_rotation

            self._update_width_height(width, height)

            cx, cy = Object._calculate_center_of_mass(mask_8u)
        else:
            u_0 = int(u_min + (u_max - u_min) / 2)
            v_0 = int(v_min + (v_max - v_min) / 2)
            cx = u_0
            cy = v_0
            self._gripper_rotation = 0.0

        self._calc_size()

        self._pose_center, self._u_rel_o, self._v_rel_o = self._calc_pose_from_uv_coords(u_0, v_0)
        self._pose_com, self._u_rel_com, self._v_rel_com = self._calc_pose_from_uv_coords(cx, cy)

    def __str__(self):
        position = "x = {:.2f}, y = {:.2f}, z = {:.2f}".format(self._pose_com.x, self._pose_com.y, self._pose_com.z)
        orientation = "roll = {:.3f}, pitch = {:.3f}, yaw = {:.3f}".format(
            self._pose_com.roll, self._pose_com.pitch, self._pose_com.yaw
        )
        return self.label() + "\n" + position + "\n" + orientation

    def __repr__(self):
        return self.__str__()

    # *** PUBLIC SET methods ***

    def set_position(self, xy_coordinate: List):
        """
        Legacy method kept for backwards compatibility.
        Consider using set_pose_com instead for full pose updates.

        Args:
            xy_coordinate: [x, y] world coordinates
        """
        # Create a pose with the new x, y but keeping old z and orientation
        new_pose = self._pose_com.copy_with_offsets(
            x_offset=xy_coordinate[0] - self._pose_com.x, y_offset=xy_coordinate[1] - self._pose_com.y
        )
        self.set_pose_com(new_pose)

    def set_pose_com(self, pose_com: "PoseObjectPNP") -> None:
        """
        Updates the object's center of mass pose and recalculates all dependent properties.

        This method is called after a robot picks and places an object at a new location.
        It handles both position changes and orientation changes (rotation around z-axis).

        The method:
        1. Calculates the rotation angle difference (delta_yaw)
        2. Rotates the bounding box coordinates around the object's center
        3. Rotates the segmentation mask if available
        4. Recalculates all derived properties (width, height, size, etc.)

        Args:
            pose_com (PoseObjectPNP): New pose for the object's center of mass
        """
        if self.verbose():
            print(f"Setting new pose_com: {pose_com}")
            print(f"Old pose_com: {self._pose_com}")

        # Calculate rotation difference (only around z-axis/yaw)
        old_yaw = self._gripper_rotation
        new_yaw = pose_com.yaw
        delta_yaw = new_yaw - old_yaw

        if self.verbose():
            print(f"Rotation change: {math.degrees(delta_yaw):.2f} degrees")

        # Get image dimensions
        img_width, img_height, _ = self._workspace.img_shape()

        # Calculate old center in pixel coordinates
        old_center_u = int(self._u_rel_com * img_width)
        old_center_v = int(self._v_rel_com * img_height)

        # Calculate old bounding box corners in pixel coordinates
        old_u_min = int(self._u_rel_min * img_width)
        old_v_min = int(self._v_rel_min * img_height)
        old_u_max = int(self._u_rel_max * img_width)
        old_v_max = int(self._v_rel_max * img_height)

        # Rotate bounding box corners around the old center
        rotated_corners = self._rotate_bounding_box(
            old_u_min, old_v_min, old_u_max, old_v_max, old_center_u, old_center_v, delta_yaw
        )

        # Find new axis-aligned bounding box from rotated corners
        new_u_min = int(min(corner[0] for corner in rotated_corners))
        new_v_min = int(min(corner[1] for corner in rotated_corners))
        new_u_max = int(max(corner[0] for corner in rotated_corners))
        new_v_max = int(max(corner[1] for corner in rotated_corners))

        # Rotate mask if available
        rotated_mask = None
        if self._original_mask_8u is not None:
            rotated_mask = self._rotate_mask(self._original_mask_8u, delta_yaw, old_center_u, old_center_v)
            # Update the original mask with the rotated version for future rotations
            self._original_mask_8u = rotated_mask.copy()

        # Now we need to translate everything to the new position
        # Calculate the new center from pose_com
        # We need to find what relative coordinates correspond to the new world position
        new_center_u_rel, new_center_v_rel = self._world_to_rel_coordinates(pose_com)
        new_center_u = int(new_center_u_rel * img_width)
        new_center_v = int(new_center_v_rel * img_height)

        # Calculate translation offset
        translation_u = new_center_u - old_center_u
        translation_v = new_center_v - old_center_v

        if self.verbose():
            print(f"Translation: u={translation_u}, v={translation_v}")

        # Apply translation to bounding box
        new_u_min += translation_u
        new_v_min += translation_v
        new_u_max += translation_u
        new_v_max += translation_v

        # Translate mask if available
        if rotated_mask is not None:
            rotated_mask = self._translate_mask(rotated_mask, translation_u, translation_v)

        # Clamp to image boundaries
        new_u_min = max(0, min(new_u_min, img_width - 1))
        new_v_min = max(0, min(new_v_min, img_height - 1))
        new_u_max = max(0, min(new_u_max, img_width - 1))
        new_v_max = max(0, min(new_v_max, img_height - 1))

        if self.verbose():
            print(f"Old bbox: ({old_u_min}, {old_v_min}) to ({old_u_max}, {old_v_max})")
            print(f"New bbox: ({new_u_min}, {new_v_min}) to ({new_u_max}, {new_v_max})")

        # Reinitialize object properties with new bounding box and rotated mask
        self._init_object_properties(new_u_min, new_v_min, new_u_max, new_v_max, rotated_mask)

        # Override the calculated pose_com with the exact one provided
        # (to avoid floating point errors from forward-backward transformations)
        self._pose_com = pose_com

    # *** PUBLIC GET methods ***

    def get_workspace_id(self) -> str:
        return self.workspace().id()

    # *** PUBLIC methods ***

    def as_string_for_llm(self) -> str:
        """
        Formats object details as a string for use with language models.

        Returns:
            str: String containing object information, including label, world coordinates,
            dimensions, and size in square centimeters.
        """
        return f"""- '{self.label()}' at world coordinates [{self.x_com():.2f}, {self.y_com():.2f}] with a width of {
        self.width_m():.2f} meters, a height of {self.height_m():.2f} meters and a size of {
        self.size_m2()*10000:.2f} square centimeters."""

    def as_string_for_llm_lbl(self) -> str:
        """
        Formats object details in a line-by-line style, optimized for language model usage.
        Alternative to as_string_for_llm.

        Returns:
            str: String containing detailed object information, line by line.
        """
        return f"""- '{self.label()}' at world coordinates [{self.x_com():.2f}, {self.y_com():.2f}] with
    - width: {self.width_m():.2f} meters,
    - height: {self.height_m():.2f} meters and
    - size: {self.size_m2() * 10000:.2f} square centimeters."""

    def as_string_for_chat_window(self) -> str:
        """
        Formats object details for display in a chat interface.

        Returns:
            str: String describing the detected object, including its label,
            world coordinates, orientation, and dimensions in meters.
        """
        return f"""Detected a new object: {self.label()} at world coordinate ({self.x_com():.2f}, {
        self.y_com():.2f}) with orientation {self.gripper_rotation():.1f} rad and size {
        self.width_m():.2f} m x {self.height_m():.2f} m."""

    # *** NEUE METHODEN FÜR JSON-SERIALISIERUNG ***

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Object instance to a dictionary that can be JSON serialized.

        Returns:
            Dict[str, Any]: Dictionary representation of the object
        """
        return {
            "id": self.generate_object_id(),  # Generate unique ID
            "label": self._label,
            "workspace_id": self.get_workspace_id(),
            "timestamp": time.time(),
            # Position information
            "position": {
                "center_of_mass": {
                    "x": float(self.x_com()),
                    "y": float(self.y_com()),
                    "z": float(self._pose_com.z) if hasattr(self._pose_com, "z") else 0.0,
                },
                "center": {
                    "x": float(self.x_center()),
                    "y": float(self.y_center()),
                    "z": float(self._pose_center.z) if hasattr(self._pose_center, "z") else 0.0,
                },
            },
            # Relative coordinates in image
            "image_coordinates": {
                "center_rel": {"u": float(self._u_rel_o), "v": float(self._v_rel_o)},
                "center_of_mass_rel": {"u": float(self._u_rel_com), "v": float(self._v_rel_com)},
                "bounding_box_rel": {
                    "u_min": float(self._u_rel_min),
                    "v_min": float(self._v_rel_min),
                    "u_max": float(self._u_rel_max),
                    "v_max": float(self._v_rel_max),
                },
            },
            # Physical dimensions
            "dimensions": {
                "width_m": float(self._width_m),
                "height_m": float(self._height_m),
                "size_m2": float(self._size_m2),
            },
            # Orientation
            "gripper_rotation": float(self._gripper_rotation),
            # Additional metadata
            "confidence": getattr(self, "_confidence", 1.0),  # If you have confidence
            "class_id": getattr(self, "_class_id", 0),  # If you have class ID
        }

    def to_json(self) -> str:
        """
        Converts the Object instance to a JSON string.

        Returns:
            str: JSON representation of the object
        """
        return json.dumps(self.to_dict(), indent=2)

    def generate_object_id(self) -> str:
        """
        Generates a unique ID for the object based on its properties.

        Returns:
            str: Unique object identifier
        """
        import hashlib

        # import uuid

        # Option 1: Hash-based ID (deterministic)
        id_string = f"{self._label}_{self.x_com():.3f}_{self.y_com():.3f}_{time.time()}"
        return hashlib.md5(id_string.encode()).hexdigest()[:8]

        # Option 2: UUID-based ID (always unique)
        # return str(uuid.uuid4())[:8]

    @staticmethod
    def _deserialize_mask(mask_data: str, shape: Union[tuple, list], dtype: str = "uint8") -> np.ndarray:
        """
        Deserialize base64 string back to numpy mask.

        Args:
            mask_data: Base64 encoded mask string
            shape: Original shape of the mask (height, width) as tuple or list
            dtype: Data type of the mask (default: 'uint8')

        Returns:
            np.ndarray: Reconstructed mask array

        Raises:
            ValueError: If mask_data is invalid or shape doesn't match
        """
        try:
            # Convert list to tuple if necessary
            # print(shape, dtype)
            if isinstance(shape, list):
                shape = tuple(shape)

            mask_bytes = base64.b64decode(mask_data.encode("utf-8"))
            mask = np.frombuffer(mask_bytes, dtype=np.dtype(dtype))
            mask = mask.reshape(shape)
            return mask
        except Exception as e:
            raise ValueError(f"Failed to deserialize mask: {e}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any], workspace: "Workspace") -> Optional["Object"]:
        """
        Creates an Object instance from a dictionary.

        Args:
            data: Dictionary containing object data (from to_dict())
            workspace: Workspace instance

        Returns:
            Object: Reconstructed object instance or None if reconstruction fails
        """
        try:
            # Check if we have the new format (from to_dict) or old format (with bbox key)
            if "bbox" in data:
                # Old format - has bbox directly
                bbox = data["bbox"]
                u_min = bbox["x_min"]
                v_min = bbox["y_min"]
                u_max = bbox["x_max"]
                v_max = bbox["y_max"]
            elif "image_coordinates" in data and "bounding_box_rel" in data["image_coordinates"]:
                # New format - convert from relative coordinates
                bbox_rel = data["image_coordinates"]["bounding_box_rel"]
                img_shape = workspace.img_shape()

                u_min = int(bbox_rel["u_min"] * img_shape[0])
                v_min = int(bbox_rel["v_min"] * img_shape[1])
                u_max = int(bbox_rel["u_max"] * img_shape[0])
                v_max = int(bbox_rel["v_max"] * img_shape[1])
            else:
                raise KeyError("Missing bounding box information in dictionary")

            # Handle mask if present
            mask_8u = None
            if data.get("has_mask", False) and "mask_data" in data and "mask_shape" in data:
                mask_8u = cls._deserialize_mask(data["mask_data"], data["mask_shape"], data.get("mask_dtype", "uint8"))

            # Create object
            obj = cls(
                label=data["label"],
                u_min=u_min,
                v_min=v_min,
                u_max=u_max,
                v_max=v_max,
                mask_8u=mask_8u,
                workspace=workspace,
            )

            # Restore additional properties if needed
            if "confidence" in data:
                obj._confidence = data["confidence"]
            if "class_id" in data:
                obj._class_id = data["class_id"]

            return obj

        except Exception as e:
            print(f"Error reconstructing object from dict: {e}")
            return None

    @classmethod
    def from_json(cls, json_str: str, workspace: "Workspace") -> Optional["Object"]:
        """
        Creates an Object instance from a JSON string.

        Args:
            json_str: JSON string containing object data
            workspace: Workspace instance

        Returns:
            Object: Reconstructed object instance or None if reconstruction fails
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data, workspace)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return None

    # *** PUBLIC STATIC/CLASS GET methods ***

    @staticmethod
    def calc_width_height(pose_ul: "PoseObjectPNP", pose_lr: "PoseObjectPNP") -> tuple[float, float]:
        """
        Calculates the width and the height between the two PoseObjects in meters
        (basically the vertical distance and the horizontal distance).

        Args:
            pose_ul: PoseObject in the upper left corner of the workspace.
            pose_lr: PoseObject in the lower right corner of the workspace.

        Returns:
            tuple[float, float]: width (horizontal distance) and height (vertical distance)
        """
        # bei niryo ist es so, dass x-Achse des World-Koordinatensystems nach vorne geht und y- nach rechts
        # deshalb hier width berechnet aus y-Koordinate und height aus x-Koordinate
        # TODO: prüfen ob das auch für widowx gilt, also ein allgemeines Prinzip ist
        width_m = pose_ul.y - pose_lr.y
        height_m = pose_ul.x - pose_lr.x

        return width_m, height_m

    # *** PRIVATE methods ***

    def _world_to_rel_coordinates(self, pose: "PoseObjectPNP") -> tuple[float, float]:
        """
        Converts world coordinates to relative image coordinates.
        This is the inverse of transform_camera2world_coords.

        Since we don't have a direct inverse transformation, we'll use
        the workspace corners to interpolate.

        Args:
            pose (PoseObjectPNP): Pose in world coordinates

        Returns:
            tuple[float, float]: Relative coordinates (u_rel, v_rel)
        """
        # Get workspace corners in world coordinates
        ul = self._workspace.xy_ul_wc()  # (0, 0) in relative coords
        lr = self._workspace.xy_lr_wc()  # (1, 1) in relative coords

        # Linear interpolation to find relative coordinates
        # Assuming linear mapping between world and relative coordinates

        # For u_rel (horizontal): interpolate along x-axis
        x_range = lr.x - ul.x
        if abs(x_range) > 1e-6:
            u_rel = (pose.x - ul.x) / x_range
        else:
            u_rel = 0.5

        # For v_rel (vertical): interpolate along y-axis
        # Note: y increases to the right in world coords but v increases downward
        y_range = ul.y - lr.y  # ul.y should be > lr.y
        if abs(y_range) > 1e-6:
            v_rel = (ul.y - pose.y) / y_range
        else:
            v_rel = 0.5

        # Clamp to [0, 1]
        u_rel = max(0.0, min(1.0, u_rel))
        v_rel = max(0.0, min(1.0, v_rel))

        return u_rel, v_rel

    def _rotate_bounding_box(
        self, u_min: int, v_min: int, u_max: int, v_max: int, center_u: int, center_v: int, angle: float
    ) -> List[tuple[int, int]]:
        """
        Rotates the four corners of a bounding box around a center point.

        Args:
            u_min, v_min: Top-left corner
            u_max, v_max: Bottom-right corner
            center_u, center_v: Center of rotation
            angle: Rotation angle in radians (counter-clockwise)

        Returns:
            List of four rotated corner coordinates [(u, v), ...]
        """
        # Define the four corners
        corners = [
            (u_min, v_min),  # Top-left
            (u_max, v_min),  # Top-right
            (u_max, v_max),  # Bottom-right
            (u_min, v_max),  # Bottom-left
        ]

        # Rotate each corner
        rotated_corners = []
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        for u, v in corners:
            # Translate to origin
            u_translated = u - center_u
            v_translated = v - center_v

            # Rotate (counter-clockwise)
            u_rotated = u_translated * cos_angle - v_translated * sin_angle
            v_rotated = u_translated * sin_angle + v_translated * cos_angle

            # Translate back
            u_final = int(u_rotated + center_u)
            v_final = int(v_rotated + center_v)

            rotated_corners.append((u_final, v_final))

        return rotated_corners

    def _rotate_mask(self, mask: np.ndarray, angle: float, center_u: int, center_v: int) -> np.ndarray:
        """
        Rotates a segmentation mask around a center point.

        Args:
            mask (np.ndarray): Original mask (2D array, uint8)
            angle (float): Rotation angle in radians (counter-clockwise)
            center_u, center_v: Center of rotation in pixel coordinates

        Returns:
            np.ndarray: Rotated mask
        """
        if mask is None:
            return None

        # Convert angle to degrees for OpenCV (OpenCV uses clockwise rotation)
        angle_degrees = -math.degrees(angle)  # Negative because CV2 rotates clockwise

        # Get rotation matrix
        height, width = mask.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((center_u, center_v), angle_degrees, 1.0)

        # Rotate the mask
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (width, height), flags=cv2.INTER_NEAREST)

        return rotated_mask

    def _translate_mask(self, mask: np.ndarray, delta_u: int, delta_v: int) -> np.ndarray:
        """
        Translates a segmentation mask by a given offset.

        Args:
            mask (np.ndarray): Original mask (2D array, uint8)
            delta_u, delta_v: Translation offsets in pixels

        Returns:
            np.ndarray: Translated mask
        """
        if mask is None:
            return None

        # Create translation matrix
        height, width = mask.shape[:2]
        translation_matrix = np.float32([[1, 0, delta_u], [0, 1, delta_v]])

        # Translate the mask
        translated_mask = cv2.warpAffine(mask, translation_matrix, (width, height), flags=cv2.INTER_NEAREST)

        return translated_mask

    def _calc_size(self) -> None:
        """
        Calculates the object's area in square meters.

        If a segmentation mask is available, the size is derived from the mask; otherwise,
        it is computed as width * height using the bounding box dimensions.
        """
        area = self._calculate_largest_contour_area()

        if area == 0:  # then there is no segmentation mask or no contour was found
            self._size_m2 = self._width_m * self._height_m
        else:
            ratio_w, ratio_h = self._calc_size_of_pixel_in_m()
            (width, height, nchannels) = self._workspace.img_shape()
            area_img = width * height  # number of square pixels that the image has
            # print(area, ratio_w, area_img)
            # area of object in pixels / area of workspace in pixels *
            # * width of workspace in m * height of workspace in m
            self._size_m2 = float(area) / float(area_img) * ratio_w * ratio_h

    def _calc_size_of_pixel_in_m(self) -> tuple[float, float]:
        """
        Computes the physical size of a single pixel (in relative coordinates: 0...1) in meters.
        For the u-axis and for the v-axis (numbers should be the same, yes they are).
        It basically just returns the width and height of the workspace in meters. The other information is wrong.

        Returns:
            tuple[float, float]: Size of one pixel along the u-axis and v-axis in meters.
        """
        ratio_w = float(self._width_m / self._width)
        ratio_h = float(self._height_m / self._height)

        if self.verbose():
            print(ratio_h, ratio_w)

        return ratio_w, ratio_h

    def _update_width_height(self, width: int, height: int) -> None:
        """
        Updates the object's width and height using values derived from a segmentation mask.
        Call this method after you determined the segmentation mask of the object.

        Args:
            width (int): Width of the object in pixels.
            height (int): Height of the object in pixels.
        """
        ratio_w, ratio_h = self._calc_size_of_pixel_in_m()

        # print(ratio_h, ratio_w, width, height, self._height, self._width, self._height_m, self._width_m)

        # from pixel coordinates get relative coordinates
        self._width, self._height = self._calc_rel_coordinates(width, height)

        self._width_m = self._width * ratio_w
        self._height_m = self._height * ratio_h

        # print(self._height_m, self._width_m)

    def _calc_pose_from_uv_coords(self, u: int, v: int) -> tuple["PoseObjectPNP", float, float]:
        """
        Calculates the object's pose in world coordinates based on pixel coordinates.

        Args:
            u (int): Pixel u-coordinate.
            v (int): Pixel v-coordinate.

        Returns:
            tuple[PoseObjectPNP, float, float]: Pose in world coordinates and relative
            image coordinates (u_rel, v_rel).
        """
        u_rel, v_rel = self._calc_rel_coordinates(u, v)

        pose = self._workspace.transform_camera2world_coords(self._workspace.id(), u_rel, v_rel, self._gripper_rotation)

        return pose, u_rel, v_rel

    @log_start_end_cls()
    def _calc_width_height_m(self) -> None:
        """
        Calculates the object's physical width and height in meters using its bounding box
        in pixel coordinates and the workspace transformation.
        """
        if self.verbose():
            print(self._label, self._u_rel_min)

        pose_min = self._workspace.transform_camera2world_coords(self._workspace.id(), self._u_rel_min, self._v_rel_min)
        pose_max = self._workspace.transform_camera2world_coords(self._workspace.id(), self._u_rel_max, self._v_rel_max)

        if self.verbose():
            print(pose_min, pose_max)

        self._width_m, self._height_m = Object.calc_width_height(pose_min, pose_max)

    def _calc_rel_coordinates(self, u: int, v: int) -> tuple[float, float]:
        """
        Converts pixel coordinates to relative coordinates (0-1) based on workspace dimensions.

        Args:
            u (int): Pixel u-coordinate.
            v (int): Pixel v-coordinate.

        Returns:
            tuple[float, float]: Relative coordinates (u_rel, v_rel).
        """
        img_workspace_shape = self._workspace.img_shape()

        u_rel = float(u / img_workspace_shape[0])
        v_rel = float(v / img_workspace_shape[1])

        return u_rel, v_rel

    @log_start_end_cls()
    def _calc_largest_contour(self, mask_8u: np.ndarray) -> None:
        """
        Determine the largest contour in a 2D segmentation mask and set parameter _largest_contour to it.

        Args:
            mask_8u (np.ndarray): 2D segmentation mask as an 8-bit unsigned integer array.
        """
        if mask_8u.dtype != np.uint8:
            raise ValueError("Input mask must be an 8-bit unsigned integer array.")

        # Find contours
        contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._largest_contour = None
            if self.verbose():
                print("No contours found in mask")
            return  # Early return

        self._largest_contour = max(contours, key=cv2.contourArea)

    def _calculate_largest_contour_area(self) -> int:
        """
        Computes the area of the largest contour in the object's segmentation mask.

        Returns:
            int: Area of the largest contour in pixels, or 0 if no contours are found.
        """
        if self._largest_contour is None or len(self._largest_contour) == 0:
            return 0  # No contours found

        # Compute the area of each contour
        largest_area = cv2.contourArea(self._largest_contour)

        return int(largest_area)

    def _rotated_bounding_box(self) -> tuple[int, int]:
        """
        Computes the dimensions of the object's rotated bounding box.

        Returns:
            tuple[int, int]: Width and height of the rotated bounding box in pixels.
        """
        center, (width, height), theta = self._get_params_of_min_area_rect()

        if width == 0 and height == 0:
            return width, height

        # if self._largest_contour is None or len(self._largest_contour) == 0:
        #     return 0, 0  # No object detected

        # Get the minimum area rectangle
        # rect = cv2.minAreaRect(self._largest_contour)
        # (center, (width, height), angle) = rect

        # _width goes along y-axis and _height goes along x-axis. to keep it this way, it is important, that the
        # returned width and height go in the same direction as they used to be.
        if self._width > self._height:
            return max(width, height), min(width, height)
        else:
            return min(width, height), max(width, height)

    def _get_params_of_min_area_rect(self) -> tuple[tuple[int, int], tuple[int, int], float]:
        """
        Calculate the parameters of the minimum area rectangle around the largest contour.

        This method computes the minimum area rectangle that encloses the largest contour.
        The rectangle is defined by its center, dimensions (width and height), and rotation angle.

        Returns:
            tuple:
                - center (tuple[int, int]): The (x, y) coordinates of the rectangle's center.
                - dimensions (tuple[int, int]): The (width, height) of the rectangle.
                - theta (float): The rotation angle of the rectangle in degrees.

        If no contours are found, the method returns default values:
            - center: (0, 0)
            - dimensions: (0, 0)
            - theta: 0
        """
        # If there are no contours, handle this case:
        if self._largest_contour is None or len(self._largest_contour) == 0:
            print("No contours found!")

            return (0, 0), (0, 0), 0
        else:
            # Get the minimum area rectangle around the largest contour
            rect = cv2.minAreaRect(self._largest_contour)

            # rect returns a tuple (center, (width, height), angle)
            center, (width, height), theta = rect

            return (int(center[0]), int(center[1])), (int(width), int(height)), theta

    def _calc_gripper_orientation_from_segmentation_mask(self) -> tuple[float, tuple[int, int]]:
        """
        Determines the optimal orientation for the robot gripper based on the object's segmentation mask.

        Returns:
            tuple[float, tuple[int, int]]: Suggested gripper rotation (radians)
            and the center of the object in pixel coordinates.
        """
        gripper_rotation = 0

        center, (width, height), theta = self._get_params_of_min_area_rect()

        if not (width == 0 and height == 0):
            # print('theta:', theta)

            theta = math.radians(theta)  # convert degrees to rad

            # print('theta_rad:', theta)

            if width < height:
                yaw_rel = theta + math.pi / 2
            else:
                yaw_rel = theta

            yaw_rel += math.pi / 2

            yaw_rel = yaw_rel % (2 * math.pi)

            # print(theta, width, height, center, yaw_rel)

            gripper_rotation = yaw_rel

        return gripper_rotation, center

    # *** PRIVATE STATIC/CLASS methods ***

    @staticmethod
    def _calculate_center_of_mass(mask_8u: np.ndarray) -> Optional[tuple[float, float]]:
        """
        Calculates the center of mass of an object in a segmentation mask.

        Args:
            mask_8u (numpy.ndarray): A 2D segmentation mask of the object (0...255, uint8).

        Returns:
            tuple: (cx, cy), the pixel coordinates of the center of mass in (x, y) format.
                   Returns None if the mask does not contain any non-zero values.
        """
        # Ensure the mask is binary: Convert non-zero values to 1
        binary_mask = (mask_8u > 0).astype(np.uint8)

        # Get the indices of non-zero pixels
        non_zero_indices = np.nonzero(binary_mask)

        if len(non_zero_indices[0]) == 0:  # No object found in the mask
            return None

        # Calculate the center of mass
        cx = np.mean(non_zero_indices[1])  # Mean of column indices (x-coordinate)
        cy = np.mean(non_zero_indices[0])  # Mean of row indices (y-coordinate)

        return cx, cy

    # *** PUBLIC properties ***

    def label(self) -> str:
        """
        Returns the label of the object (e.g., "chocolate bar").

        Returns:
            str: Label of the object.
        """
        return self._label

    def uv_rel_o(self) -> tuple[float, float]:
        """

        Returns:
            center of object in relative coordinates (u,v) along in image of workspace from 0 to 1
        """
        return self._u_rel_o, self._v_rel_o

    def u_rel_o(self) -> float:
        """

        Returns:
            center of object in relative coordinates along u-axis in image of workspace from 0 to 1
        """
        return self._u_rel_o

    def v_rel_o(self) -> float:
        """

        Returns:
            center of object in relative coordinates along v-axis in image of workspace from 0 to 1
        """
        return self._v_rel_o

    def pose_center(self) -> "PoseObjectPNP":
        """

        Returns:
            PoseObjectPNP: center of the object
        """
        return self._pose_center

    def x_center(self) -> float:
        """

        Returns:
            float: the x-coordinate of the center of the object in meters.
        """
        return self._pose_center.x

    def y_center(self) -> float:
        """

        Returns:
            float: the y-coordinate of the center of the object in meters.
        """
        return self._pose_center.y

    def xy_center(self) -> tuple[float, float]:
        """

        Returns:
            tuple[float, float]: the (x, y)-coordinate of the center of the object in meters.
        """
        return self._pose_center.x, self._pose_center.y

    def pose_com(self) -> "PoseObjectPNP":
        """

        Returns:
            PoseObjectPNP: center of mass of the object
        """
        return self._pose_com

    def x_com(self) -> float:
        """

        Returns:
            float: the x-coordinate of the center of mass of the object in meters.
        """
        return self._pose_com.x

    def y_com(self) -> float:
        """

        Returns:
            float: the y-coordinate of the center of mass of the object in meters.
        """
        return self._pose_com.y

    def xy_com(self) -> tuple[float, float]:
        """

        Returns:
            tuple[float, float]: the (x, y)-coordinate of the center of mass of the object in meters.
        """
        return self._pose_com.x, self._pose_com.y

    def coordinate(self) -> List[float]:
        """
        Returns (x,y) world coordinates of the center of mass of the object, measured in meters.
        At these coordinates you can pick the object. The List contains two float values.

        Returns:
            List[float]: x,y world coordinates of the object. At these coordinates you can pick the object.
        """
        return [self._pose_com.x, self._pose_com.y]

    def shape_m(self) -> tuple[float, float]:
        """

        Returns:
            tuple[float, float]: width and height of the object measured in meters.
        """
        return self._width_m, self._height_m

    def width_m(self) -> float:
        """
        The width of the object is always measured along the y-axis of the world coordinate system of the robot.

        Returns:
            the width of the object in meters.
        """
        return self._width_m

    def height_m(self) -> float:
        """
        The height of the object is always measured along the x-axis of the world coordinate system of the robot.

        Returns:
            the height of the object in meters.
        """
        return self._height_m

    def size_m2(self) -> float:
        """
        Returns the size of the object in square meters.

        Returns:
            float: Object size in m².
        """
        return self._size_m2

    def largest_contour(self) -> np.ndarray:
        return self._largest_contour

    def gripper_rotation(self) -> float:
        """
        Returns the optimal rotation angle for the robot gripper to pick up the object.

        Returns:
            float: Rotation angle in radians.
        """
        return self._gripper_rotation

    def workspace(self) -> "Workspace":
        return self._workspace

    def verbose(self) -> bool:
        """

        Returns:
            True, if verbose is on, else False
        """
        return self._verbose

    # *** PRIVATE variables ***

    # label of object such as chocolate bar, apple, pen, ...
    _label = ""

    # center of object in relative coordinates in image of workspace from 0 to 1
    _u_rel_o = 0.0
    _v_rel_o = 0.0

    # PoseObject of center of object
    _pose_center = None

    # center of mass of object in relative coordinates in image of workspace from 0 to 1
    _u_rel_com = 0.0
    _v_rel_com = 0.0

    # PoseObject of center of mass
    _pose_com = None

    # dimension of bounding box in relative coordinates in image of workspace from 0 to 1
    _u_rel_min = 0.0
    _v_rel_min = 0.0
    _u_rel_max = 0.0
    _v_rel_max = 0.0

    # width and height of image in relative coordinates in image of workspace from 0 to 1
    _width = 0.0
    _height = 0.0

    # width and height in meter
    _width_m = 0.0
    _height_m = 0.0
    # height of object in z-coordinate. Could be estimated using monocular depth estimation
    _depth_m = 0.0

    # size of object in m^2. if segmentation mask available, then
    _size_m2 = 0.0

    _largest_contour = None

    # gripper rotation needed to pick this object
    _gripper_rotation = 0.0

    # workspace this object can be found in
    _workspace = None

    _original_mask_8u = None

    _verbose = False
