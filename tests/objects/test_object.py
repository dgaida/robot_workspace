"""
Unit tests for Object class
"""

import pytest
import numpy as np
import json
import math
from unittest.mock import Mock
from robot_workspace.objects.object import Object
from robot_workspace.objects.pose_object import PoseObjectPNP
from robot_workspace import Location


class TestObjectAdvancedInitialization:
    """Advanced initialization tests"""

    def test_initialization_no_workspace_image_shape_raises_error(self):
        """Test that initialization fails if workspace has no image shape"""
        workspace = Mock()
        workspace.id.return_value = "test_workspace"
        workspace.img_shape.return_value = None

        with pytest.raises(ValueError, match="Object has no image shape"):
            Object("test", 100, 100, 200, 200, None, workspace)

    def test_initialization_with_verbose(self, mock_workspace):
        """Test initialization with verbose enabled"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace, verbose=True)
        assert obj.verbose() is True

    def test_original_mask_is_copied(self, mock_workspace):
        """Test that original mask is copied, not referenced"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[100:200, 100:200] = 255

        obj = Object("test", 100, 100, 200, 200, mask, mock_workspace)

        # Modify original mask
        mask[100:200, 100:200] = 0

        # Object should still have the copied mask
        assert obj._original_mask_8u is not None
        assert np.any(obj._original_mask_8u > 0)


class TestObjectSetPoseCom:
    """Tests for set_pose_com method (rotation and translation)"""

    def test_set_pose_com_with_rotation(self, mock_workspace):
        """Test set_pose_com with rotation"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[150:250, 150:250] = 255

        obj = Object("test", 100, 100, 300, 300, mask, mock_workspace)

        # Store original pose
        # original_pose = obj.pose_com()
        # original_yaw = obj.gripper_rotation()

        # Create new pose with rotation
        rotation_offset = math.pi / 4
        new_pose = obj.pose_com().copy_with_offsets(x_offset=0.05, y_offset=0.05, yaw_offset=rotation_offset)

        obj.set_pose_com(new_pose)

        # The rotation in the pose should match the new pose's yaw
        # The gripper_rotation is recalculated from the mask after rotation
        # So we check that the pose yaw was updated correctly
        assert abs(obj.pose_com().yaw - new_pose.yaw) < 0.01

        # Verify position changed
        assert abs(obj.x_com() - new_pose.x) < 0.01
        assert abs(obj.y_com() - new_pose.y) < 0.01

    def test_set_pose_com_without_mask(self, mock_workspace):
        """Test set_pose_com without segmentation mask"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        new_pose = obj.pose_com().copy_with_offsets(x_offset=0.1, y_offset=0.1)
        obj.set_pose_com(new_pose)

        # Should complete without error
        assert obj.pose_com() == new_pose

    def test_set_pose_com_with_verbose(self, mock_workspace):
        """Test set_pose_com with verbose output"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace, verbose=True)

        new_pose = obj.pose_com().copy_with_offsets(x_offset=0.1)

        # Should print verbose output (captured by pytest if needed)
        obj.set_pose_com(new_pose)

    def test_set_pose_com_boundary_clamping(self, mock_workspace):
        """Test that bounding box is clamped to image boundaries"""
        obj = Object("test", 500, 400, 600, 450, None, mock_workspace)

        # Create pose that would push bbox outside boundaries
        new_pose = PoseObjectPNP(0.5, 0.5, 0.05, 0.0, 1.57, 0.0)
        obj.set_pose_com(new_pose)

        # Bounding box should be within image boundaries
        assert 0 <= obj._u_rel_min <= 1
        assert 0 <= obj._v_rel_min <= 1
        assert 0 <= obj._u_rel_max <= 1
        assert 0 <= obj._v_rel_max <= 1


class TestObjectSetPosition:
    """Tests for legacy set_position method"""

    def test_set_position_legacy_method(self, mock_workspace):
        """Test legacy set_position method"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        original_z = obj.pose_com().z
        obj.set_position([0.3, 0.05])

        # Position should be updated
        assert abs(obj.x_com() - 0.3) < 0.01
        assert abs(obj.y_com() - 0.05) < 0.01
        # Z should remain unchanged
        assert obj.pose_com().z == original_z


class TestObjectStringFormatting:
    """Tests for different string formatting methods"""

    def test_as_string_for_llm_lbl(self, mock_workspace):
        """Test as_string_for_llm_lbl method"""
        obj = Object("test_object", 100, 100, 200, 200, None, mock_workspace)
        llm_lbl_str = obj.as_string_for_llm_lbl()

        assert "test_object" in llm_lbl_str
        assert "width:" in llm_lbl_str
        assert "height:" in llm_lbl_str
        assert "size:" in llm_lbl_str


class TestObjectSerialization:
    """Advanced serialization tests"""

    def test_to_dict_with_mask(self, mock_workspace):
        """Test to_dict with segmentation mask"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[100:200, 100:200] = 255

        obj = Object("masked_obj", 100, 100, 200, 200, mask, mock_workspace)
        obj_dict = obj.to_dict()

        assert obj_dict["label"] == "masked_obj"
        assert "dimensions" in obj_dict
        assert "position" in obj_dict

    def test_deserialize_mask_with_list_shape(self):
        """Test _deserialize_mask with list shape"""
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        import base64

        mask_bytes = mask.tobytes()
        mask_data = base64.b64encode(mask_bytes).decode("utf-8")

        # Test with list shape instead of tuple
        reconstructed = Object._deserialize_mask(mask_data, [50, 50], "uint8")

        assert reconstructed.shape == (50, 50)
        assert np.array_equal(reconstructed, mask)

    def test_deserialize_mask_invalid_data(self):
        """Test _deserialize_mask with invalid data"""
        with pytest.raises(ValueError, match="Failed to deserialize mask"):
            Object._deserialize_mask("invalid_base64", (50, 50), "uint8")

    def test_from_dict_with_old_format(self, mock_workspace):
        """Test from_dict with old format (bbox key)"""
        obj_dict = {
            "label": "test",
            "bbox": {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
            "has_mask": False,
        }

        reconstructed = Object.from_dict(obj_dict, mock_workspace)

        assert reconstructed is not None
        assert reconstructed.label() == "test"

    def test_from_dict_with_mask(self, mock_workspace):
        """Test from_dict with mask data"""
        # Create original object with mask
        mask = np.ones((640, 480), dtype=np.uint8) * 200
        Object("test", 100, 100, 200, 200, mask, mock_workspace)

        # Serialize with mask
        import base64

        mask_bytes = mask.tobytes()
        mask_data = base64.b64encode(mask_bytes).decode("utf-8")

        obj_dict = {
            "label": "test",
            "bbox": {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
            "has_mask": True,
            "mask_data": mask_data,
            "mask_shape": list(mask.shape),
            "mask_dtype": "uint8",
        }

        reconstructed = Object.from_dict(obj_dict, mock_workspace)

        assert reconstructed is not None
        assert reconstructed.label() == "test"

    def test_from_dict_missing_bbox_raises_error(self, mock_workspace):
        """Test from_dict with missing bounding box"""
        obj_dict = {"label": "test"}

        reconstructed = Object.from_dict(obj_dict, mock_workspace)

        # Should handle gracefully and return None
        assert reconstructed is None

    def test_from_dict_with_confidence_and_class_id(self, mock_workspace):
        """Test from_dict preserves confidence and class_id"""
        obj_dict = {
            "label": "test",
            "bbox": {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
            "has_mask": False,
            "confidence": 0.95,
            "class_id": 5,
        }

        reconstructed = Object.from_dict(obj_dict, mock_workspace)

        assert reconstructed is not None
        assert hasattr(reconstructed, "_confidence")
        assert reconstructed._confidence == 0.95
        assert hasattr(reconstructed, "_class_id")
        assert reconstructed._class_id == 5


class TestObjectMaskOperations:
    """Tests for mask-related operations"""

    def test_rotate_mask(self, mock_workspace):
        """Test _rotate_mask method"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[200:300, 200:300] = 255

        obj = Object("test", 100, 100, 400, 400, mask, mock_workspace)

        # Rotate mask by 45 degrees
        rotated = obj._rotate_mask(mask, math.pi / 4, 320, 240)

        assert rotated is not None
        assert rotated.shape == mask.shape

    def test_rotate_mask_with_none(self, mock_workspace):
        """Test _rotate_mask with None mask"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        result = obj._rotate_mask(None, math.pi / 4, 100, 100)

        assert result is None

    def test_translate_mask(self, mock_workspace):
        """Test _translate_mask method"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[100:200, 100:200] = 255

        obj = Object("test", 100, 100, 200, 200, mask, mock_workspace)

        # Translate mask by 50 pixels
        translated = obj._translate_mask(mask, 50, 50)

        assert translated is not None
        assert translated.shape == mask.shape

    def test_translate_mask_with_none(self, mock_workspace):
        """Test _translate_mask with None mask"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        result = obj._translate_mask(None, 50, 50)

        assert result is None

    def test_calc_largest_contour_with_invalid_mask_dtype(self, mock_workspace):
        """Test _calc_largest_contour raises error for invalid dtype"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        invalid_mask = np.zeros((640, 480), dtype=np.float32)

        with pytest.raises(ValueError, match="Input mask must be an 8-bit"):
            obj._calc_largest_contour(invalid_mask)

    def test_calc_largest_contour_no_contours(self, mock_workspace):
        """Test _calc_largest_contour with empty mask"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        empty_mask = np.zeros((640, 480), dtype=np.uint8)
        obj._calc_largest_contour(empty_mask)

        assert obj._largest_contour is None or len(obj._largest_contour) == 0

    def test_calculate_largest_contour_area_no_contours(self, mock_workspace):
        """Test _calculate_largest_contour_area with no contours"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        obj._largest_contour = None

        area = obj._calculate_largest_contour_area()

        assert area == 0

    def test_calculate_largest_contour_area_empty_contour(self, mock_workspace):
        """Test _calculate_largest_contour_area with empty contour"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        obj._largest_contour = np.array([])

        area = obj._calculate_largest_contour_area()

        assert area == 0


class TestObjectBoundingBoxRotation:
    """Tests for bounding box rotation"""

    def test_rotate_bounding_box(self, mock_workspace):
        """Test _rotate_bounding_box method"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        rotated_corners = obj._rotate_bounding_box(100, 100, 200, 200, 150, 150, math.pi / 4)

        assert len(rotated_corners) == 4
        assert all(isinstance(corner, tuple) for corner in rotated_corners)
        assert all(len(corner) == 2 for corner in rotated_corners)

    def test_rotate_bounding_box_90_degrees(self, mock_workspace):
        """Test bounding box rotation by 90 degrees"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        # Rotate by 90 degrees around center
        rotated_corners = obj._rotate_bounding_box(100, 100, 200, 200, 150, 150, math.pi / 2)

        # After 90° rotation, corners should have swapped positions
        assert len(rotated_corners) == 4


class TestObjectMinAreaRect:
    """Tests for minimum area rectangle calculations"""

    def test_get_params_of_min_area_rect_with_mask(self, mock_workspace):
        """Test _get_params_of_min_area_rect with valid mask"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[100:200, 100:300] = 255  # Rectangle

        obj = Object("test", 100, 100, 300, 200, mask, mock_workspace)

        center, dimensions, theta = obj._get_params_of_min_area_rect()

        assert center[0] > 0
        assert center[1] > 0
        assert dimensions[0] > 0
        assert dimensions[1] > 0

    def test_get_params_of_min_area_rect_no_contours(self, mock_workspace):
        """Test _get_params_of_min_area_rect with no contours"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        obj._largest_contour = None

        center, dimensions, theta = obj._get_params_of_min_area_rect()

        assert center == (0, 0)
        assert dimensions == (0, 0)
        assert theta == 0

    def test_rotated_bounding_box_no_mask(self, mock_workspace):
        """Test _rotated_bounding_box with no mask"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        width, height = obj._rotated_bounding_box()

        assert width == 0
        assert height == 0

    def test_rotated_bounding_box_width_greater_than_height(self, mock_workspace):
        """Test _rotated_bounding_box when width > height"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[200:250, 100:400] = 255  # Wide rectangle

        obj = Object("test", 100, 200, 400, 250, mask, mock_workspace)

        width, height = obj._rotated_bounding_box()

        # Should return max as width, min as height
        assert width >= height


class TestObjectCoordinateTransformations:
    """Tests for coordinate transformation methods"""

    def test_world_to_rel_coordinates(self, mock_workspace):
        """Test _world_to_rel_coordinates method"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        pose = PoseObjectPNP(0.25, 0.0, 0.05, 0.0, 1.57, 0.0)
        u_rel, v_rel = obj._world_to_rel_coordinates(pose)

        assert 0 <= u_rel <= 1
        assert 0 <= v_rel <= 1

    def test_world_to_rel_coordinates_with_zero_range(self, mock_workspace):
        """Test _world_to_rel_coordinates with zero range (edge case)"""
        # Create workspace with same corner coordinates
        workspace = Mock()
        workspace.id.return_value = "test"
        workspace.img_shape.return_value = (640, 480, 3)

        same_pose = PoseObjectPNP(0.3, 0.1, 0.05, 0.0, 1.57, 0.0)

        def mock_transform(ws_id, u_rel, v_rel, yaw=0.0):
            # Always return same position (zero range)
            return same_pose

        workspace.transform_camera2world_coords = mock_transform

        workspace.xy_ul_wc = Mock(return_value=same_pose)
        workspace.xy_lr_wc = Mock(return_value=same_pose)

        obj = Object("test", 100, 100, 200, 200, None, workspace)

        # Should handle zero range gracefully
        u_rel, v_rel = obj._world_to_rel_coordinates(same_pose)

        # Should be clamped to valid range (defaults to 0.5 when range is zero)
        assert 0 <= u_rel <= 1
        assert 0 <= v_rel <= 1
        # When range is zero, should default to center (0.5)
        assert abs(u_rel - 0.5) < 0.01
        assert abs(v_rel - 0.5) < 0.01


class TestObjectSizeCalculations:
    """Tests for size calculation methods"""

    def test_calc_size_with_mask(self, mock_workspace):
        """Test _calc_size with segmentation mask"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[100:300, 100:300] = 255  # 200x200 square

        obj = Object("test", 100, 100, 300, 300, mask, mock_workspace)

        assert obj.size_m2() > 0

    def test_calc_size_without_mask(self, mock_workspace):
        """Test _calc_size without segmentation mask"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        # Should calculate from bounding box
        assert obj.size_m2() > 0

    def test_calc_size_of_pixel_in_m_with_verbose(self, mock_workspace):
        """Test _calc_size_of_pixel_in_m with verbose output"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace, verbose=True)

        ratio_w, ratio_h = obj._calc_size_of_pixel_in_m()

        assert ratio_w > 0
        assert ratio_h > 0

    def test_update_width_height(self, mock_workspace):
        """Test _update_width_height method"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[100:200, 100:250] = 255

        obj = Object("test", 100, 100, 250, 200, mask, mock_workspace)

        original_width = obj.width_m()

        # Update with new dimensions
        obj._update_width_height(100, 150)

        # Dimensions should have changed
        assert obj.width_m() != original_width


class TestObjectGripperOrientation:
    """Tests for gripper orientation calculations"""

    def test_calc_gripper_orientation_with_vertical_object(self, mock_workspace):
        """Test gripper orientation for vertical object"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        # Vertical rectangle (height > width)
        mask[100:300, 200:250] = 255

        obj = Object("test", 200, 100, 250, 300, mask, mock_workspace)

        rotation, center = obj._calc_gripper_orientation_from_segmentation_mask()

        assert isinstance(rotation, float)
        assert 0 <= rotation <= 2 * math.pi
        assert len(center) == 2

    def test_calc_gripper_orientation_with_horizontal_object(self, mock_workspace):
        """Test gripper orientation for horizontal object"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        # Horizontal rectangle (width > height)
        mask[200:250, 100:300] = 255

        obj = Object("test", 100, 200, 300, 250, mask, mock_workspace)

        rotation, center = obj._calc_gripper_orientation_from_segmentation_mask()

        assert isinstance(rotation, float)
        assert 0 <= rotation <= 2 * math.pi


class TestObjectEdgeCases:
    """Tests for edge cases and error handling"""

    def test_object_with_very_small_bounding_box(self, mock_workspace):
        """Test object with minimal bounding box"""
        obj = Object("tiny", 100, 100, 101, 101, None, mock_workspace)

        assert obj.width_m() > 0
        assert obj.height_m() > 0
        assert obj.size_m2() > 0

    def test_object_at_image_boundaries(self, mock_workspace):
        """Test object at the boundaries of the image"""
        obj = Object("boundary", 0, 0, 10, 10, None, mock_workspace)

        assert obj.u_rel_o() >= 0
        assert obj.v_rel_o() >= 0

    def test_generate_object_id_uniqueness(self, mock_workspace):
        """Test that object IDs are unique"""
        obj1 = Object("test", 100, 100, 200, 200, None, mock_workspace)
        obj2 = Object("test", 150, 150, 250, 250, None, mock_workspace)

        id1 = obj1.generate_object_id()
        id2 = obj2.generate_object_id()

        # IDs should be different (different positions/time)
        assert id1 != id2


class TestObjectProperties:
    """Tests for property accessors"""

    def test_all_relative_coordinate_properties(self, mock_workspace):
        """Test all relative coordinate properties"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        # Test all UV properties
        assert isinstance(obj.u_rel_o(), float)
        assert isinstance(obj.v_rel_o(), float)
        assert isinstance(obj.uv_rel_o(), tuple)
        assert len(obj.uv_rel_o()) == 2

    def test_all_pose_properties(self, mock_workspace):
        """Test all pose-related properties"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        # Center pose
        assert isinstance(obj.pose_center(), PoseObjectPNP)
        assert isinstance(obj.x_center(), float)
        assert isinstance(obj.y_center(), float)
        assert isinstance(obj.xy_center(), tuple)

        # COM pose
        assert isinstance(obj.pose_com(), PoseObjectPNP)
        assert isinstance(obj.x_com(), float)
        assert isinstance(obj.y_com(), float)
        assert isinstance(obj.xy_com(), tuple)

    def test_all_dimension_properties(self, mock_workspace):
        """Test all dimension properties"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        assert isinstance(obj.shape_m(), tuple)
        assert len(obj.shape_m()) == 2
        assert isinstance(obj.width_m(), float)
        assert isinstance(obj.height_m(), float)
        assert isinstance(obj.size_m2(), float)

        assert obj.width_m() > 0
        assert obj.height_m() > 0
        assert obj.size_m2() > 0


class TestLocation:
    """Test suite for Location enum"""

    def test_location_values(self):
        """Test that all location values are correct"""
        assert Location.LEFT_NEXT_TO.value == "left next to"
        assert Location.RIGHT_NEXT_TO.value == "right next to"
        assert Location.ABOVE.value == "above"
        assert Location.BELOW.value == "below"
        assert Location.ON_TOP_OF.value == "on top of"
        assert Location.INSIDE.value == "inside"
        assert Location.CLOSE_TO.value == "close to"
        assert Location.NONE.value is None

    def test_convert_str2location_with_string(self):
        """Test string to Location conversion"""
        assert Location.convert_str2location("left next to") == Location.LEFT_NEXT_TO
        assert Location.convert_str2location("right next to") == Location.RIGHT_NEXT_TO
        assert Location.convert_str2location("above") == Location.ABOVE
        assert Location.convert_str2location("below") == Location.BELOW
        assert Location.convert_str2location("on top of") == Location.ON_TOP_OF
        assert Location.convert_str2location("inside") == Location.INSIDE
        assert Location.convert_str2location("close to") == Location.CLOSE_TO

    def test_convert_str2location_with_enum(self):
        """Test Location to Location conversion (identity)"""
        assert Location.convert_str2location(Location.LEFT_NEXT_TO) == Location.LEFT_NEXT_TO
        assert Location.convert_str2location(Location.RIGHT_NEXT_TO) == Location.RIGHT_NEXT_TO

    def test_convert_str2location_with_none(self):
        """Test None to Location.NONE conversion"""
        assert Location.convert_str2location(None) == Location.NONE

    def test_convert_str2location_invalid_string(self):
        """Test invalid string raises ValueError"""
        with pytest.raises(ValueError, match="Invalid location string"):
            Location.convert_str2location("invalid location")

    def test_convert_str2location_invalid_type(self):
        """Test invalid type raises TypeError"""
        with pytest.raises(TypeError, match="Location must be either a string or a Location enum"):
            Location.convert_str2location(123)

    def test_all_locations_have_unique_values(self):
        """Test that all location values are unique"""
        values = [loc.value for loc in Location if loc != Location.NONE]
        assert len(values) == len(set(values))

    def test_location_enumeration(self):
        """Test iterating over Location enum"""
        locations = list(Location)
        assert len(locations) == 8  # All defined locations
        assert Location.LEFT_NEXT_TO in locations
        assert Location.NONE in locations


@pytest.fixture
def mock_workspace():
    """Create a mock workspace"""
    workspace = Mock()
    workspace.id.return_value = "test_workspace"
    workspace.img_shape.return_value = (640, 480, 3)

    # Mock transform method
    # For Niryo: width goes along y-axis, height along x-axis
    # Upper-left should have higher x and higher y than lower-right
    def mock_transform(ws_id, u_rel, v_rel, yaw=0.0):
        # Map relative coords to world coords
        # u_rel increases downward (0 to 1), x should decrease (higher to lower)
        # v_rel increases rightward (0 to 1), y should decrease (higher to lower)
        x = 0.4 - u_rel * 0.3  # x decreases as u increases
        y = 0.15 - v_rel * 0.3  # y decreases as v increases
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    workspace.transform_camera2world_coords = mock_transform

    # Mock workspace corner methods with actual PoseObjectPNP objects
    workspace.xy_ul_wc = Mock(return_value=PoseObjectPNP(0.4, 0.15, 0.05, 0.0, 1.57, 0.0))
    workspace.xy_lr_wc = Mock(return_value=PoseObjectPNP(0.1, -0.15, 0.05, 0.0, 1.57, 0.0))

    return workspace


class TestObject:
    """Test suite for Object class"""

    def test_initialization_without_mask(self, mock_workspace):
        """Test object initialization without segmentation mask"""
        obj = Object(label="test_object", u_min=100, v_min=100, u_max=200, v_max=200, mask_8u=None, workspace=mock_workspace)

        assert obj.label() == "test_object"
        assert obj.workspace() == mock_workspace

    def test_initialization_with_mask(self, mock_workspace):
        """Test object initialization with segmentation mask"""
        # Create a simple rectangular mask
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[100:200, 100:200] = 255

        obj = Object(label="masked_object", u_min=100, v_min=100, u_max=200, v_max=200, mask_8u=mask, workspace=mock_workspace)

        assert obj.label() == "masked_object"
        assert obj.largest_contour() is not None

    def test_label_property(self, mock_workspace):
        """Test label property"""
        obj = Object("pencil", 10, 10, 50, 50, None, mock_workspace)
        assert obj.label() == "pencil"

    def test_coordinate_property(self, mock_workspace):
        """Test coordinate property returns [x, y]"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        coords = obj.coordinate()

        assert isinstance(coords, list)
        assert len(coords) == 2
        assert isinstance(coords[0], float)
        assert isinstance(coords[1], float)

    def test_xy_com(self, mock_workspace):
        """Test center of mass coordinates"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        x, y = obj.xy_com()

        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_xy_center(self, mock_workspace):
        """Test center coordinates"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        x, y = obj.xy_center()

        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_shape_m(self, mock_workspace):
        """Test shape in meters"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        width, height = obj.shape_m()

        assert width > 0
        assert height > 0

    def test_size_m2(self, mock_workspace):
        """Test size in square meters"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        size = obj.size_m2()

        assert size > 0
        assert isinstance(size, float)

    def test_gripper_rotation(self, mock_workspace):
        """Test gripper rotation"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        rotation = obj.gripper_rotation()

        assert isinstance(rotation, float)
        assert 0 <= rotation <= 2 * np.pi

    def test_to_dict(self, mock_workspace):
        """Test conversion to dictionary"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        obj_dict = obj.to_dict()

        assert isinstance(obj_dict, dict)
        assert obj_dict["label"] == "test"
        assert "position" in obj_dict
        assert "dimensions" in obj_dict
        assert "workspace_id" in obj_dict

    def test_to_json(self, mock_workspace):
        """Test conversion to JSON"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        json_str = obj.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["label"] == "test"

    def test_from_dict(self, mock_workspace):
        """Test reconstruction from dictionary"""
        # Create original object
        original = Object("test", 100, 100, 200, 200, None, mock_workspace)
        obj_dict = original.to_dict()

        # Add image_coordinates for compatibility with from_dict
        # The from_dict expects bbox in image_coordinates
        if "image_coordinates" in obj_dict and "bounding_box_rel" in obj_dict["image_coordinates"]:
            bbox_rel = obj_dict["image_coordinates"]["bounding_box_rel"]
            img_shape = mock_workspace.img_shape()
            obj_dict["bbox"] = {
                "x_min": int(bbox_rel["u_min"] * img_shape[0]),
                "y_min": int(bbox_rel["v_min"] * img_shape[1]),
                "x_max": int(bbox_rel["u_max"] * img_shape[0]),
                "y_max": int(bbox_rel["v_max"] * img_shape[1]),
            }
            obj_dict["has_mask"] = False

        # Reconstruct
        reconstructed = Object.from_dict(obj_dict, mock_workspace)

        assert reconstructed is not None
        assert reconstructed.label() == original.label()

    def test_as_string_for_llm(self, mock_workspace):
        """Test LLM string formatting"""
        obj = Object("test_object", 100, 100, 200, 200, None, mock_workspace)
        llm_str = obj.as_string_for_llm()

        assert "test_object" in llm_str
        assert "meters" in llm_str
        assert "centimeters" in llm_str

    def test_as_string_for_chat_window(self, mock_workspace):
        """Test chat window string formatting"""
        obj = Object("test_object", 100, 100, 200, 200, None, mock_workspace)
        chat_str = obj.as_string_for_chat_window()

        assert "Detected" in chat_str
        assert "test_object" in chat_str

    def test_calc_width_height_static(self):
        """Test static width/height calculation"""
        pose_ul = PoseObjectPNP(0.3, 0.2, 0.0, 0.0, 0.0, 0.0)
        pose_lr = PoseObjectPNP(0.1, -0.1, 0.0, 0.0, 0.0, 0.0)

        width, height = Object.calc_width_height(pose_ul, pose_lr)

        assert abs(width - 0.3) < 0.0001  # y_ul - y_lr
        assert abs(height - 0.2) < 0.0001  # x_ul - x_lr

    def test_invalid_mask_dtype(self, mock_workspace):
        """Test that invalid mask dtype raises error"""
        invalid_mask = np.zeros((640, 480), dtype=np.float32)

        with pytest.raises(ValueError):
            Object("test", 100, 100, 200, 200, invalid_mask, mock_workspace)

    def test_uv_rel_o(self, mock_workspace):
        """Test relative UV coordinates"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        u, v = obj.uv_rel_o()

        assert 0 <= u <= 1
        assert 0 <= v <= 1

    def test_pose_center(self, mock_workspace):
        """Test pose_center returns PoseObjectPNP"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        pose = obj.pose_center()

        assert isinstance(pose, PoseObjectPNP)

    def test_pose_com(self, mock_workspace):
        """Test pose_com returns PoseObjectPNP"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        pose = obj.pose_com()

        assert isinstance(pose, PoseObjectPNP)

    def test_workspace_property(self, mock_workspace):
        """Test workspace property"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        assert obj.workspace() == mock_workspace

    def test_get_workspace_id(self, mock_workspace):
        """Test getting workspace ID"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        ws_id = obj.get_workspace_id()

        assert ws_id == "test_workspace"

    def test_str_representation(self, mock_workspace):
        """Test string representation"""
        obj = Object("test_object", 100, 100, 200, 200, None, mock_workspace)
        str_repr = str(obj)

        assert "test_object" in str_repr

    def test_repr_equals_str(self, mock_workspace):
        """Test that repr equals str"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        assert repr(obj) == str(obj)

    def test_calculate_center_of_mass_static(self):
        """Test static center of mass calculation"""
        # Create a simple mask with uniform distribution
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        cx, cy = Object._calculate_center_of_mass(mask)

        # Center should be around (50, 50)
        assert 48 < cx < 52
        assert 48 < cy < 52

    def test_calculate_center_of_mass_empty_mask(self):
        """Test center of mass with empty mask"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = Object._calculate_center_of_mass(mask)

        assert result is None

    def test_json_roundtrip(self, mock_workspace):
        """Test JSON serialization roundtrip"""
        original = Object("roundtrip_test", 100, 100, 200, 200, None, mock_workspace)

        # Convert to JSON and back - need to add bbox
        json_str = original.to_json()
        obj_dict = json.loads(json_str)

        # Add bbox for from_dict compatibility
        if "image_coordinates" in obj_dict and "bounding_box_rel" in obj_dict["image_coordinates"]:
            bbox_rel = obj_dict["image_coordinates"]["bounding_box_rel"]
            img_shape = mock_workspace.img_shape()
            obj_dict["bbox"] = {
                "x_min": int(bbox_rel["u_min"] * img_shape[0]),
                "y_min": int(bbox_rel["v_min"] * img_shape[1]),
                "x_max": int(bbox_rel["u_max"] * img_shape[0]),
                "y_max": int(bbox_rel["v_max"] * img_shape[1]),
            }
            obj_dict["has_mask"] = False

        reconstructed = Object.from_dict(obj_dict, mock_workspace)

        assert reconstructed is not None
        assert reconstructed.label() == original.label()

    def test_from_json_invalid(self, mock_workspace):
        """Test from_json with invalid JSON"""
        result = Object.from_json("invalid json", mock_workspace)
        assert result is None

    def test_generate_object_id(self, mock_workspace):
        """Test object ID generation"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        obj_id = obj.generate_object_id()

        assert isinstance(obj_id, str)
        assert len(obj_id) == 8  # MD5 hash truncated to 8 chars


class TestLocationIntegration:
    """Integration tests for Location usage"""

    def test_location_in_dict(self):
        """Test that Location can be used as dict key"""
        loc_dict = {Location.LEFT_NEXT_TO: "left", Location.RIGHT_NEXT_TO: "right"}

        assert loc_dict[Location.LEFT_NEXT_TO] == "left"
        assert loc_dict[Location.RIGHT_NEXT_TO] == "right"

    def test_location_comparison(self):
        """Test Location comparison"""
        loc1 = Location.LEFT_NEXT_TO
        loc2 = Location.LEFT_NEXT_TO
        loc3 = Location.RIGHT_NEXT_TO

        assert loc1 == loc2
        assert loc1 != loc3
        assert loc1 is loc2  # Same enum instance

    def test_location_string_representation(self):
        """Test string representation of Location"""
        loc = Location.LEFT_NEXT_TO

        # The name property
        assert loc.name == "LEFT_NEXT_TO"
        # The value property
        assert loc.value == "left next to"

    def test_location_in_set(self):
        """Test that Location can be used in sets"""
        locations = {Location.LEFT_NEXT_TO, Location.RIGHT_NEXT_TO, Location.LEFT_NEXT_TO}

        # Should deduplicate
        assert len(locations) == 2
        assert Location.LEFT_NEXT_TO in locations

    def test_location_switch_pattern(self):
        """Test switch-like pattern with Location"""

        def get_offset(location: Location):
            if location == Location.LEFT_NEXT_TO:
                return (0, 1)
            elif location == Location.RIGHT_NEXT_TO:
                return (0, -1)
            elif location == Location.ABOVE:
                return (1, 0)
            elif location == Location.BELOW:
                return (-1, 0)
            else:
                return (0, 0)

        assert get_offset(Location.LEFT_NEXT_TO) == (0, 1)
        assert get_offset(Location.RIGHT_NEXT_TO) == (0, -1)
        assert get_offset(Location.ABOVE) == (1, 0)
        assert get_offset(Location.BELOW) == (-1, 0)
        assert get_offset(Location.NONE) == (0, 0)

    # def test_location_with_match_statement(self):
    #     """Test Location with match statement (Python 3.10+)"""
    #     import sys
    #
    #     if sys.version_info >= (3, 10):
    #
    #         def describe_location(loc: Location) -> str:
    #             match loc:
    #                 case Location.LEFT_NEXT_TO:
    #                     return "to the left"
    #                 case Location.RIGHT_NEXT_TO:
    #                     return "to the right"
    #                 case Location.ABOVE:
    #                     return "above"
    #                 case Location.BELOW:
    #                     return "below"
    #                 case _:
    #                     return "somewhere"
    #
    #         assert describe_location(Location.LEFT_NEXT_TO) == "to the left"
    #         assert describe_location(Location.NONE) == "somewhere"

    def test_location_without_match_statement(self):
        """Test Location without match statement (Python 3.9 compatible)"""

        # This version works for all Python versions, including <3.10
        def describe_location(loc: Location) -> str:
            if loc == Location.LEFT_NEXT_TO:
                return "to the left"
            elif loc == Location.RIGHT_NEXT_TO:
                return "to the right"
            elif loc == Location.ABOVE:
                return "above"
            elif loc == Location.BELOW:
                return "below"
            else:
                return "somewhere"

        assert describe_location(Location.LEFT_NEXT_TO) == "to the left"
        assert describe_location(Location.NONE) == "somewhere"
