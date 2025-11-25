"""
Unit tests for Object class
"""

import pytest
import numpy as np
import json
from unittest.mock import Mock
from robot_workspace.objects.object import Object
from robot_workspace.objects.pose_object import PoseObjectPNP
from robot_workspace import Location


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

    # Mock transform method - FIXED coordinate system
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
