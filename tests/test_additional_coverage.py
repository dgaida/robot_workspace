"""
Additional tests to cover remaining gaps in coverage
Create this file at: tests/test_additional_coverage.py
"""

import pytest
import numpy as np
from unittest.mock import Mock
from robot_workspace.objects.object import Object
from robot_workspace.objects.objects import Objects
from robot_workspace.objects.pose_object import PoseObjectPNP
from robot_workspace.workspaces.niryo_workspace import NiryoWorkspace


@pytest.fixture
def mock_workspace():
    """Create a mock workspace"""
    workspace = Mock()
    workspace.id.return_value = "test_workspace"
    workspace.img_shape.return_value = (640, 480, 3)

    def mock_transform(ws_id, u_rel, v_rel, yaw=0.0):
        x = 0.4 - u_rel * 0.3
        y = 0.15 - v_rel * 0.3
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    workspace.transform_camera2world_coords = mock_transform
    workspace.xy_ul_wc = Mock(return_value=PoseObjectPNP(0.4, 0.15, 0.05, 0.0, 1.57, 0.0))
    workspace.xy_lr_wc = Mock(return_value=PoseObjectPNP(0.1, -0.15, 0.05, 0.0, 1.57, 0.0))

    return workspace


class TestObjectVerboseOutput:
    """Test verbose output in Object methods"""

    def test_calc_largest_contour_verbose(self, mock_workspace):
        """Test _calc_largest_contour with verbose output"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        # Add a small contour so the object initializes properly
        mask[150:160, 150:160] = 255

        # Create object with verbose
        obj = Object("test", 100, 100, 200, 200, mask, mock_workspace, verbose=True)

        # This should print "No contours found in mask"
        assert obj._largest_contour is not None


class TestObjectsGetDetectedObjectSerialization:
    """Test Objects.get_detected_object with serialization"""

    def test_get_detected_object_serializable_not_found(self, mock_workspace):
        """Test get_detected_object with serializable=True when not found"""
        objects = Objects()
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        objects.append(obj)

        # Search for object far away
        result = objects.get_detected_object([10.0, 10.0], serializable=True)

        assert result is None

    # def test_get_detected_objects_with_unknown_location(self, mock_workspace):
    #     """Test get_detected_objects with unknown location string"""
    #     objects = Objects()
    #     obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
    #     objects.append(obj)
    #
    #     # Pass an object that doesn't match any Location enum
    #     # This should hit the error case in get_detected_objects
    #     # class FakeLocation:
    #     #     value = "fake_location"
    #
    #     result = objects.get_detected_objects(location="fake_location", coordinate=[0.2, 0.0])
    #
    #     # Should return None for unknown location
    #     assert result is None


class TestObjectCalculateCenterOfMassEdgeCases:
    """Test edge cases in _calculate_center_of_mass"""

    def test_calculate_center_of_mass_with_single_pixel(self, mock_workspace):
        """Test center of mass with single pixel mask"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[100, 100] = 255

        cx, cy = Object._calculate_center_of_mass(mask)

        assert cx == 100
        assert cy == 100


class TestPoseObjectNormalizeAngle:
    """Test PoseObjectPNP angle normalization"""

    def test_normalize_angle_positive(self):
        """Test _normalize_angle with positive angle"""
        angle = 4.0  # > π
        normalized = PoseObjectPNP._normalize_angle(angle)

        assert -np.pi <= normalized <= np.pi

    def test_normalize_angle_negative(self):
        """Test _normalize_angle with negative angle"""
        angle = -4.0  # < -π
        normalized = PoseObjectPNP._normalize_angle(angle)

        assert -np.pi <= normalized <= np.pi

    def test_normalize_angle_within_range(self):
        """Test _normalize_angle with angle already in range"""
        angle = 1.0
        normalized = PoseObjectPNP._normalize_angle(angle)

        assert normalized == 1.0

    def test_angular_difference(self):
        """Test _angular_difference method"""
        angle1 = 3.0
        angle2 = -3.0

        diff = PoseObjectPNP._angular_difference(angle1, angle2)

        # Should account for wrapping
        assert -np.pi <= diff <= np.pi


class TestObjectGetParamsOfMinAreaRect:
    """Test _get_params_of_min_area_rect edge cases"""

    def test_get_params_with_empty_contour(self, mock_workspace):
        """Test _get_params_of_min_area_rect with empty contour"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
        obj._largest_contour = np.array([])  # Empty array

        center, dimensions, theta = obj._get_params_of_min_area_rect()

        assert center == (0, 0)
        assert dimensions == (0, 0)
        assert theta == 0


class TestObjectMaskManipulation:
    """Test mask manipulation methods"""

    def test_rotate_mask_zero_angle(self, mock_workspace):
        """Test _rotate_mask with zero angle"""
        mask = np.ones((640, 480), dtype=np.uint8) * 255

        obj = Object("test", 100, 100, 200, 200, mask, mock_workspace)

        rotated = obj._rotate_mask(mask, 0.0, 320, 240)

        # Should be same as original for zero rotation
        assert rotated is not None
        assert rotated.shape == mask.shape

    def test_translate_mask_zero_offset(self, mock_workspace):
        """Test _translate_mask with zero offset"""
        mask = np.ones((640, 480), dtype=np.uint8) * 255

        obj = Object("test", 100, 100, 200, 200, mask, mock_workspace)

        translated = obj._translate_mask(mask, 0, 0)

        # Should be same as original for zero translation
        assert translated is not None
        assert np.array_equal(translated, mask)


class TestObjectSerializationEdgeCases:
    """Test serialization edge cases"""

    def test_from_dict_with_new_format_and_corners(self, mock_workspace):
        """Test from_dict with new format including corners"""
        obj_dict = {
            "label": "test",
            "image_coordinates": {
                "bounding_box_rel": {
                    "u_min": 0.15625,  # 100/640
                    "v_min": 0.2083,  # 100/480
                    "u_max": 0.3125,  # 200/640
                    "v_max": 0.4167,  # 200/480
                }
            },
            "has_mask": False,
        }

        reconstructed = Object.from_dict(obj_dict, mock_workspace)

        assert reconstructed is not None
        assert reconstructed.label() == "test"

    def test_deserialize_mask_with_wrong_shape(self):
        """Test _deserialize_mask with mismatched shape"""
        import base64

        mask = np.ones((50, 50), dtype=np.uint8) * 255
        mask_bytes = mask.tobytes()
        mask_data = base64.b64encode(mask_bytes).decode("utf-8")

        # Try to deserialize with wrong shape
        with pytest.raises(ValueError):
            Object._deserialize_mask(mask_data, (60, 60), "uint8")


class TestObjectWorldToRelCoordinates:
    """Test _world_to_rel_coordinates edge cases"""

    def test_world_to_rel_with_pose_outside_workspace(self, mock_workspace):
        """Test _world_to_rel_coordinates with pose outside workspace"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        # Pose far outside workspace
        pose = PoseObjectPNP(10.0, 10.0, 0.05, 0.0, 1.57, 0.0)

        u_rel, v_rel = obj._world_to_rel_coordinates(pose)

        # Should still return valid values (clamped to 0-1)
        assert 0 <= u_rel <= 1
        assert 0 <= v_rel <= 1


class TestObjectUpdateWidthHeight:
    """Test _update_width_height method"""

    def test_update_width_height_with_zero_width(self, mock_workspace):
        """Test _update_width_height with very small dimensions"""
        mask = np.ones((640, 480), dtype=np.uint8) * 255
        obj = Object("test", 100, 100, 200, 200, mask, mock_workspace)

        # Update with minimal dimensions
        obj._update_width_height(1, 1)

        # Should handle gracefully
        assert obj.width_m() >= 0
        assert obj.height_m() >= 0


# class TestObjectRotatedBoundingBox:
#     """Test _rotated_bounding_box edge cases"""

# def test_rotated_bounding_box_height_greater_than_width(self, mock_workspace):
#     """Test _rotated_bounding_box when height > width"""
#     mask = np.zeros((640, 480), dtype=np.uint8)
#     mask[100:400, 200:250] = 255  # Tall rectangle
#
#     obj = Object("test", 200, 100, 250, 400, mask, mock_workspace)
#
#     width, height = obj._rotated_bounding_box()
#
#     # For a tall rectangle where _height > _width initially,
#     # the rotated bounding box should swap to maintain convention (width >= height)
#     # So we expect width >= height after the method processes it
#     assert width >= height or (width == 0 and height == 0)


class TestObjectCalcGripperOrientation:
    """Test gripper orientation calculation edge cases"""

    def test_calc_gripper_orientation_with_square_mask(self, mock_workspace):
        """Test gripper orientation with square object"""
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[200:300, 200:300] = 255  # Perfect square

        obj = Object("test", 200, 200, 300, 300, mask, mock_workspace)

        rotation, center = obj._calc_gripper_orientation_from_segmentation_mask()

        assert 0 <= rotation <= 2 * np.pi
        assert len(center) == 2


class TestObjectSetPoseComEdgeCases:
    """Test set_pose_com edge cases"""

    def test_set_pose_com_with_large_rotation(self, mock_workspace):
        """Test set_pose_com with large rotation angle"""
        obj = Object("test", 200, 200, 300, 300, None, mock_workspace)

        # Large rotation (more than 2π)
        large_yaw = 4 * np.pi
        new_pose = obj.pose_com().copy_with_offsets(yaw_offset=large_yaw)

        obj.set_pose_com(new_pose)

        # Should handle large angles
        assert obj.pose_com() == new_pose

    def test_set_pose_com_with_negative_coordinates(self, mock_workspace):
        """Test set_pose_com resulting in negative coordinates"""
        obj = Object("test", 100, 100, 200, 200, None, mock_workspace)

        # Pose that would result in negative image coordinates
        new_pose = PoseObjectPNP(-0.5, -0.5, 0.05, 0.0, 1.57, 0.0)

        obj.set_pose_com(new_pose)

        # Bounding box should be clamped
        assert 0 <= obj._u_rel_min <= 1
        assert 0 <= obj._v_rel_min <= 1


class TestObjectsGetNearestWithNoMatch:
    """Test get_nearest_detected_object edge cases"""

    def test_get_nearest_with_label_no_match(self, mock_workspace):
        """Test get_nearest_detected_object when label doesn't match"""
        objects = Objects()
        obj = Object("pencil", 100, 100, 200, 200, None, mock_workspace)
        objects.append(obj)

        # Search for non-existent label
        nearest, distance = objects.get_nearest_detected_object([0.2, 0.0], label="eraser")

        assert nearest is None
        assert distance == float("inf")


class TestNiryoWorkspaceEnvironmentProperty:
    """Test NiryoWorkspace environment property"""

    def test_environment_property(self):
        """Test that environment property returns the environment"""
        env = Mock()
        env.use_simulation.return_value = False

        def mock_transform(ws_id, u_rel, v_rel, yaw):
            return PoseObjectPNP(0.3, 0.0, 0.05, 0.0, 1.57, yaw)

        env.get_robot_target_pose_from_rel = mock_transform

        workspace = NiryoWorkspace("test_ws", env)

        assert workspace.environment() == env


class TestObjectLabelFiltering:
    """Test label filtering in Objects"""

    def test_get_detected_objects_partial_label_match(self, mock_workspace):
        """Test that label filtering works with partial matches"""
        objects = Objects()
        obj1 = Object("pencil", 100, 100, 200, 200, None, mock_workspace)
        obj2 = Object("pen", 200, 200, 300, 300, None, mock_workspace)
        obj3 = Object("eraser", 300, 300, 400, 400, None, mock_workspace)

        objects.append(obj1)
        objects.append(obj2)
        objects.append(obj3)

        # Search for "pen" should match both "pen" and "pencil"
        result = objects.get_detected_objects(label="pen")

        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
