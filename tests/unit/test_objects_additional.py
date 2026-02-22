"""
Additional unit tests for objects to increase coverage.
"""

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from robot_workspace.objects.object import Object
from robot_workspace.objects.objects import Location, Objects
from robot_workspace.objects.pose_object import PoseObjectPNP


@pytest.fixture
def mock_workspace():
    ws = Mock()
    ws.id.return_value = "test_ws"
    ws.img_shape.return_value = (640, 480, 3)
    ws.xy_ul_wc.return_value = PoseObjectPNP(1.0, 1.0, 0, 0, 0, 0)
    ws.xy_lr_wc.return_value = PoseObjectPNP(0.0, 0.0, 0, 0, 0, 0)
    ws.transform_camera2world_coords.return_value = PoseObjectPNP(0.5, 0.5, 0.05, 0, 0, 0)
    return ws


def test_object_center_of_mass_fallback(mock_workspace):
    """Test Object center of mass fallback when mask is empty or CM cannot be calculated."""
    # An all-zero mask will make _calculate_center_of_mass return None
    mask = np.zeros((10, 10), dtype=np.uint8)
    obj = Object("cube", 0, 0, 10, 10, mask, mock_workspace)
    # Should fall back to center of bbox
    assert obj.pose_com() is not None


def test_object_from_dict_missing_img_shape(mock_workspace):
    """Test Object.from_dict when workspace img_shape is None."""
    mock_workspace.img_shape.return_value = None
    data = {
        "label": "cube",
        "image_coordinates": {"bounding_box_rel": {"u_min": 0.1, "v_min": 0.1, "u_max": 0.2, "v_max": 0.2}},
    }
    # Object.from_dict returns None on error
    assert Object.from_dict(data, mock_workspace) is None


def test_object_from_json(mock_workspace):
    """Test Object.from_json success and failure."""
    valid_json = json.dumps(
        {"label": "cube", "image_coordinates": {"bounding_box_rel": {"u_min": 0, "v_min": 0, "u_max": 0.1, "v_max": 0.1}}}
    )
    obj = Object.from_json(valid_json, mock_workspace)
    assert obj is not None
    assert obj.label() == "cube"

    invalid_json = "{ invalid json }"
    assert Object.from_json(invalid_json, mock_workspace) is None


def test_object_world_to_rel_coordinates_no_corners(mock_workspace):
    """Test _world_to_rel_coordinates when workspace corners are None."""
    obj = Object("cube", 0, 0, 10, 10, None, mock_workspace)
    mock_workspace.xy_ul_wc.return_value = None
    u, v = obj._world_to_rel_coordinates(PoseObjectPNP(0, 0, 0, 0, 0, 0))
    assert u == 0.5
    assert v == 0.5


def test_object_calc_size_no_img_shape(mock_workspace):
    """Test _calc_size when workspace img_shape is None."""
    mask = np.ones((10, 10), dtype=np.uint8)
    obj = Object("cube", 0, 0, 10, 10, mask, mock_workspace)

    # Now set it to None and call _calc_size
    mock_workspace.img_shape.return_value = None
    obj._calc_size()
    assert obj.size_m2() == 0.0


def test_object_calc_size_of_pixel_in_m_zero_dim(mock_workspace):
    """Test _calc_size_of_pixel_in_m when dimensions are zero."""
    obj = Object("cube", 0, 0, 10, 10, None, mock_workspace)
    obj._width = 0
    assert obj._calc_size_of_pixel_in_m() == (0.0, 0.0)


def test_object_calc_rel_coordinates_edge_cases(mock_workspace):
    """Test _calc_rel_coordinates with None or zero img_shape."""
    obj = Object("cube", 0, 0, 10, 10, None, mock_workspace)

    mock_workspace.img_shape.return_value = None
    assert obj._calc_rel_coordinates(100, 100) == (0.5, 0.5)

    mock_workspace.img_shape.return_value = (0, 0, 3)
    with pytest.raises(ValueError, match="Invalid workspace image shape"):
        obj._calc_rel_coordinates(100, 100)


def test_object_poses_not_initialized(mock_workspace):
    """Test pose_center and pose_com when not initialized."""
    obj = Object("cube", 0, 0, 10, 10, None, mock_workspace)
    obj._pose_center = None
    with pytest.raises(ValueError, match="Pose center not initialized"):
        obj.pose_center()

    obj._pose_com = None
    with pytest.raises(ValueError, match="Pose CoM not initialized"):
        obj.pose_com()


def test_objects_filter_missing_coordinate():
    """Test Objects.get_detected_objects missing coordinate for location filter."""
    objs = Objects()
    with pytest.raises(ValueError, match="Coordinate must be provided"):
        objs.get_detected_objects(location=Location.LEFT_NEXT_TO, coordinate=None)


def test_objects_filter_unknown_location():
    """Test Objects.get_detected_objects with unknown location."""
    objs = Objects()
    with patch("builtins.print") as mock_print, patch(
        "robot_workspace.objects.objects.Location.convert_str2location"
    ) as mock_conv:
        mock_conv.return_value = "UNKNOWN"
        result = objs.get_detected_objects(location="SOME_STR", coordinate=(0, 0))
        assert len(result) == 0
        mock_print.assert_called()


def test_pose_object_eq_not_implemented():
    """Test PoseObjectPNP.__eq__ with non-PoseObject."""
    pose = PoseObjectPNP(0, 0, 0, 0, 0, 0)
    assert pose.__eq__("not a pose") == NotImplemented


def test_pose_object_niryo_conversions():
    """Test PoseObjectPNP conversions to/from Niryo."""
    # Mock Niryo PoseObject
    mock_niryo_pose = Mock()
    mock_niryo_pose.x = 1.0
    mock_niryo_pose.y = 2.0
    mock_niryo_pose.z = 3.0
    mock_niryo_pose.roll = 0.1
    mock_niryo_pose.pitch = 0.2
    mock_niryo_pose.yaw = 0.3

    # Test convert_niryo_pose_object2pose_object
    pose = PoseObjectPNP.convert_niryo_pose_object2pose_object(mock_niryo_pose)
    assert pose.x == 1.0
    assert pose.z == 3.0

    # Test convert_pose_object2niryo_pose_object
    with patch("robot_workspace.objects.pose_object.PoseObject") as mock_pose_cls:
        PoseObjectPNP.convert_pose_object2niryo_pose_object(pose)
        mock_pose_cls.assert_called_with(1.0, 2.0, 3.0, pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3))
