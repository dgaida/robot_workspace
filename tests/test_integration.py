"""
Integration tests for robot_workspace package.

These tests verify end-to-end workflows and interactions between multiple components:
- Object creation and workspace integration
- Spatial queries with real coordinate transformations
- Serialization roundtrips
- Multi-workspace operations
- Pick-and-place scenarios
"""

import pytest
import numpy as np
import json
import math
from unittest.mock import Mock
from robot_workspace.objects.object import Object
from robot_workspace.objects.objects import Objects
from robot_workspace.objects.pose_object import PoseObjectPNP
from robot_workspace.objects.object_api import Location
from robot_workspace.workspaces.niryo_workspaces import NiryoWorkspaces


@pytest.fixture
def mock_environment():
    """Create a comprehensive mock environment for integration tests."""
    env = Mock()
    env.use_simulation.return_value = False
    env.verbose.return_value = False

    def mock_transform(ws_id, u_rel, v_rel, yaw):
        """Realistic coordinate transformation for Niryo robot."""
        # For Niryo: width along y-axis, height along x-axis
        x = 0.4 - u_rel * 0.3  # x: 0.4 to 0.1
        y = 0.15 - v_rel * 0.3  # y: 0.15 to -0.15
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    env.get_robot_target_pose_from_rel = mock_transform
    return env


@pytest.fixture
def workspace_with_objects(mock_environment):
    """Create a workspace with multiple objects for testing."""
    workspaces = NiryoWorkspaces(mock_environment, verbose=False)
    workspace = workspaces.get_home_workspace()
    workspace.set_img_shape((640, 480, 3))

    # Create diverse objects
    objects = Objects()

    # Small object in upper-left
    obj1 = Object("pencil", 100, 100, 150, 180, None, workspace)
    objects.append(obj1)

    # Medium object in center
    obj2 = Object("pen", 280, 220, 360, 280, None, workspace)
    objects.append(obj2)

    # Large object in lower-right
    obj3 = Object("eraser", 450, 350, 550, 450, None, workspace)
    objects.append(obj3)

    return workspace, objects


@pytest.mark.integration
class TestObjectWorkspaceIntegration:
    """Integration tests for Object-Workspace interactions."""

    def test_object_creation_with_workspace_transforms(self, mock_environment):
        """Test that objects correctly use workspace coordinate transformations."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)
        workspace = workspaces.get_home_workspace()
        workspace.set_img_shape((640, 480, 3))

        # Create object at known pixel position
        obj = Object("test", 320, 240, 420, 340, None, workspace)

        # Verify coordinate transformation was applied
        x, y = obj.coordinate()
        assert 0.1 <= x <= 0.4  # Within workspace bounds
        assert -0.15 <= y <= 0.15  # Within workspace bounds

        # Verify pose includes orientation
        pose = obj.pose_com()
        assert pose.pitch == pytest.approx(1.57)  # Downward gripper

    def test_object_dimensions_calculation_with_real_workspace(self, mock_environment):
        """Test that object dimensions are correctly calculated using workspace."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)
        workspace = workspaces.get_home_workspace()
        workspace.set_img_shape((640, 480, 3))

        # Create object spanning known pixel range
        obj = Object("ruler", 100, 100, 300, 200, None, workspace)

        # Verify physical dimensions are calculated
        width = obj.width_m()
        height = obj.height_m()
        size = obj.size_m2()

        assert width > 0
        assert height > 0
        assert size > 0
        assert abs(size - (width * height)) < 0.0001

    def test_object_with_segmentation_mask_integration(self, mock_environment):
        """Test full integration with segmentation mask."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)
        workspace = workspaces.get_home_workspace()
        workspace.set_img_shape((640, 480, 3))

        # Create realistic segmentation mask (rectangle)
        mask = np.zeros((640, 480), dtype=np.uint8)
        mask[100:200, 100:300] = 255

        obj = Object("masked_object", 100, 100, 300, 200, mask, workspace)

        # Verify mask-derived properties
        assert obj.largest_contour() is not None
        assert obj.gripper_rotation() >= 0

        # Verify center of mass is calculated from mask
        x_com, y_com = obj.xy_com()
        x_center, y_center = obj.xy_center()

        # Center of mass and geometric center may differ with mask
        assert isinstance(x_com, float)
        assert isinstance(y_com, float)


@pytest.mark.integration
class TestSpatialQueriesIntegration:
    """Integration tests for spatial queries with real coordinate systems."""

    def test_spatial_filtering_with_real_coordinates(self, workspace_with_objects):
        """Test spatial filtering using actual workspace coordinates."""
        workspace, objects = workspace_with_objects

        # Get reference point (center of workspace)
        center = workspace.xy_center_wc()
        ref_point = [center.x, center.y]

        # Test all spatial filters
        left_objects = objects.get_detected_objects(location=Location.LEFT_NEXT_TO, coordinate=ref_point)

        right_objects = objects.get_detected_objects(location=Location.RIGHT_NEXT_TO, coordinate=ref_point)

        above_objects = objects.get_detected_objects(location=Location.ABOVE, coordinate=ref_point)

        below_objects = objects.get_detected_objects(location=Location.BELOW, coordinate=ref_point)

        # All objects should be accounted for in directional queries
        total_directional = len(left_objects) + len(right_objects) + len(above_objects) + len(below_objects)

        # Some objects might be counted in multiple directions
        assert total_directional >= len(objects)

    def test_nearest_object_search_accuracy(self, workspace_with_objects):
        """Test nearest object search with real coordinate system."""
        workspace, objects = workspace_with_objects

        # Search from a known location
        search_point = [0.25, 0.0]  # Center of workspace

        nearest, distance = objects.get_nearest_detected_object(search_point)

        assert nearest is not None
        assert distance >= 0

        # Verify distance calculation
        actual_distance = math.sqrt((nearest.x_com() - search_point[0]) ** 2 + (nearest.y_com() - search_point[1]) ** 2)

        assert abs(distance - actual_distance) < 0.001

        # Verify no other object is closer
        for obj in objects:
            obj_distance = math.sqrt((obj.x_com() - search_point[0]) ** 2 + (obj.y_com() - search_point[1]) ** 2)
            assert obj_distance >= distance

    def test_size_based_queries_integration(self, workspace_with_objects):
        """Test size-based queries with real object dimensions."""
        workspace, objects = workspace_with_objects

        # Get largest and smallest
        largest, largest_size = objects.get_largest_detected_object()
        smallest, smallest_size = objects.get_smallest_detected_object()

        assert largest is not None
        assert smallest is not None
        assert largest_size >= smallest_size

        # Get sorted objects
        sorted_asc = objects.get_detected_objects_sorted(ascending=True)
        sorted_desc = objects.get_detected_objects_sorted(ascending=False)

        # Verify sorting
        assert sorted_asc[0] == smallest
        assert sorted_asc[-1] == largest
        assert sorted_desc[0] == largest
        assert sorted_desc[-1] == smallest

        # Verify all sizes are in order
        sizes_asc = [obj.size_m2() for obj in sorted_asc]
        assert sizes_asc == sorted(sizes_asc)

        sizes_desc = [obj.size_m2() for obj in sorted_desc]
        assert sizes_desc == sorted(sizes_desc, reverse=True)


@pytest.mark.integration
class TestSerializationRoundtrips:
    """Integration tests for complete serialization workflows."""

    def test_object_serialization_roundtrip_with_workspace(self, workspace_with_objects):
        """Test full object serialization and reconstruction."""
        workspace, objects = workspace_with_objects

        original = objects[0]

        # Serialize
        obj_dict = original.to_dict()
        json_str = original.to_json()

        # Verify JSON is valid
        parsed = json.loads(json_str)
        assert parsed["label"] == original.label()

        # Add bbox for from_dict compatibility
        if "image_coordinates" in obj_dict:
            bbox_rel = obj_dict["image_coordinates"]["bounding_box_rel"]
            img_shape = workspace.img_shape()
            obj_dict["bbox"] = {
                "x_min": int(bbox_rel["u_min"] * img_shape[0]),
                "y_min": int(bbox_rel["v_min"] * img_shape[1]),
                "x_max": int(bbox_rel["u_max"] * img_shape[0]),
                "y_max": int(bbox_rel["v_max"] * img_shape[1]),
            }
            obj_dict["has_mask"] = False

        # Reconstruct
        reconstructed = Object.from_dict(obj_dict, workspace)

        assert reconstructed is not None
        assert reconstructed.label() == original.label()

        # Verify coordinates match (with tolerance for float operations)
        assert abs(reconstructed.x_com() - original.x_com()) < 0.01
        assert abs(reconstructed.y_com() - original.y_com()) < 0.01

    def test_collection_serialization_roundtrip(self, workspace_with_objects):
        """Test serialization of entire object collection."""
        workspace, original_objects = workspace_with_objects

        # Serialize collection
        dict_list = Objects.objects_to_dict_list(original_objects)

        # Verify serialization
        assert len(dict_list) == len(original_objects)
        assert all(isinstance(d, dict) for d in dict_list)

        # Reconstruct collection
        reconstructed_objects = Objects.dict_list_to_objects(dict_list, workspace)

        assert len(reconstructed_objects) == len(original_objects)

        # Verify all objects reconstructed correctly
        for orig, recon in zip(original_objects, reconstructed_objects):
            assert recon.label() == orig.label()
            assert abs(recon.x_com() - orig.x_com()) < 0.01
            assert abs(recon.y_com() - orig.y_com()) < 0.01

    def test_serialization_with_mask_roundtrip(self, mock_environment):
        """Test serialization with segmentation mask."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)
        workspace = workspaces.get_home_workspace()
        workspace.set_img_shape((640, 480, 3))

        # Create object with mask
        mask = np.ones((640, 480), dtype=np.uint8) * 200
        mask[100:200, 100:200] = 255

        original = Object("masked", 100, 100, 200, 200, mask, workspace)

        # Note: Current implementation doesn't serialize masks
        # This test verifies graceful handling
        obj_dict = original.to_dict()
        reconstructed = Object.from_dict(obj_dict, workspace)

        assert reconstructed is not None
        assert reconstructed.label() == original.label()


@pytest.mark.integration
class TestPickAndPlaceScenarios:
    """Integration tests simulating pick-and-place operations."""

    def test_complete_pick_workflow(self, workspace_with_objects):
        """Test complete pick operation workflow."""
        workspace, objects = workspace_with_objects

        # 1. Find target object
        target = objects.get_detected_object([0.25, 0.0], label="pen")
        assert target is not None

        # 2. Get pickup pose
        pickup_pose = target.pose_com()
        pickup_rotation = target.gripper_rotation()

        # 3. Verify pose is within workspace bounds
        assert workspace.xy_ll_wc().x <= pickup_pose.x <= workspace.xy_ul_wc().x
        assert workspace.xy_lr_wc().y <= pickup_pose.y <= workspace.xy_ul_wc().y

        # 4. Verify gripper orientation
        assert 0 <= pickup_rotation <= 2 * math.pi

        # 5. Verify pickup height
        assert pickup_pose.z > 0

    def test_complete_place_workflow(self, workspace_with_objects):
        """Test complete place operation workflow."""
        workspace, objects = workspace_with_objects

        # 1. Pick object
        target = objects[0]
        original_pos = target.coordinate()

        # 2. Define place location
        place_pose = PoseObjectPNP(0.3, -0.1, 0.05, 0.0, 1.57, 0.5)

        # 3. Update object position
        target.set_pose_com(place_pose)

        # 4. Verify position updated
        new_pos = target.coordinate()
        assert abs(new_pos[0] - place_pose.x) < 0.01
        assert abs(new_pos[1] - place_pose.y) < 0.01

        # 5. Verify position changed
        assert new_pos != original_pos

    def test_pick_and_place_with_rotation(self, workspace_with_objects):
        """Test pick-and-place with object rotation."""
        workspace, objects = workspace_with_objects

        target = objects[0]

        # Store original orientation
        original_rotation = target.gripper_rotation()

        # Create new pose with rotation
        rotation_offset = math.pi / 4  # 45 degrees
        new_pose = target.pose_com().copy_with_offsets(x_offset=0.1, y_offset=0.05, yaw_offset=rotation_offset)

        # Apply new pose
        target.set_pose_com(new_pose)

        # Verify rotation changed
        new_rotation = target.gripper_rotation()
        expected_rotation = (original_rotation + rotation_offset) % (2 * math.pi)

        # Account for angle wrapping
        diff = abs(new_rotation - expected_rotation)
        if diff > math.pi:
            diff = 2 * math.pi - diff

        assert diff < 0.2  # Tolerance for rotation


@pytest.mark.integration
class TestMultiWorkspaceScenarios:
    """Integration tests for multi-workspace operations."""

    def test_object_transfer_between_workspaces(self, mock_environment):
        """Test transferring object from one workspace to another."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)

        left_ws = workspaces.get_workspace_left()
        right_ws = workspaces.get_workspace_right()

        left_ws.set_img_shape((640, 480, 3))
        right_ws.set_img_shape((640, 480, 3))

        # Create object in left workspace
        left_objects = Objects()
        obj = Object("cube", 200, 200, 300, 300, None, left_ws)
        left_objects.append(obj)

        # Create right workspace collection
        right_objects = Objects()

        # Transfer object
        transferred = left_objects[0]
        left_objects.remove(transferred)

        # Calculate new position in right workspace
        new_pose = right_ws.transform_camera2world_coords(right_ws.id(), 0.5, 0.5, 0.0)

        transferred.set_pose_com(new_pose)
        right_objects.append(transferred)

        # Verify transfer
        assert len(left_objects) == 0
        assert len(right_objects) == 1
        assert transferred in right_objects

    def test_sorting_across_workspaces(self, mock_environment):
        """Test sorting objects from multiple workspaces."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)

        left_ws = workspaces.get_workspace_left()
        right_ws = workspaces.get_workspace_right()

        left_ws.set_img_shape((640, 480, 3))
        right_ws.set_img_shape((640, 480, 3))

        # Create objects in both workspaces with different sizes
        all_objects = Objects()

        # Large object in left workspace
        obj1 = Object("large", 100, 100, 300, 300, None, left_ws)
        all_objects.append(obj1)

        # Small object in right workspace
        obj2 = Object("small", 200, 200, 250, 250, None, right_ws)
        all_objects.append(obj2)

        # Medium object in left workspace
        obj3 = Object("medium", 150, 150, 250, 250, None, left_ws)
        all_objects.append(obj3)

        # Sort by size
        sorted_objs = all_objects.get_detected_objects_sorted(ascending=True)

        # Verify sorting regardless of workspace
        assert sorted_objs[0].label() == "small"
        assert sorted_objs[-1].label() == "large"

        # Verify objects maintain workspace association
        assert sorted_objs[0].workspace() == right_ws
        assert sorted_objs[-1].workspace() == left_ws


@pytest.mark.integration
class TestComplexWorkflows:
    """Integration tests for complex multi-step workflows."""

    def test_workspace_scanning_pattern(self, mock_environment):
        """Test systematic workspace scanning pattern."""
        workspaces = NiryoWorkspaces(mock_environment, verbose=False)
        workspace = workspaces.get_home_workspace()
        workspace.set_img_shape((640, 480, 3))

        # Simulate scanning workspace in a grid pattern
        scan_points = []
        for u in [0.25, 0.5, 0.75]:
            for v in [0.25, 0.5, 0.75]:
                pose = workspace.transform_camera2world_coords(workspace.id(), u, v, 0.0)
                scan_points.append(pose)

        # Verify all scan points are within workspace
        assert len(scan_points) == 9

        for pose in scan_points:
            assert workspace.xy_ll_wc().x <= pose.x <= workspace.xy_ul_wc().x
            assert workspace.xy_lr_wc().y <= pose.y <= workspace.xy_ul_wc().y

    def test_object_grouping_and_batch_operations(self, workspace_with_objects):
        """Test grouping objects and performing batch operations."""
        workspace, objects = workspace_with_objects

        # Group objects by size category
        small_objects = Objects()
        large_objects = Objects()

        for obj in objects:
            size_cm2 = obj.size_m2() * 10000
            if size_cm2 < 40:
                small_objects.append(obj)
            else:
                large_objects.append(obj)

        # Verify grouping
        assert len(small_objects) + len(large_objects) == len(objects)

        # Perform batch operation: move all small objects to a row
        for i, obj in enumerate(small_objects):
            new_pose = PoseObjectPNP(
                x=0.3, y=-0.1 + i * 0.05, z=0.05, roll=0.0, pitch=1.57, yaw=obj.gripper_rotation()  # Space 5cm apart
            )
            obj.set_pose_com(new_pose)

        # Verify all small objects are in a row
        for i, obj in enumerate(small_objects):
            expected_y = -0.1 + i * 0.05
            assert abs(obj.y_com() - expected_y) < 0.01

    def test_llm_integration_workflow(self, workspace_with_objects):
        """Test workflow for LLM integration."""
        workspace, objects = workspace_with_objects

        # 1. Generate scene description for LLM
        scene_description = []
        for obj in objects:
            scene_description.append(obj.as_string_for_llm())

        # 2. Verify all objects described
        assert len(scene_description) == len(objects)

        # 3. Simulate LLM requesting object by description
        # "Pick the largest object"
        largest, _ = objects.get_largest_detected_object()

        # 4. Get serialized object for LLM response
        largest_dict = largest.to_dict()

        # 5. Verify LLM can reconstruct object
        assert "position" in largest_dict
        assert "dimensions" in largest_dict
        assert "label" in largest_dict

        # 6. Simulate LLM providing placement instruction
        placement_instruction = {"x": 0.3, "y": 0.0, "z": 0.05, "yaw": 0.5}

        # 7. Execute placement
        new_pose = PoseObjectPNP(
            placement_instruction["x"],
            placement_instruction["y"],
            placement_instruction["z"],
            0.0,
            1.57,
            placement_instruction["yaw"],
        )

        largest.set_pose_com(new_pose)

        # 8. Verify placement
        assert abs(largest.x_com() - 0.3) < 0.01
        assert abs(largest.y_com() - 0.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
