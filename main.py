"""
Main example script demonstrating the robot_workspace package functionality.

This script shows how to:
1. Create and manage workspaces
2. Work with objects and their properties
3. Transform coordinates between camera and world frames
4. Serialize and deserialize objects
"""

from robot_workspace.workspaces.niryo_workspaces import NiryoWorkspaces
from robot_workspace.objects.object import Object
from robot_workspace.objects.objects import Objects
from robot_workspace.objects.pose_object import PoseObjectPNP
from robot_workspace.objects.object_api import Location


def demo_pose_objects():
    """Demonstrate PoseObjectPNP functionality"""
    print("\n" + "=" * 60)
    print("DEMO: PoseObjectPNP - Position and Orientation")
    print("=" * 60)

    # Create pose objects
    pose1 = PoseObjectPNP(x=0.2, y=0.1, z=0.3, roll=0.0, pitch=1.57, yaw=0.0)
    pose2 = PoseObjectPNP(x=0.1, y=0.05, z=0.1, roll=0.0, pitch=0.0, yaw=0.785)

    print(f"\nPose 1:\n{pose1}")
    print(f"\nPose 2:\n{pose2}")

    # Demonstrate arithmetic operations
    combined_pose = pose1 + pose2
    print(f"\nCombined Pose (pose1 + pose2):\n{combined_pose}")

    # Demonstrate coordinate transformations
    print(f"\nPose 1 as list: {pose1.to_list()}")
    print(f"Pose 1 quaternion: {pose1.quaternion}")
    print(f"Pose 1 XY coordinate: {pose1.xy_coordinate()}")

    # Demonstrate approximate equality
    pose3 = pose1.copy_with_offsets(x_offset=0.01)
    print(f"\nPose 1 ≈ Pose 3 (with 0.01m offset): {pose1.approx_eq(pose3, eps_position=0.02)}")


def demo_workspace_without_robot():
    """Demonstrate workspace functionality without requiring a real robot"""
    print("\n" + "=" * 60)
    print("DEMO: Workspace Management (Simulation Mode)")
    print("=" * 60)

    # Create a workspace collection for simulation
    workspaces = NiryoWorkspaces(use_simulation=True, verbose=False)

    print(f"\nNumber of workspaces: {len(workspaces)}")
    print(f"Workspace IDs: {workspaces.get_workspace_ids()}")

    # Get the home workspace
    home_workspace = workspaces.get_home_workspace()
    print(f"\nHome workspace ID: {home_workspace.id()}")
    print(f"Workspace dimensions: {home_workspace.width_m():.3f}m x {home_workspace.height_m():.3f}m")

    # Get workspace corners
    print("\nWorkspace corners (world coordinates):")
    print(f"  Upper-left:  {home_workspace.xy_ul_wc()}")
    print(f"  Lower-right: {home_workspace.xy_lr_wc()}")
    print(f"  Center:      {home_workspace.xy_center_wc()}")

    # Demonstrate observation pose
    obs_pose = home_workspace.observation_pose()
    print(f"\nObservation pose:\n{obs_pose}")

    return workspaces


def demo_objects_with_mock_workspace():
    """Demonstrate object creation and manipulation with a mock workspace"""
    print("\n" + "=" * 60)
    print("DEMO: Object Detection and Management")
    print("=" * 60)

    # Create a simple mock workspace for demonstration
    class MockWorkspace:
        def __init__(self):
            self._id = "demo_workspace"

        def id(self):
            return self._id

        def img_shape(self):
            return (640, 480, 3)

        def transform_camera2world_coords(self, ws_id, u_rel, v_rel, yaw=0.0):
            # Simple linear transformation for demo
            x = 0.4 - u_rel * 0.3
            y = 0.15 - v_rel * 0.3
            return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    workspace = MockWorkspace()

    # Create sample objects with different positions and sizes
    print("\nCreating sample objects...")

    # Object 1: A pencil in the upper-left area
    obj1 = Object(label="pencil", u_min=100, v_min=100, u_max=180, v_max=140, mask_8u=None, workspace=workspace)

    # Object 2: A pen in the center
    obj2 = Object(label="pen", u_min=280, v_min=200, u_max=360, v_max=260, mask_8u=None, workspace=workspace)

    # Object 3: An eraser in the lower-right area
    obj3 = Object(label="eraser", u_min=450, v_min=350, u_max=510, v_max=410, mask_8u=None, workspace=workspace)

    # Display object properties
    print(f"\nObject 1 - {obj1.label()}:")
    print(f"  Position (x, y): ({obj1.x_com():.3f}, {obj1.y_com():.3f}) m")
    print(f"  Dimensions: {obj1.width_m():.3f}m x {obj1.height_m():.3f}m")
    print(f"  Size: {obj1.size_m2() * 10000:.2f} cm²")
    print(f"  Gripper rotation: {obj1.gripper_rotation():.3f} rad")

    print(f"\nObject 2 - {obj2.label()}:")
    print(f"  Position (x, y): ({obj2.x_com():.3f}, {obj2.y_com():.3f}) m")
    print(f"  Dimensions: {obj2.width_m():.3f}m x {obj2.height_m():.3f}m")
    print(f"  Size: {obj2.size_m2() * 10000:.2f} cm²")

    print(f"\nObject 3 - {obj3.label()}:")
    print(f"  Position (x, y): ({obj3.x_com():.3f}, {obj3.y_com():.3f}) m")
    print(f"  Dimensions: {obj3.width_m():.3f}m x {obj3.height_m():.3f}m")
    print(f"  Size: {obj3.size_m2() * 10000:.2f} cm²")

    return [obj1, obj2, obj3], workspace


def demo_objects_collection(objects):
    """Demonstrate Objects collection functionality"""
    print("\n" + "=" * 60)
    print("DEMO: Objects Collection Operations")
    print("=" * 60)

    # Create Objects collection
    objects_collection = Objects(objects)

    print(f"\nTotal objects detected: {len(objects_collection)}")
    print(f"Objects: {objects_collection.get_detected_objects_as_comma_separated_string()}")

    # Find largest and smallest objects
    largest, largest_size = objects_collection.get_largest_detected_object()
    print(f"\nLargest object: {largest.label()} ({largest_size * 10000:.2f} cm²)")

    smallest, smallest_size = objects_collection.get_smallest_detected_object()
    print(f"Smallest object: {smallest.label()} ({smallest_size * 10000:.2f} cm²)")

    # Sort objects by size
    sorted_objects = objects_collection.get_detected_objects_sorted(ascending=True)
    print("\nObjects sorted by size (ascending):")
    for i, obj in enumerate(sorted_objects, 1):
        print(f"  {i}. {obj.label()}: {obj.size_m2() * 10000:.2f} cm²")

    # Spatial queries
    reference_point = [0.25, 0.0]
    print(f"\nSpatial queries relative to point {reference_point}:")

    left_objects = objects_collection.get_detected_objects(location=Location.LEFT_NEXT_TO, coordinate=reference_point)
    print(f"  Objects to the left: {len(left_objects)}")

    right_objects = objects_collection.get_detected_objects(location=Location.RIGHT_NEXT_TO, coordinate=reference_point)
    print(f"  Objects to the right: {len(right_objects)}")

    # Find nearest object
    nearest, distance = objects_collection.get_nearest_detected_object(reference_point)
    print(f"\nNearest object to {reference_point}: {nearest.label()} at distance {distance:.3f}m")

    # Filter by label
    pens = objects_collection.get_detected_objects(label="pen")
    print(f"\nObjects with 'pen' in label: {len(pens)}")
    for obj in pens:
        print(f"  - {obj.label()}")

    return objects_collection


def demo_serialization(obj):
    """Demonstrate object serialization and deserialization"""
    print("\n" + "=" * 60)
    print("DEMO: Object Serialization")
    print("=" * 60)

    # Serialize to dictionary
    obj_dict = obj.to_dict()
    print(f"\nSerialized {obj.label()} to dictionary:")
    print(f"  Keys: {list(obj_dict.keys())}")
    print(f"  Label: {obj_dict['label']}")
    print(f"  Position: {obj_dict['position']['center_of_mass']}")
    print(f"  Dimensions: {obj_dict['dimensions']}")

    # Serialize to JSON
    json_str = obj.to_json()
    print(f"\nJSON string length: {len(json_str)} characters")
    print(f"First 200 characters:\n{json_str[:200]}...")

    # Demonstrate collection serialization
    objects_list = [obj]
    dict_list = Objects.objects_to_dict_list(Objects(objects_list))
    print(f"\nSerialized collection of {len(dict_list)} object(s)")


def demo_llm_formatting(objects_collection):
    """Demonstrate LLM-friendly string formatting"""
    print("\n" + "=" * 60)
    print("DEMO: LLM-Friendly String Formatting")
    print("=" * 60)

    print("\nFormatted for LLM (compact):")
    for obj in objects_collection:
        print(obj.as_string_for_llm())

    print("\nFormatted for chat window:")
    for obj in objects_collection:
        print(obj.as_string_for_chat_window())


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 60)
    print("ROBOT WORKSPACE PACKAGE DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo showcases the robot_workspace package functionality")
    print("without requiring a physical robot or external dependencies.")

    # Demo 1: Pose objects
    demo_pose_objects()

    # Demo 2: Workspace management
    demo_workspace_without_robot()

    # Demo 3: Object creation and properties
    objects, workspace = demo_objects_with_mock_workspace()

    # Demo 4: Objects collection operations
    objects_collection = demo_objects_collection(objects)

    # Demo 5: Serialization
    demo_serialization(objects[0])

    # Demo 6: LLM formatting
    demo_llm_formatting(objects_collection)

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nFor more information, see README.md")
    print("To run tests: pytest")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
