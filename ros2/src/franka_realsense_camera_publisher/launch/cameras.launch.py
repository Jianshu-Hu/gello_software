import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml


def _load_camera_parameters(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return loaded.get("realsense_camera_publisher", {}).get("ros__parameters", {})


def _single_camera_overrides(camera_index):
    overrides = {}
    for idx in range(1, 4):
        overrides[f"camera_{idx}_enabled"] = idx == camera_index
    return overrides


def generate_camera_nodes(context):
    config_file_name = LaunchConfiguration("config_file").perform(context)
    package_share_dir = FindPackageShare("franka_realsense_camera_publisher").perform(context)
    config_file = (
        config_file_name
        if os.path.isabs(config_file_name)
        else os.path.join(package_share_dir, "config", config_file_name)
    )

    parameters = _load_camera_parameters(config_file)
    nodes = []
    for camera_index in range(1, 4):
        if not bool(parameters.get(f"camera_{camera_index}_enabled", True)):
            continue

        camera_name = str(parameters.get(f"camera_{camera_index}_name", f"cam_{camera_index}"))
        node_parameters = dict(parameters)
        node_parameters.update(_single_camera_overrides(camera_index))

        nodes.append(
            Node(
                package="franka_realsense_camera_publisher",
                executable="realsense_camera_publisher",
                name=f"realsense_camera_publisher_{camera_name}",
                parameters=[node_parameters],
                output="screen",
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value="example_three_cameras.yaml",
                description="Name of the RealSense camera config file to load",
            ),
            OpaqueFunction(function=generate_camera_nodes),
        ]
    )
