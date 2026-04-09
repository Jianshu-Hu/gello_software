import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_camera_node(context):
    config_file_name = LaunchConfiguration("config_file").perform(context)
    package_share_dir = FindPackageShare("franka_realsense_camera_publisher").perform(context)
    config_file = (
        config_file_name
        if os.path.isabs(config_file_name)
        else os.path.join(package_share_dir, "config", config_file_name)
    )

    return [
        Node(
            package="franka_realsense_camera_publisher",
            executable="realsense_camera_publisher",
            name="realsense_camera_publisher",
            parameters=[config_file],
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value="example_three_cameras.yaml",
                description="Name of the RealSense camera config file to load",
            ),
            OpaqueFunction(function=generate_camera_node),
        ]
    )
