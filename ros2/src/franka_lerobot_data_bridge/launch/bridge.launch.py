import os
from glob import glob

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_bridge_node(context):
    config_file_name = LaunchConfiguration("config_file").perform(context)
    package_share_dir = FindPackageShare("franka_lerobot_data_bridge").perform(context)
    config_dir = os.path.join(package_share_dir, "config")
    config_file = (
        config_file_name
        if os.path.isabs(config_file_name)
        else os.path.join(config_dir, config_file_name)
    )

    if not os.path.isfile(config_file):
        available_configs = ", ".join(
            sorted(os.path.basename(path) for path in glob(os.path.join(config_dir, "*.yaml")))
        )
        raise FileNotFoundError(
            "LeRobot bridge config file not found: "
            f"{config_file}. Available package configs: {available_configs}"
        )

    return [
        Node(
            package="franka_lerobot_data_bridge",
            executable="lerobot_data_bridge",
            name="lerobot_data_bridge",
            parameters=[config_file],
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value="example_duo.yaml",
                description="Name of the LeRobot bridge config file to load",
            ),
            OpaqueFunction(function=generate_bridge_node),
        ]
    )
