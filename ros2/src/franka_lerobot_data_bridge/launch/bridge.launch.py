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

    publish_host = LaunchConfiguration("publish_host").perform(context)
    publish_port = int(LaunchConfiguration("publish_port").perform(context))
    command_host = LaunchConfiguration("command_host").perform(context)
    command_port = int(LaunchConfiguration("command_port").perform(context))
    include_gripper_text = LaunchConfiguration("include_gripper").perform(context).strip().lower()
    overrides = {}
    if publish_host:
        overrides["publish_host"] = publish_host
    if publish_port > 0:
        overrides["publish_port"] = publish_port
    if command_host:
        overrides["command_host"] = command_host
    if command_port > 0:
        overrides["command_port"] = command_port
    if include_gripper_text in {"true", "false"}:
        overrides["include_gripper"] = include_gripper_text == "true"

    return [
        Node(
            package="franka_lerobot_data_bridge",
            executable="lerobot_data_bridge",
            name="lerobot_data_bridge",
            parameters=[config_file, overrides],
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
            DeclareLaunchArgument(
                "publish_host",
                default_value="",
                description="Override the bridge ZMQ bind address; empty keeps YAML value.",
            ),
            DeclareLaunchArgument(
                "publish_port",
                default_value="0",
                description="Override the bridge ZMQ port; zero keeps YAML value.",
            ),
            DeclareLaunchArgument(
                "command_host",
                default_value="",
                description="Override the deployment command ZMQ bind address; empty keeps YAML value.",
            ),
            DeclareLaunchArgument(
                "command_port",
                default_value="0",
                description="Override the deployment command ZMQ port; zero keeps YAML value.",
            ),
            DeclareLaunchArgument(
                "include_gripper",
                default_value="",
                description="Override whether gripper topics are included; empty keeps YAML value.",
            ),
            OpaqueFunction(function=generate_bridge_node),
        ]
    )
