#  Copyright (c) 2025 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import tempfile
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Opens the specified YAML file and loads its contents into a Python dictionary.


def load_yaml(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def generate_robot_nodes(context):
    config_file_name = LaunchConfiguration("robot_config_file").perform(context)
    deployment_mode = LaunchConfiguration("deployment_mode").perform(context)
    deployment_mode_enabled = deployment_mode.lower() == "true"
    package_config_dir = FindPackageShare("franka_fr3_arm_controllers").perform(context)
    config_file = os.path.join(package_config_dir, "config", config_file_name)
    controllers_yaml = os.path.join(package_config_dir, "config", "controllers.yaml")
    configs = load_yaml(config_file)
    nodes = []
    for item_name, config in configs.items():
        namespace = config["namespace"]
        override_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_{namespace}_deployment_controller.yaml",
            prefix="franka_fr3_arm_controllers_",
            delete=False,
        )
        yaml.safe_dump(
            {
                "/**": {
                    "joint_impedance_controller": {
                        "ros__parameters": {"deployment_mode": deployment_mode_enabled}
                    }
                }
            },
            override_file,
        )
        override_file.flush()
        override_file.close()
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("franka_fr3_arm_controllers"),
                            "launch",
                            "franka.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "arm_id": str(config["arm_id"]),
                    "arm_prefix": str(config["arm_prefix"]),
                    "namespace": str(namespace),
                    "urdf_file": str(config["urdf_file"]),
                    "robot_ip": str(config["robot_ip"]),
                    "load_gripper": str(config["load_gripper"]),
                    "use_fake_hardware": str(config["use_fake_hardware"]),
                    "fake_sensor_commands": str(config["fake_sensor_commands"]),
                    "joint_sources": ",".join(config["joint_sources"]),
                    "joint_state_rate": str(config["joint_state_rate"]),
                    "deployment_mode": deployment_mode,
                }.items(),
            )
        )
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                namespace=namespace,
                arguments=[
                    "joint_impedance_controller",
                    "--controller-manager-timeout",
                    "30",
                    "--param-file",
                    controllers_yaml,
                    "--param-file",
                    override_file.name,
                ]
                + (["--inactive"] if deployment_mode_enabled else []),
                output="screen",
            )
        )
    if any(str(config.get("use_rviz", "false")).lower() == "true" for config in configs.values()):
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=[
                    "--display-config",
                    PathJoinSubstitution(
                        [
                            FindPackageShare("franka_description"),
                            "rviz",
                            "visualize_franka_duo.rviz",
                        ]
                    ),
                ],
                output="screen",
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_config_file",
                default_value="example_fr3_config.yaml",
                description="Name of the robot configuration file to load (relative to config/ in franka_arm_controllers)",
            ),
            DeclareLaunchArgument(
                "deployment_mode",
                default_value="false",
                description="Enable deployment-mode gating for joint_impedance_controller.",
            ),
            OpaqueFunction(function=generate_robot_nodes),
        ]
    )
