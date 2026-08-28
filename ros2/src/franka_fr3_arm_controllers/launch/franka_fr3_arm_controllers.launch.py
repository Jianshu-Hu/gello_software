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
import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
    motion_controller = LaunchConfiguration("motion_controller").perform(context)
    deployment_mode_enabled = deployment_mode.lower() == "true"
    package_config_dir = FindPackageShare("franka_fr3_arm_controllers").perform(context)
    config_file = os.path.join(package_config_dir, "config", config_file_name)
    controllers_yaml = os.path.join(package_config_dir, "config", "controllers.yaml")
    configs = load_yaml(config_file)
    nodes = []
    for config in configs.values():
        namespace = config["namespace"]
        override_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_{namespace}_controller.yaml",
            prefix="franka_fr3_arm_controllers_",
            delete=False,
        )
        if motion_controller == "trajectory":
            if deployment_mode_enabled:
                raise ValueError("trajectory motion controller cannot be combined with deployment_mode")
            controller_name = "fr3_arm_controller"
            joint_names = [f"{config['arm_prefix']}_fr3_joint{index}" for index in range(1, 8)]
            # Match Franka's supported MoveIt trajectory controller. The hardware
            # torque path has a rate limiter; its raw position path does not.
            gains = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]
            damping = [30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0]
            controller_parameters = {
                "/**": {
                    "fr3_arm_controller": {
                        "ros__parameters": {
                            "joints": joint_names,
                            "command_interfaces": ["effort"],
                            "state_interfaces": ["position", "velocity"],
                            "allow_partial_joints_goal": False,
                            "allow_nonzero_velocity_at_trajectory_end": False,
                            "action_monitor_rate": 20.0,
                            "state_publish_rate": 50.0,
                            "constraints": {
                                "goal_time": 3.0,
                                "stopped_velocity_tolerance": 0.02,
                                **{
                                    # This is a Cartesian pose-goal workflow. A
                                    # redundant IK joint endpoint is diagnostic;
                                    # measured Cartesian tolerances decide success.
                                    joint: {"trajectory": 0.05, "goal": 0.03}
                                    for joint in joint_names
                                },
                            },
                            "gains": {
                                joint: {
                                    "p": gains[index],
                                    "d": damping[index],
                                    "i": 0.0,
                                    "i_clamp": 1.0,
                                    "ff_velocity_scale": 0.0,
                                }
                                for index, joint in enumerate(joint_names)
                            },
                        }
                    }
                }
            }
            yaml.safe_dump(controller_parameters, override_file)
        elif motion_controller == "impedance":
            controller_name = (
                "deployment_joint_impedance_controller"
                if deployment_mode_enabled
                else "joint_impedance_controller"
            )
            yaml.safe_dump({}, override_file)
        else:
            raise ValueError(
                f"unsupported motion_controller {motion_controller!r}; expected impedance or trajectory"
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
                    controller_name,
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
        if motion_controller == "trajectory":
            nodes.append(build_move_group_node(config, namespace))
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


def build_move_group_node(config, namespace):
    """Build one namespaced OMPL planning pipeline for an FR3 arm."""
    description_share = FindPackageShare("franka_description").find("franka_description")
    urdf_path = os.path.join(description_share, "robots", config["urdf_file"])
    robot_description_xml = xacro.process_file(
        urdf_path,
        mappings={
            "ros2_control": "false",
            "arm_id": str(config["arm_id"]),
            "arm_prefix": str(config["arm_prefix"]),
            "robot_ip": "",
            "hand": "false",
            "use_fake_hardware": "false",
            "fake_sensor_commands": "false",
        },
    ).toxml()
    srdf_path = os.path.join(description_share, "robots", "fr3", "fr3.srdf.xacro")
    robot_description_semantic_xml = xacro.process_file(
        srdf_path,
        mappings={"hand": "false", "arm_prefix": str(config["arm_prefix"])},
    ).toxml()

    prefix = str(config["arm_prefix"])
    group_name = f"{prefix}_fr3_arm"
    joints = [f"{prefix}_fr3_joint{index}" for index in range(1, 8)]
    planning_pipeline = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": (
                "default_planner_request_adapters/AddTimeOptimalParameterization "
                "default_planner_request_adapters/ResolveConstraintFrames "
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ),
            "start_state_max_bounds_error": 0.05,
            "planner_configs": {
                "RRTConnectkConfigDefault": {
                    "type": "geometric::RRTConnect",
                    "range": 0.0,
                }
            },
            group_name: {
                "default_planner_config": "RRTConnectkConfigDefault",
                "planner_configs": ["RRTConnectkConfigDefault"],
                "longest_valid_segment_fraction": 0.005,
            },
        }
    }
    kinematics = {
        "robot_description_kinematics": {
            group_name: {
                "kinematics_solver": "lma_kinematics_plugin/LMAKinematicsPlugin",
                "kinematics_solver_search_resolution": 0.005,
                "kinematics_solver_timeout": 0.05,
                "kinematics_solver_attempts": 5,
            }
        }
    }
    joint_limits = {
        "robot_description_planning": {
            "joint_limits": {
                joint: {
                    "has_position_limits": True,
                    "min_position": [-2.6937, -1.7337, -2.8507, -2.9921, -2.7565, 0.5945, -2.9659][index],
                    "max_position": [2.6937, 1.7337, 2.8507, -0.2018, 2.7565, 4.4669, 2.9659][index],
                    "has_velocity_limits": True,
                    "max_velocity": 0.60 if index < 4 else 0.75,
                    "has_acceleration_limits": True,
                    "max_acceleration": 2.0,
                }
                for index, joint in enumerate(joints)
            }
        }
    }
    scene_monitor = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }
    return Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace=namespace,
        output="screen",
        parameters=[
            {"allow_trajectory_execution": False},
            {"robot_description": ParameterValue(robot_description_xml, value_type=str)},
            {
                "robot_description_semantic": ParameterValue(
                    robot_description_semantic_xml, value_type=str
                )
            },
            kinematics,
            joint_limits,
            planning_pipeline,
            scene_monitor,
        ],
    )


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
                description="Load the deployment controller instead of the teleoperation controller.",
            ),
            DeclareLaunchArgument(
                "motion_controller",
                default_value="impedance",
                description="impedance for teleoperation/deployment or trajectory for MoveIt execution",
            ),
            OpaqueFunction(function=generate_robot_nodes),
        ]
    )
