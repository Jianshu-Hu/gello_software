"""Launch Cartesian end-effector VR teleoperation without IK."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    robot_ip_parameter_name = "robot_ip"
    load_gripper_parameter_name = "load_gripper"
    use_fake_hardware_parameter_name = "use_fake_hardware"
    fake_sensor_commands_parameter_name = "fake_sensor_commands"

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)

    pkg_share = get_package_share_directory("franka_fr3_arm_controllers")
    controllers_file = os.path.join(pkg_share, "config", "cartesian_controllers.yaml")

    franka_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots", "fr3", "fr3.urdf.xacro",
    )

    robot_description = Command(
        [
            FindExecutable(name="xacro"), " ", franka_xacro_file,
            " hand:=", load_gripper,
            " robot_ip:=", robot_ip,
            " use_fake_hardware:=", use_fake_hardware,
            " fake_sensor_commands:=", fake_sensor_commands,
            " ros2_control:=true",
            " arm_id:=fr3",
        ]
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[{"robot_description": ParameterValue(robot_description, value_type=str)}],
    )

    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": ParameterValue(robot_description, value_type=str)},
            controllers_file,
        ],
        output={"stdout": "screen", "stderr": "screen"},
    )

    franka_robot_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["franka_robot_state_broadcaster"],
        condition=UnlessCondition(use_fake_hardware),
        output="screen",
    )

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    cartesian_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "cartesian_end_effector_controller",
            "--controller-manager-timeout", "30",
            "--param-file", controllers_file,
        ],
        output="screen",
    )

    vr_cartesian_node = Node(
        package="franka_fr3_arm_controllers",
        executable="vr_cartesian_teleop_node",
        name="vr_cartesian_teleop_node",
        parameters=[controllers_file],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                robot_ip_parameter_name,
                default_value="172.16.0.2",
                description="Hostname or IP address of the robot.",
            ),
            DeclareLaunchArgument(
                use_fake_hardware_parameter_name,
                default_value="false",
                description="Use fake hardware",
            ),
            DeclareLaunchArgument(
                fake_sensor_commands_parameter_name,
                default_value="false",
                description="Fake sensor commands",
            ),
            DeclareLaunchArgument(
                load_gripper_parameter_name,
                default_value="true",
                description="Use Franka Gripper as an end-effector.",
            ),
            robot_state_publisher,
            controller_manager_node,
            franka_robot_state_broadcaster,
            joint_state_broadcaster,
            cartesian_controller_spawner,
            vr_cartesian_node,
        ]
    )
