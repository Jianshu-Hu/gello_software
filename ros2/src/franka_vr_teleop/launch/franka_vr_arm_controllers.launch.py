"""Launch file for VR teleoperation of Franka FR3.

Spawns:
  - Robot State Publisher
  - ros2_control Controller Manager  (loads JointImpedanceController)
  - joint_state_broadcaster           (publishes /joint_states)
  - franka_robot_state_broadcaster    (Franka-specific state)
  - joint_impedance_controller        (tracks gello/joint_states)
  - vr_teleop_node                    (VR UDP → IK → gello/joint_states)
"""

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

    # ── Config paths ──────────────────────────────────────────────
    vr_teleop_pkg = get_package_share_directory("franka_vr_teleop")
    controllers_file = os.path.join(vr_teleop_pkg, "config", "vr_controllers.yaml")

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

    # ── Nodes ─────────────────────────────────────────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            {"robot_description": ParameterValue(robot_description, value_type=str)}
        ],
    )

    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": ParameterValue(robot_description, value_type=str)},
            controllers_file,
        ],
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
    )

    # Franka state broadcaster (real hardware only)
    franka_robot_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["franka_robot_state_broadcaster"],
        condition=UnlessCondition(use_fake_hardware),
        output="screen",
    )

    # Standard joint state broadcaster → publishes /joint_states
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    # JointImpedanceController — subscribes to gello/joint_states
    joint_impedance_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_impedance_controller",
            "--controller-manager-timeout", "30",
        ],
        output="screen",
    )

    # VR Teleop IK bridge node
    vr_bridge_node = Node(
        package="franka_vr_teleop",
        executable="vr_teleop_node",
        name="vr_teleop_node",
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
                description=(
                    "Fake sensor commands. Only valid when "
                    f"'{use_fake_hardware_parameter_name}' is true"
                ),
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
            joint_impedance_controller_spawner,
            vr_bridge_node,
        ]
    )
