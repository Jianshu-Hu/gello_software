"""Launch Cartesian end-effector VR teleoperation without IK."""

import os
import tempfile
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _parse_float_list(raw_value, expected_size, argument_name):
    values = [float(value.strip()) for value in raw_value.split(",") if value.strip()]
    if len(values) != expected_size:
        raise ValueError(f"{argument_name} must contain {expected_size} comma-separated values")
    return values


def _load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def _generate_nodes(context):
    robot_ip = LaunchConfiguration("robot_ip")
    load_gripper = LaunchConfiguration("load_gripper")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")

    arm_id_value = LaunchConfiguration("arm_id").perform(context)
    hand_value = LaunchConfiguration("hand").perform(context)
    udp_port_value = int(LaunchConfiguration("udp_port").perform(context))
    vr_to_robot_rotation_rpy = _parse_float_list(
        LaunchConfiguration("vr_to_robot_rotation_rpy").perform(context),
        3,
        "vr_to_robot_rotation_rpy",
    )

    target_pose_topic_value = LaunchConfiguration("target_pose_topic").perform(context)
    pkg_share = get_package_share_directory("franka_fr3_arm_controllers")
    controllers_file = os.path.join(pkg_share, "config", "cartesian_controllers.yaml")
    controller_overrides = _load_yaml(controllers_file)
    controller_overrides["cartesian_end_effector_controller"]["ros__parameters"].update(
        {
            "arm_id": arm_id_value,
            "target_pose_topic": target_pose_topic_value,
            "vr_to_robot_rotation_rpy": vr_to_robot_rotation_rpy,
        }
    )
    controller_overrides["vr_cartesian_teleop_node"]["ros__parameters"].update(
        {
            "udp_port": udp_port_value,
            "target_pose_topic": target_pose_topic_value,
            "hand": hand_value,
        }
    )
    override_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_cartesian_end_effector_controller.yaml",
        prefix="franka_fr3_arm_controllers_",
        delete=False,
    )
    yaml.safe_dump(controller_overrides, override_file)
    override_file.flush()
    override_file.close()

    franka_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots", arm_id_value, f"{arm_id_value}.urdf.xacro",
    )

    robot_description = Command(
        [
            FindExecutable(name="xacro"), " ", franka_xacro_file,
            " hand:=", load_gripper,
            " robot_ip:=", robot_ip,
            " use_fake_hardware:=", use_fake_hardware,
            " fake_sensor_commands:=", fake_sensor_commands,
            " ros2_control:=true",
            " arm_id:=", arm_id_value,
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
            override_file.name,
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
            "--param-file", override_file.name,
        ],
        output="screen",
    )

    vr_cartesian_node = Node(
        package="franka_fr3_arm_controllers",
        executable="vr_cartesian_teleop_node",
        name="vr_cartesian_teleop_node",
        parameters=[override_file.name],
        output="screen",
    )

    return [
        robot_state_publisher,
        controller_manager_node,
        franka_robot_state_broadcaster,
        joint_state_broadcaster,
        cartesian_controller_spawner,
        vr_cartesian_node,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_ip",
                default_value="172.16.0.2",
                description="Hostname or IP address of the robot.",
            ),
            DeclareLaunchArgument(
                "arm_id",
                default_value="fr3",
                description="Franka arm id, e.g. fr3.",
            ),
            DeclareLaunchArgument(
                "use_fake_hardware",
                default_value="false",
                description="Use fake hardware",
            ),
            DeclareLaunchArgument(
                "fake_sensor_commands",
                default_value="false",
                description="Fake sensor commands",
            ),
            DeclareLaunchArgument(
                "load_gripper",
                default_value="true",
                description="Use Franka Gripper as an end-effector.",
            ),
            DeclareLaunchArgument(
                "udp_port",
                default_value="9876",
                description="UDP port receiving Quest bridge packets.",
            ),
            DeclareLaunchArgument(
                "hand",
                default_value="right",
                description="VR controller hand to use: right or left.",
            ),
            DeclareLaunchArgument(
                "target_pose_topic",
                default_value="/cartesian_end_effector_controller/target_pose",
                description="Pose topic published by the VR bridge and consumed by the controller.",
            ),
            DeclareLaunchArgument(
                "vr_to_robot_rotation_rpy",
                default_value="0.0,0.0,0.0",
                description="Fixed VR-frame to robot-base rotation as roll,pitch,yaw radians.",
            ),
            OpaqueFunction(function=_generate_nodes),
        ]
    )
