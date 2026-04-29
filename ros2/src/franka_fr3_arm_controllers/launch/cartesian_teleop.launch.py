"""One-command launcher for direct Cartesian VR teleoperation on the Franka FR3."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_args = [
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
            description="Use fake hardware.",
        ),
        DeclareLaunchArgument(
            "fake_sensor_commands",
            default_value="false",
            description="Fake sensor commands.",
        ),
        DeclareLaunchArgument(
            "load_gripper",
            default_value="true",
            description="Load the Franka hand in the robot description.",
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
    ]

    cartesian_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("franka_fr3_arm_controllers"),
                    "launch",
                    "franka_vr_cartesian_end_effector.launch.py",
                ]
            )
        ),
        launch_arguments={
            "robot_ip": LaunchConfiguration("robot_ip"),
            "arm_id": LaunchConfiguration("arm_id"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "fake_sensor_commands": LaunchConfiguration("fake_sensor_commands"),
            "load_gripper": LaunchConfiguration("load_gripper"),
            "udp_port": LaunchConfiguration("udp_port"),
            "hand": LaunchConfiguration("hand"),
            "target_pose_topic": LaunchConfiguration("target_pose_topic"),
            "vr_to_robot_rotation_rpy": LaunchConfiguration("vr_to_robot_rotation_rpy"),
        }.items(),
    )

    return LaunchDescription(launch_args + [cartesian_stack])
