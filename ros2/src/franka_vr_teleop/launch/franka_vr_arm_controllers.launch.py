import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    robot_ip_parameter_name = 'robot_ip'
    load_gripper_parameter_name = 'load_gripper'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)

    # We use our new vr_controllers.yaml from our new package
    vr_teleop_pkg = get_package_share_directory('franka_vr_teleop')
    controllers_file = os.path.join(vr_teleop_pkg, 'config', 'vr_controllers.yaml')

    franka_xacro_file = os.path.join(get_package_share_directory('franka_description'), 'robots', 'fr3', 'fr3.urdf.xacro')

    robot_description = Command(
        [FindExecutable(name='xacro'), ' ', franka_xacro_file, ' hand:=', load_gripper,
         ' robot_ip:=', robot_ip, ' use_fake_hardware:=', use_fake_hardware,
         ' fake_sensor_commands:=', fake_sensor_commands]
    )

    # Spawn the Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': robot_description}],
    )

    # Spawn the Controller Manager
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[{'robot_description': robot_description}, controllers_file],
        output={
            'stdout': 'screen',
            'stderr': 'screen',
        },
    )

    # Load State Broadcaster
    franka_robot_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['franka_robot_state_broadcaster'],
        output='screen',
    )

    # Standard TF Broadcaster (Joint State)
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    # NEW: Spawn the Cartesian Controller instead of the Joint Impedance Controller
    cartesian_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['cartesian_impedance_example_controller'],
        output='screen',
    )

    # NEW: Spawn the VR bridge node
    vr_bridge_node = Node(
        package='franka_vr_teleop',
        executable='vr_teleop_node',
        name='vr_teleop_node',
        output='screen',
    )


    return LaunchDescription(
        [
            DeclareLaunchArgument(
                robot_ip_parameter_name,
                default_value='172.16.0.2',
                description='Hostname or IP address of the robot.'),
            DeclareLaunchArgument(
                use_fake_hardware_parameter_name,
                default_value='false',
                description='Use fake hardware'),
            DeclareLaunchArgument(
                fake_sensor_commands_parameter_name,
                default_value='false',
                description="Fake sensor commands. Only valid when '{}' is true".format(
                    use_fake_hardware_parameter_name)),
            DeclareLaunchArgument(
                load_gripper_parameter_name,
                default_value='true',
                description='Use Franka Gripper as an end-effector.'),
            robot_state_publisher,
            controller_manager_node,
            franka_robot_state_broadcaster,
            # joint_state_broadcaster, # Unused initially if cartesian handles joints, but standard to keep
            cartesian_controller,
            vr_bridge_node
        ]
    )
