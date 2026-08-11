import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from rcl_interfaces.msg import ParameterEvent
from rclpy.parameter import parameter_value_to_python
from franka_gello_state_publisher.gello_hardware import GelloHardware, GelloHardwareParams
from franka_gello_state_publisher.gello_parameter_config import (
    ParameterConfig,
    GelloParameterConfig,
)


def _add_repository_root_to_path() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "utils" / "limit.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    raise ImportError("Could not locate the repository-level utils/limit.py.")


_add_repository_root_to_path()

from utils.limit import JointPositionLimiter, SustainedViolationMonitor  # noqa: E402

GRIPPER_COMMAND_MODE = "absolute_width"  # "absolute_width" or "binary_open_close"


class GelloPublisher(Node):
    """ROS2 node for publishing GELLO device joint states and handling parameter updates."""

    def __init__(self) -> None:
        super().__init__("gello_publisher")
        self.PUBLISHING_RATE = 25  # Hz
        self.declare_parameter("gripper_binary_open_threshold", 0.6)
        self.declare_parameter("gripper_binary_close_threshold", 0.4)
        self.gripper_binary_open_threshold = float(
            self.get_parameter("gripper_binary_open_threshold").value
        )
        self.gripper_binary_close_threshold = float(
            self.get_parameter("gripper_binary_close_threshold").value
        )
        self._latched_gripper_command: float | None = None
        self._joint_limiter = JointPositionLimiter(max_dt=1.0 / self.PUBLISHING_RATE * 2.0)
        self._unsafe_target_monitor = SustainedViolationMonitor(stop_after_s=1.0)

        hardware_params: GelloHardwareParams = self._setup_hardware_parameters()

        try:
            self.gello_hardware = GelloHardware(hardware_params, self.get_logger())
        except ConnectionError as e:
            self.get_logger().error(f"Failed to initialize GELLO hardware: {e}")
            raise

        self.arm_joint_publisher = self.create_publisher(JointState, "gello/joint_states", 10)
        self.gripper_joint_publisher = self.create_publisher(
            Float32, "gripper/gripper_client/target_gripper_width_percent", 10
        )

        # Subscribe to parameter events to allow dynamic updates of parameters
        self.parameter_subscription = self.create_subscription(
            ParameterEvent, "/parameter_events", self.parameter_event_callback, 10
        )

        self.get_logger().info("Publishing GELLO joint states.")
        self.timer = self.create_timer(1 / self.PUBLISHING_RATE, self.publish_joint_jog)

    def parameter_event_callback(self, event: ParameterEvent) -> None:
        """Handle parameter change events for this node."""
        if event.node != self.get_fully_qualified_name():
            return

        for param in event.changed_parameters:
            # Skip parameters that are not related to the Dynamixel control parameters
            if not param.name.startswith("dynamixel_"):
                continue
            param_value = parameter_value_to_python(param.value)
            self.gello_hardware.update_dynamixel_control_parameter(param.name, param_value)

    def _binary_gripper_command(self, gripper_position: float) -> float:
        if self._latched_gripper_command is None:
            self._latched_gripper_command = 1.0 if gripper_position >= 0.5 else 0.0
        elif gripper_position >= self.gripper_binary_open_threshold:
            self._latched_gripper_command = 1.0
        elif gripper_position <= self.gripper_binary_close_threshold:
            self._latched_gripper_command = 0.0

        return self._latched_gripper_command

    def _continuous_gripper_command(self, gripper_position: float) -> float:
        return max(0.0, min(1.0, gripper_position))

    def _gripper_command(self, gripper_position: float) -> float:
        if GRIPPER_COMMAND_MODE == "absolute_width":
            return self._continuous_gripper_command(gripper_position)
        if GRIPPER_COMMAND_MODE == "binary_open_close":
            return self._binary_gripper_command(gripper_position)
        raise ValueError(
            "Unsupported GRIPPER_COMMAND_MODE "
            f"{GRIPPER_COMMAND_MODE!r}. Expected 'absolute_width' or 'binary_open_close'."
        )

    def publish_joint_jog(self) -> None:
        """Publish current joint states and gripper position."""
        JOINT_NAMES = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]
        [gello_arm_joints, gripper_position] = self.gello_hardware.read_joint_states()

        now = time.monotonic()
        raw_target = np.asarray(gello_arm_joints, dtype=np.float64)
        if not self._joint_limiter.initialized and not np.all(np.isfinite(raw_target)):
            violation_names = ("non_finite",)
            safe_target = None
        else:
            limit_result = self._joint_limiter.filter(raw_target, now)
            violation_names = limit_result.violation.names
            safe_target = limit_result.position

        if violation_names:
            should_stop = self._unsafe_target_monitor.update(True, now)
            unsafe_duration = self._unsafe_target_monitor.duration(now)
            self.get_logger().warning(
                "Filtering unsafe GELLO joint target: "
                f"{', '.join(violation_names)} violation; duration={unsafe_duration:.3f}s",
                throttle_duration_sec=1.0,
            )
            if should_stop:
                self.get_logger().fatal(
                    "Unsafe GELLO targets persisted for at least "
                    f"{self._unsafe_target_monitor.stop_after_s:.1f}s; stopping teleoperation."
                )
                rclpy.try_shutdown()
                return
        else:
            self._unsafe_target_monitor.update(False, now)

        if safe_target is None:
            return

        arm_joint_states = JointState()
        arm_joint_states.header.stamp = self.get_clock().now().to_msg()
        arm_joint_states.name = JOINT_NAMES
        arm_joint_states.header.frame_id = "fr3_link0"
        arm_joint_states.position = safe_target.tolist()

        gripper_joint_states = Float32()
        gripper_joint_states.data = self._gripper_command(gripper_position)
        self.arm_joint_publisher.publish(arm_joint_states)
        self.gripper_joint_publisher.publish(gripper_joint_states)

    def destroy_node(self) -> None:
        """Override the destroy_node method to disable torque mode before shutting down."""
        self.gello_hardware.disable_torque()
        super().destroy_node()

    def _declare_ros2_param(self, param: ParameterConfig):
        """Declare ROS2 parameters."""
        parameter_value = self.declare_parameter(
            param.descriptor.name, param.default, param.descriptor
        ).get_parameter_value()

        return parameter_value_to_python(parameter_value)

    def _setup_hardware_parameters(self):
        """Declare and setup all hardware configuration parameters."""
        config = GelloParameterConfig()

        hardware_params: GelloHardwareParams = {}
        for param in config:
            hardware_params[param.descriptor.name] = self._declare_ros2_param(param)

        return hardware_params


def main(args=None):
    rclpy.init(args=args)

    try:
        gello_publisher = GelloPublisher()
    except ConnectionError:
        rclpy.try_shutdown()
        return

    try:
        rclpy.spin(gello_publisher)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        gello_publisher.gello_hardware.disable_torque()
        gello_publisher.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
