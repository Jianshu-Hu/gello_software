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

from utils.limit import (  # noqa: E402
    FR3_SAFE_POSITION_LOWER_RAD,
    FR3_SAFE_POSITION_UPPER_RAD,
    SustainedViolationMonitor,
)
from utils.trajectory import QuinticJointTrajectory  # noqa: E402

GRIPPER_COMMAND_MODE = "absolute_width"  # "absolute_width" or "binary_open_close"
JOINT_NAMES = [f"fr3_joint{index}" for index in range(1, 8)]


class GelloPublisher(Node):
    """ROS2 node for publishing GELLO device joint states and handling parameter updates."""

    def __init__(self) -> None:
        super().__init__("gello_publisher")
        self.declare_parameter("command_rate_hz", 15.0)
        self.declare_parameter("reference_rate_hz", 1000.0)
        self.declare_parameter("gripper_binary_open_threshold", 0.6)
        self.declare_parameter("gripper_binary_close_threshold", 0.4)
        self.command_rate_hz = float(self.get_parameter("command_rate_hz").value)
        self.reference_rate_hz = float(self.get_parameter("reference_rate_hz").value)
        if self.command_rate_hz <= 0.0 or self.reference_rate_hz <= 0.0:
            raise ValueError("GELLO command and reference rates must be positive.")
        self.gripper_binary_open_threshold = float(
            self.get_parameter("gripper_binary_open_threshold").value
        )
        self.gripper_binary_close_threshold = float(
            self.get_parameter("gripper_binary_close_threshold").value
        )
        self._latched_gripper_command: float | None = None
        self._joint_trajectory = QuinticJointTrajectory()
        self._unsafe_target_monitor = SustainedViolationMonitor(stop_after_s=1.0)

        hardware_params: GelloHardwareParams = self._setup_hardware_parameters()

        try:
            self.gello_hardware = GelloHardware(hardware_params, self.get_logger())
        except ConnectionError as e:
            self.get_logger().error(f"Failed to initialize GELLO hardware: {e}")
            raise

        self.raw_arm_joint_publisher = self.create_publisher(
            JointState, "gello/raw_joint_states", 10
        )
        self.arm_reference_publisher = self.create_publisher(
            JointState, "gello/joint_states", 10
        )
        self.gripper_joint_publisher = self.create_publisher(
            Float32, "gripper/gripper_client/target_gripper_width_percent", 10
        )

        # Subscribe to parameter events to allow dynamic updates of parameters
        self.parameter_subscription = self.create_subscription(
            ParameterEvent, "/parameter_events", self.parameter_event_callback, 10
        )

        self.get_logger().info(
            f"Publishing raw GELLO waypoints at {self.command_rate_hz:g} Hz and "
            f"quintic controller references at {self.reference_rate_hz:g} Hz."
        )
        self.command_timer = self.create_timer(
            1.0 / self.command_rate_hz, self.publish_raw_joint_target
        )
        self.reference_timer = self.create_timer(
            1.0 / self.reference_rate_hz, self.publish_interpolated_joint_reference
        )

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

    def _joint_state_message(self, positions: np.ndarray) -> JointState:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "fr3_link0"
        message.name = JOINT_NAMES
        message.position = positions.tolist()
        return message

    def publish_raw_joint_target(self) -> None:
        """Read and publish the low-rate absolute GELLO waypoint."""
        [gello_arm_joints, gripper_position] = self.gello_hardware.read_joint_states()

        now = time.monotonic()
        raw_target = np.asarray(gello_arm_joints, dtype=np.float64)
        if raw_target.shape != (7,) or not np.all(np.isfinite(raw_target)):
            violation_names = ("non_finite",)
            accepted_target = None
        else:
            position_violation = np.logical_or(
                raw_target < FR3_SAFE_POSITION_LOWER_RAD,
                raw_target > FR3_SAFE_POSITION_UPPER_RAD,
            )
            violation_names = ("position",) if np.any(position_violation) else ()
            accepted_target = np.clip(
                raw_target,
                FR3_SAFE_POSITION_LOWER_RAD,
                FR3_SAFE_POSITION_UPPER_RAD,
            )

        if violation_names:
            should_stop = self._unsafe_target_monitor.update(True, now)
            unsafe_duration = self._unsafe_target_monitor.duration(now)
            self.get_logger().warning(
                "Rejecting or clipping unsafe raw GELLO waypoint: "
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

        if accepted_target is None:
            return

        try:
            if not self._joint_trajectory.initialized:
                self._joint_trajectory.reset(accepted_target, now)
            else:
                self._joint_trajectory.update_target(accepted_target, now)
        except (TypeError, ValueError, RuntimeError) as exc:
            self.get_logger().fatal(f"Failed to generate safe GELLO trajectory: {exc}")
            rclpy.try_shutdown()
            return

        gripper_joint_states = Float32()
        gripper_joint_states.data = self._gripper_command(gripper_position)
        self.raw_arm_joint_publisher.publish(self._joint_state_message(accepted_target))
        self.gripper_joint_publisher.publish(gripper_joint_states)

    def publish_interpolated_joint_reference(self) -> None:
        """Publish the high-rate reference consumed by the impedance controller."""
        if not self._joint_trajectory.initialized:
            return
        reference = self._joint_trajectory.sample(time.monotonic())
        self.arm_reference_publisher.publish(self._joint_state_message(reference))

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
