from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
import zmq
from controller_manager_msgs.srv import SwitchController
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from std_srvs.srv import SetBool

try:
    import pylibfranka
except ModuleNotFoundError:  # pragma: no cover - depends on robot computer env
    pylibfranka = None


def _stamp_to_float_seconds(sec: int, nanosec: int) -> float:
    return float(sec) + (float(nanosec) * 1e-9)


def _image_message_to_numpy(msg: Image) -> np.ndarray:
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }
    encoding = msg.encoding.lower()
    if encoding not in channels_by_encoding:
        raise ValueError(
            f"Unsupported image encoding '{msg.encoding}'. "
            "Expected one of rgb8, bgr8, rgba8, bgra8 or mono8."
        )

    channels = channels_by_encoding[encoding]
    data = np.frombuffer(msg.data, dtype=np.uint8)
    expected_size = msg.height * msg.width * channels
    if data.size < expected_size:
        raise ValueError(
            f"Image buffer is too small for {msg.width}x{msg.height} {msg.encoding}. "
            f"Expected at least {expected_size} bytes, got {data.size}."
        )

    image = data[:expected_size].reshape((msg.height, msg.width, channels))
    if encoding == "bgr8":
        image = image[:, :, ::-1]
    elif encoding == "bgra8":
        image = image[:, :, [2, 1, 0, 3]]
    elif encoding == "mono8":
        image = np.repeat(image, repeats=3, axis=2)
    elif encoding == "rgba8":
        image = image[:, :, :3]

    return np.ascontiguousarray(image)


@dataclass
class JointSample:
    values: list[float]
    stamp_s: float


@dataclass
class ImageSample:
    image: np.ndarray
    stamp_s: float
    height: int
    width: int


class LeRobotDataBridge(Node):
    """Collect latest ROS 2 robot and camera data and publish samples over ZMQ."""

    _ARM_ACTION_SOURCE_ALIASES = {
        "topic": "topic",
        "topic_delta": "topic_delta",
        "external_topic": "topic",
        "gello": "topic",
        "gello_joint_state": "topic",
        "gello_joint_states": "topic",
        "q_goal": "topic",
        "target_joint": "topic",
        "target_joint_state": "topic",
        "target_joint_states": "topic",
        "commanded_joint": "topic",
        "commanded_joint_state": "topic",
        "commanded_joint_states": "topic",
        "delta_joint": "topic_delta",
        "delta_joint_state": "topic_delta",
        "delta_joint_states": "topic_delta",
        "target_delta": "topic_delta",
        "q_goal_delta": "topic_delta",
        "robot_state": "robot_state",
        "franka_state": "robot_state",
        "franka_joint_state": "robot_state",
        "franka_joint_states": "robot_state",
        "robot_delta": "robot_delta",
    }

    def __init__(self) -> None:
        super().__init__("lerobot_data_bridge")

        self._declare_parameters()

        self.sample_rate_hz = float(self.get_parameter("sample_rate_hz").value)
        self.max_data_age_sec = float(self.get_parameter("max_data_age_sec").value)
        self.publish_host = str(self.get_parameter("publish_host").value)
        self.publish_port = int(self.get_parameter("publish_port").value)
        self.command_host = str(self.get_parameter("command_host").value)
        self.command_port = int(self.get_parameter("command_port").value)
        self.command_max_age_sec = float(self.get_parameter("command_max_age_sec").value)
        self.task_name = str(self.get_parameter("task_name").value)
        self.deployment_mode = bool(self.get_parameter("deployment_mode").value)
        self.deployment_start_active = bool(self.get_parameter("deployment_start_active").value)
        self.deployment_hold_warmup_sec = float(
            self.get_parameter("deployment_hold_warmup_sec").value
        )
        self.deployment_hold_stability_tolerance = float(
            self.get_parameter("deployment_hold_stability_tolerance").value
        )
        self.deployment_state_source = str(self.get_parameter("deployment_state_source").value).strip().lower()
        self.deployment_gripper_action_value = float(
            self.get_parameter("deployment_gripper_action_value").value
        )
        self.left_robot_ip = str(self.get_parameter("left_robot_ip").value)
        self.right_robot_ip = str(self.get_parameter("right_robot_ip").value)

        self.include_gripper = bool(self.get_parameter("include_gripper").value)
        self.include_right_arm = bool(self.get_parameter("include_right_arm").value)
        self.require_gripper_freshness = bool(
            self.get_parameter("require_gripper_freshness").value
        )
        self.arm_action_source = self._parse_arm_action_source(
            str(self.get_parameter("arm_action_source").value)
        )

        self.left_robot_joint_state_topic = str(
            self.get_parameter("left_robot_joint_state_topic").value
        )
        self.left_arm_action_topic = str(self.get_parameter("left_arm_action_topic").value)
        self.left_robot_gripper_state_topic = str(
            self.get_parameter("left_robot_gripper_state_topic").value
        )
        self.left_gripper_action_topic = str(
            self.get_parameter("left_gripper_action_topic").value
        )
        self.left_deployment_joint_command_topic = str(
            self.get_parameter("left_deployment_joint_command_topic").value
        )
        self.left_deployment_gripper_command_topic = str(
            self.get_parameter("left_deployment_gripper_command_topic").value
        )
        self.left_deployment_enable_service = str(
            self.get_parameter("left_deployment_enable_service").value
        )

        self.right_robot_joint_state_topic = str(
            self.get_parameter("right_robot_joint_state_topic").value
        )
        self.right_arm_action_topic = str(self.get_parameter("right_arm_action_topic").value)
        self.right_robot_gripper_state_topic = str(
            self.get_parameter("right_robot_gripper_state_topic").value
        )
        self.right_gripper_action_topic = str(
            self.get_parameter("right_gripper_action_topic").value
        )
        self.right_deployment_joint_command_topic = str(
            self.get_parameter("right_deployment_joint_command_topic").value
        )
        self.right_deployment_gripper_command_topic = str(
            self.get_parameter("right_deployment_gripper_command_topic").value
        )
        self.right_deployment_enable_service = str(
            self.get_parameter("right_deployment_enable_service").value
        )

        self.camera_names: list[str] = []
        self.camera_topics: list[str] = []
        for idx in range(1, 4):
            if not bool(self.get_parameter(f"camera_{idx}_enabled").value):
                continue
            self.camera_names.append(str(self.get_parameter(f"camera_{idx}_name").value))
            self.camera_topics.append(str(self.get_parameter(f"camera_{idx}_topic").value))

        self.latest_robot_arm_samples: dict[str, JointSample | None] = {"left": None, "right": None}
        self.latest_arm_action_samples: dict[str, JointSample | None] = {"left": None, "right": None}
        self.latest_robot_gripper_samples: dict[str, JointSample | None] = {"left": None, "right": None}
        self.latest_gripper_action_samples: dict[str, JointSample | None] = {"left": None, "right": None}
        self.last_published_arm_action_samples: dict[str, JointSample | None] = {
            "left": None,
            "right": None,
        }
        self.latest_camera_samples: list[ImageSample | None] = [None] * len(self.camera_topics)
        self._deployment_robots: dict[str, Any] = {}
        self._deployment_grippers: dict[str, Any] = {}
        self._last_deployment_warning_time_s: dict[str, float] = {}
        self._hold_reference_samples: dict[str, JointSample | None] = {"left": None, "right": None}
        self._hold_reference_start_s: dict[str, float] = {"left": 0.0, "right": 0.0}

        self._zmq_context = zmq.Context()
        self._socket = self._zmq_context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.bind(f"tcp://{self.publish_host}:{self.publish_port}")
        self._command_socket: zmq.Socket | None = None
        self._last_wait_reason: str | None = None
        self._last_wait_reason_log_time_s = 0.0
        self._latest_deployment_command: dict[str, Any] | None = None
        self._latest_deployment_command_stamp_s = 0.0
        self._deployment_command_active = False
        self._deployment_active = self.deployment_start_active
        self._deployment_controller_enabled = False
        self._deployment_controller_target_enabled: bool | None = None
        self._deployment_controller_transition_phase: str | None = None
        self._deployment_controller_transition_futures: list[tuple[str, Any, Any]] = []
        self._deployment_arm_controller_available: dict[str, bool] = {"left": True, "right": True}
        self._deployment_joint_command_publishers: dict[str, Any] = {}
        self._deployment_gripper_command_publishers: dict[str, Any] = {}
        self._deployment_enable_clients: dict[str, Any] = {}
        self._deployment_switch_clients: dict[str, Any] = {}
        self._last_published_gripper_command: dict[str, float | None] = {"left": None, "right": None}
        self._activation_service = None

        if self.deployment_mode:
            if self.deployment_state_source == "pylibfranka":
                self._initialize_deployment_clients()
            elif self.deployment_state_source == "topics":
                self._create_live_state_subscriptions()
            else:
                raise ValueError(
                    "deployment_state_source must be either 'pylibfranka' or 'topics'. "
                    f"Got '{self.deployment_state_source}'."
                )
            self._initialize_deployment_command_io()
            self._activation_service = self.create_service(
                SetBool,
                "set_deployment_active",
                self._handle_set_deployment_active,
            )
        else:
            self._create_live_state_subscriptions()

        for idx, topic in enumerate(self.camera_topics):
            self.create_subscription(
                Image,
                topic,
                lambda msg, camera_index=idx: self._on_camera_image(camera_index, msg),
                10,
            )

        self.timer = self.create_timer(1.0 / self.sample_rate_hz, self._publish_sample)
        self.get_logger().info(
            f"Publishing LeRobot samples over ZMQ on tcp://{self.publish_host}:{self.publish_port}"
        )
        if self.deployment_mode:
            self.get_logger().info(
                f"Deployment mode enabled: reading Franka state via {self.deployment_state_source}."
            )
            self.get_logger().info(
                "Deployment bridge starts in "
                + ("ACTIVE" if self._deployment_active else "STANDBY")
                + " mode."
            )
        else:
            self.get_logger().info(f"Arm action source mode: {self.arm_action_source}")

    def _declare_parameters(self) -> None:
        self.declare_parameter("sample_rate_hz", 15.0)
        self.declare_parameter("max_data_age_sec", 0.5)
        self.declare_parameter("publish_host", "127.0.0.1")
        self.declare_parameter("publish_port", 5555)
        self.declare_parameter("command_host", "127.0.0.1")
        self.declare_parameter("command_port", 5556)
        self.declare_parameter("command_max_age_sec", 0.5)
        self.declare_parameter("task_name", "franka_gello_teleop")
        self.declare_parameter("deployment_mode", False)
        self.declare_parameter("deployment_start_active", True)
        self.declare_parameter("deployment_hold_warmup_sec", 1.0)
        self.declare_parameter("deployment_hold_stability_tolerance", 0.002)
        self.declare_parameter("deployment_state_source", "pylibfranka")
        self.declare_parameter("deployment_gripper_action_value", 0.0)
        self.declare_parameter("left_robot_ip", "172.16.0.3")
        self.declare_parameter("right_robot_ip", "172.16.0.2")

        self.declare_parameter("include_gripper", True)
        self.declare_parameter("include_right_arm", True)
        self.declare_parameter("require_gripper_freshness", False)
        self.declare_parameter("arm_action_source", "topic")

        self.declare_parameter("left_robot_joint_state_topic", "/left/franka/joint_states")
        self.declare_parameter("left_arm_action_topic", "/left/franka/commanded_joint_states")
        self.declare_parameter(
            "left_robot_gripper_state_topic", "/left/franka_gripper/joint_states"
        )
        self.declare_parameter(
            "left_gripper_action_topic",
            "/left/gripper/gripper_client/target_gripper_width_percent",
        )
        self.declare_parameter(
            "left_deployment_joint_command_topic", "/left/deployment/joint_states"
        )
        self.declare_parameter(
            "left_deployment_gripper_command_topic",
            "/left/gripper/gripper_client/target_gripper_width_percent",
        )
        self.declare_parameter(
            "left_deployment_enable_service",
            "/left/deployment_joint_impedance_controller/set_deployment_enabled",
        )

        self.declare_parameter("right_robot_joint_state_topic", "/right/franka/joint_states")
        self.declare_parameter("right_arm_action_topic", "/right/franka/commanded_joint_states")
        self.declare_parameter(
            "right_robot_gripper_state_topic", "/right/franka_gripper/joint_states"
        )
        self.declare_parameter(
            "right_gripper_action_topic",
            "/right/gripper/gripper_client/target_gripper_width_percent",
        )
        self.declare_parameter(
            "right_deployment_joint_command_topic", "/right/deployment/joint_states"
        )
        self.declare_parameter(
            "right_deployment_gripper_command_topic",
            "/right/gripper/gripper_client/target_gripper_width_percent",
        )
        self.declare_parameter(
            "right_deployment_enable_service",
            "/right/deployment_joint_impedance_controller/set_deployment_enabled",
        )

        for idx, default_name in enumerate(["cam_left", "cam_front", "cam_right"], start=1):
            self.declare_parameter(f"camera_{idx}_enabled", True)
            self.declare_parameter(f"camera_{idx}_name", default_name)
            self.declare_parameter(f"camera_{idx}_topic", f"/cameras/{default_name}/image_raw")

    def _create_live_state_subscriptions(self) -> None:
        self.create_subscription(
            JointState,
            self.left_robot_joint_state_topic,
            lambda msg: self._store_joint_sample(self.latest_robot_arm_samples, "left", msg),
            10,
        )
        if self.arm_action_source in {"topic", "topic_delta"}:
            self.create_subscription(
                JointState,
                self.left_arm_action_topic,
                lambda msg: self._store_joint_sample(self.latest_arm_action_samples, "left", msg),
                10,
            )

        if self.include_gripper:
            self.create_subscription(
                JointState,
                self.left_robot_gripper_state_topic,
                lambda msg: self._store_robot_gripper_sample("left", msg),
                10,
            )
            self.create_subscription(
                Float32,
                self.left_gripper_action_topic,
                lambda msg: self._store_gripper_action_sample("left", msg),
                10,
            )

        if self.include_right_arm:
            self.create_subscription(
                JointState,
                self.right_robot_joint_state_topic,
                lambda msg: self._store_joint_sample(self.latest_robot_arm_samples, "right", msg),
                10,
            )
            if self.arm_action_source in {"topic", "topic_delta"}:
                self.create_subscription(
                    JointState,
                    self.right_arm_action_topic,
                    lambda msg: self._store_joint_sample(self.latest_arm_action_samples, "right", msg),
                    10,
                )

            if self.include_gripper:
                self.create_subscription(
                    JointState,
                    self.right_robot_gripper_state_topic,
                    lambda msg: self._store_robot_gripper_sample("right", msg),
                    10,
                )
                self.create_subscription(
                    Float32,
                    self.right_gripper_action_topic,
                    lambda msg: self._store_gripper_action_sample("right", msg),
                    10,
                )

    def _initialize_deployment_clients(self) -> None:
        if pylibfranka is None:
            raise RuntimeError(
                "deployment_mode requires pylibfranka in the ROS 2 Python environment."
            )

        arm_ips = {"left": self.left_robot_ip, "right": self.right_robot_ip}
        for arm_name in self._required_arms():
            robot_ip = arm_ips[arm_name]
            try:
                self._deployment_robots[arm_name] = pylibfranka.Robot(robot_ip)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to connect to {arm_name} Franka robot at {robot_ip}: {exc}"
                ) from exc

            if self.include_gripper:
                try:
                    self._deployment_grippers[arm_name] = pylibfranka.Gripper(robot_ip)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to connect to {arm_name} Franka gripper at {robot_ip}: {exc}"
                    ) from exc

    def _initialize_deployment_command_io(self) -> None:
        self._command_socket = self._zmq_context.socket(zmq.PULL)
        self._command_socket.setsockopt(zmq.RCVHWM, 1)
        self._command_socket.bind(f"tcp://{self.command_host}:{self.command_port}")

        joint_topics = {
            "left": self.left_deployment_joint_command_topic,
            "right": self.right_deployment_joint_command_topic,
        }
        gripper_topics = {
            "left": self.left_deployment_gripper_command_topic,
            "right": self.right_deployment_gripper_command_topic,
        }
        for arm_name in self._required_arms():
            self._deployment_joint_command_publishers[arm_name] = self.create_publisher(
                JointState, joint_topics[arm_name], 10
            )
            if self.include_gripper:
                self._deployment_gripper_command_publishers[arm_name] = self.create_publisher(
                    Float32, gripper_topics[arm_name], 10
                )
        enable_services = {
            "left": self.left_deployment_enable_service,
            "right": self.right_deployment_enable_service,
        }
        controller_manager_services = {
            "left": "/left/controller_manager/switch_controller",
            "right": "/right/controller_manager/switch_controller",
        }
        for arm_name in self._required_arms():
            self._deployment_enable_clients[arm_name] = self.create_client(
                SetBool, enable_services[arm_name]
            )
            self._deployment_switch_clients[arm_name] = self.create_client(
                SwitchController, controller_manager_services[arm_name]
            )

        self.get_logger().info(
            f"Deployment command bridge listening on tcp://{self.command_host}:{self.command_port}"
        )

    def _set_joint_impedance_controller_active(self, active: bool) -> bool:
        clients: dict[str, Any] = {}
        for arm_name in self._required_arms():
            client = self._deployment_switch_clients.get(arm_name)
            if client is None:
                self._deployment_arm_controller_available[arm_name] = False
                continue
            if not client.wait_for_service(timeout_sec=0.2):
                self._log_deployment_warning(
                    f"{arm_name}_controller_manager_missing",
                    f"Controller manager switch service for {arm_name} arm is not available yet.",
                )
                self._deployment_arm_controller_available[arm_name] = False
                continue
            self._deployment_arm_controller_available[arm_name] = True
            clients[arm_name] = client

        if not clients:
            return False

        futures: list[tuple[str, Any, Any]] = []
        for arm_name, client in clients.items():
            request = SwitchController.Request()
            if active:
                request.activate_controllers = ["deployment_joint_impedance_controller"]
                request.deactivate_controllers = []
            else:
                request.activate_controllers = []
                request.deactivate_controllers = ["deployment_joint_impedance_controller"]
            request.strictness = SwitchController.Request.BEST_EFFORT
            request.activate_asap = True
            futures.append((arm_name, client.srv_name, client.call_async(request)))

        self._deployment_controller_transition_futures = futures
        return True

    def _set_deployment_enable_state(self, enabled: bool) -> bool:
        clients: dict[str, Any] = {}
        for arm_name in self._required_arms():
            client = self._deployment_enable_clients.get(arm_name)
            if client is None:
                self._deployment_arm_controller_available[arm_name] = False
                continue
            if not client.wait_for_service(timeout_sec=0.2):
                self._log_deployment_warning(
                    f"{arm_name}_enable_service_missing",
                    f"Deployment enable service for {arm_name} arm is not available yet.",
                )
                self._deployment_arm_controller_available[arm_name] = False
                continue
            self._deployment_arm_controller_available[arm_name] = True
            clients[arm_name] = client

        if not clients:
            return False

        futures: list[tuple[str, Any, Any]] = []
        for arm_name, client in clients.items():
            request = SetBool.Request()
            request.data = enabled
            futures.append((arm_name, client.srv_name, client.call_async(request)))

        self._deployment_controller_transition_futures = futures
        return True

    def _process_pending_deployment_controller_transition(self) -> None:
        if self._deployment_controller_transition_phase is None:
            return

        if any(not future.done() for _, _, future in self._deployment_controller_transition_futures):
            return

        target_enabled = self._deployment_controller_target_enabled
        phase = self._deployment_controller_transition_phase
        success = True
        for arm_name, service_name, future in self._deployment_controller_transition_futures:
            try:
                response = future.result()
            except Exception as exc:
                self._log_deployment_warning(
                    f"{arm_name}_transition_exception",
                    f"Deployment controller transition via {service_name} failed for {arm_name}: {exc}",
                )
                success = False
                continue

            if hasattr(response, "ok") and not response.ok:
                self._log_deployment_warning(
                    f"{arm_name}_transition_rejected",
                    f"Controller manager rejected deployment controller transition for {arm_name}.",
                )
                success = False
                continue

            if hasattr(response, "success") and not response.success:
                message = getattr(response, "message", "")
                suffix = f" ({message})" if message else ""
                self._log_deployment_warning(
                    f"{arm_name}_deployment_toggle_rejected",
                    f"Deployment enable service rejected request for {arm_name}{suffix}.",
                )
                success = False
                continue

            self._deployment_arm_controller_available[arm_name] = True

        self._deployment_controller_transition_futures.clear()

        if not success:
            self._deployment_controller_target_enabled = None
            self._deployment_controller_transition_phase = None
            ready_arms = [
                arm_name
                for arm_name in self._required_arms()
                if self._deployment_arm_controller_available.get(arm_name, False)
            ]
            if ready_arms:
                self.get_logger().warning(
                    "Deployment controller transition is incomplete; ready arms: "
                    + ", ".join(sorted(ready_arms))
                )
            return

        if phase == "activate":
            if not self._set_deployment_enable_state(True):
                self._deployment_controller_target_enabled = None
                self._deployment_controller_transition_phase = None
                return
            self._deployment_controller_transition_phase = "enable"
            self.get_logger().info("Deployment controllers enable requested.")
            return

        if phase == "enable":
            self._deployment_controller_enabled = True
            self._deployment_controller_target_enabled = None
            self._deployment_controller_transition_phase = None
            self.get_logger().info("Deployment controllers enabled.")
            return

        if phase == "disable":
            if not self._set_joint_impedance_controller_active(False):
                self._deployment_controller_target_enabled = None
                self._deployment_controller_transition_phase = None
                return
            self._deployment_controller_transition_phase = "deactivate"
            return

        if phase == "deactivate":
            self._deployment_controller_enabled = False
            self._deployment_controller_target_enabled = None
            self._deployment_controller_transition_phase = None
            self.get_logger().info("Deployment controllers disabled.")
            return

        self._deployment_controller_target_enabled = None
        self._deployment_controller_transition_phase = None

    def _required_arms(self) -> list[str]:
        return ["left", "right"] if self.include_right_arm else ["left"]

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _log_deployment_warning(self, key: str, message: str) -> None:
        now_s = self._now_s()
        last_log_time_s = self._last_deployment_warning_time_s.get(key, 0.0)
        if (now_s - last_log_time_s) > 5.0:
            self.get_logger().warning(message)
            self._last_deployment_warning_time_s[key] = now_s

    def _parse_arm_action_source(self, source: str) -> str:
        normalized_source = source.strip().lower()
        if normalized_source not in self._ARM_ACTION_SOURCE_ALIASES:
            supported_sources = ", ".join(sorted(self._ARM_ACTION_SOURCE_ALIASES))
            raise ValueError(
                f"Unsupported arm_action_source '{source}'. Supported values: {supported_sources}."
            )
        return self._ARM_ACTION_SOURCE_ALIASES[normalized_source]

    def _extract_ordered_arm_joint_values(self, msg: JointState) -> list[float] | None:
        expected_size = 7
        if len(msg.position) < expected_size:
            return None

        if len(msg.name) < expected_size:
            return [float(value) for value in msg.position[:expected_size]]

        ordered_values: list[float | None] = [None] * expected_size
        matched_arm_joints = 0
        for joint_name, joint_value in zip(msg.name, msg.position, strict=True):
            match = re.search(r"(?:^|_)(?:fr3|panda)_joint([1-7])$", joint_name)
            if match is None:
                continue
            joint_index = int(match.group(1)) - 1
            if 0 <= joint_index < expected_size:
                ordered_values[joint_index] = float(joint_value)
                matched_arm_joints += 1

        if all(value is not None for value in ordered_values):
            return [float(value) for value in ordered_values]

        if matched_arm_joints > 0:
            self.get_logger().warning(
                "Could not extract a complete 7-DoF arm JointState sample. "
                "Ignoring this sample instead of falling back to a potentially mixed joint order."
            )
            return None

        self.get_logger().warning(
            "Could not reorder JointState by arm joint names. "
            "Falling back to raw position order for this sample."
        )
        return [float(value) for value in msg.position[:expected_size]]

    def _store_joint_sample(
        self, storage: dict[str, JointSample | None], arm_name: str, msg: JointState
    ) -> None:
        ordered_values = self._extract_ordered_arm_joint_values(msg)
        if ordered_values is None:
            return
        storage[arm_name] = JointSample(
            values=ordered_values,
            stamp_s=_stamp_to_float_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec),
        )

    def _store_gripper_action_sample(self, arm_name: str, msg: Float32) -> None:
        now = self.get_clock().now().to_msg()
        self.latest_gripper_action_samples[arm_name] = JointSample(
            values=[float(msg.data)],
            stamp_s=_stamp_to_float_seconds(now.sec, now.nanosec),
        )

    def _store_robot_gripper_sample(self, arm_name: str, msg: JointState) -> None:
        if not msg.position:
            return

        self.latest_robot_gripper_samples[arm_name] = JointSample(
            values=[float(sum(msg.position))],
            stamp_s=_stamp_to_float_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec),
        )

    def _on_camera_image(self, camera_index: int, msg: Image) -> None:
        try:
            image = _image_message_to_numpy(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return

        self.latest_camera_samples[camera_index] = ImageSample(
            image=image,
            stamp_s=_stamp_to_float_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec),
            height=msg.height,
            width=msg.width,
        )

    def _refresh_deployment_samples(self) -> None:
        arm_ips = {"left": self.left_robot_ip, "right": self.right_robot_ip}
        for arm_name in self._required_arms():
            robot = self._deployment_robots.get(arm_name)
            if robot is None:
                continue
            try:
                state = robot.read_once()
                self.latest_robot_arm_samples[arm_name] = JointSample(
                    values=[float(value) for value in np.asarray(state.q, dtype=float)[:7]],
                    stamp_s=self._now_s(),
                )
            except Exception as exc:
                self._log_deployment_warning(
                    f"{arm_name}_robot",
                    f"Failed to read {arm_name} robot state from {arm_ips[arm_name]}: {exc}",
                )

            if not self.include_gripper:
                continue

            gripper = self._deployment_grippers.get(arm_name)
            if gripper is None:
                continue
            try:
                state = gripper.read_once()
                self.latest_robot_gripper_samples[arm_name] = JointSample(
                    values=[float(state.width)],
                    stamp_s=self._now_s(),
                )
            except Exception as exc:
                self._log_deployment_warning(
                    f"{arm_name}_gripper",
                    f"Failed to read {arm_name} gripper state from {arm_ips[arm_name]}: {exc}",
                )

    def _drain_deployment_command_socket(self) -> None:
        if self._command_socket is None:
            return

        while True:
            try:
                command = self._command_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

            if isinstance(command, dict):
                self._latest_deployment_command = command
                self._latest_deployment_command_stamp_s = self._now_s()
                if not self._deployment_active:
                    continue
                if not self._deployment_command_active and self._command_has_payload(command):
                    self._set_deployment_controller_enabled(True)
                    if self._deployment_controller_enabled:
                        self._deployment_command_active = True
                        self.get_logger().info(
                            "Received first deployment command packet; enabling ROS2 command publishing."
                        )

    def _command_has_payload(self, command: dict[str, Any]) -> bool:
        for arm_name in self._required_arms():
            joint_target = command.get(f"{arm_name}_joint_target")
            if isinstance(joint_target, list) and joint_target:
                return True
            gripper_command = command.get(f"{arm_name}_gripper_command")
            if gripper_command is not None:
                return True
        return False

    def _joint_state_message(self, arm_name: str, positions: list[float]) -> JointState:
        now_msg = self.get_clock().now().to_msg()
        msg = JointState()
        msg.header.stamp = now_msg
        msg.name = [f"fr3_joint{index}" for index in range(1, len(positions) + 1)]
        msg.position = [float(value) for value in positions]
        return msg

    def _set_deployment_controller_enabled(self, enabled: bool) -> bool:
        self._process_pending_deployment_controller_transition()

        if self._deployment_controller_enabled == enabled:
            return True
        if self._deployment_controller_target_enabled == enabled:
            return False
        if self._deployment_controller_transition_phase is not None:
            return False

        self._deployment_controller_target_enabled = enabled
        if enabled:
            if not self._set_joint_impedance_controller_active(True):
                self._deployment_controller_target_enabled = None
                return False
            self._deployment_controller_transition_phase = "activate"
            self.get_logger().info("Deployment controller activation requested.")
            return True

        if self._deployment_controller_enabled:
            if not self._set_deployment_enable_state(False):
                self._deployment_controller_target_enabled = None
                return False
            self._deployment_controller_transition_phase = "disable"
        else:
            if not self._set_joint_impedance_controller_active(False):
                self._deployment_controller_target_enabled = None
                return False
            self._deployment_controller_transition_phase = "deactivate"

        self.get_logger().info("Deployment controllers disable requested.")
        return True

    def _handle_set_deployment_active(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        self._deployment_active = bool(request.data)
        if not self._deployment_active:
            self._deployment_command_active = False
            self._latest_deployment_command = None
            self._latest_deployment_command_stamp_s = 0.0
            self._set_deployment_controller_enabled(False)
            self.get_logger().info(
                "Deployment bridge switched to STANDBY mode. Publishing hold commands only."
            )
        else:
            self.get_logger().info(
                "Deployment bridge switched to ACTIVE mode. Observation streaming enabled."
            )
        response.success = True
        response.message = "active" if self._deployment_active else "standby"
        return response

    def _command_is_fresh(self) -> bool:
        return (
            self._latest_deployment_command is not None
            and (self._now_s() - self._latest_deployment_command_stamp_s) <= self.command_max_age_sec
        )

    def _current_or_command_joint_target(self, arm_name: str) -> list[float]:
        if self.latest_robot_arm_samples[arm_name] is None:
            return []

        if not self._command_is_fresh():
            return list(self.latest_robot_arm_samples[arm_name].values)

        command_key = f"{arm_name}_joint_target"
        command = self._latest_deployment_command or {}
        target = command.get(command_key)
        if not isinstance(target, list) or not target:
            return list(self.latest_robot_arm_samples[arm_name].values)
        return [float(value) for value in target]

    def _arm_state_is_stable_for_hold(self, arm_name: str) -> bool:
        sample = self.latest_robot_arm_samples[arm_name]
        if sample is None:
            self._log_deployment_warning(
                f"{arm_name}_hold_no_state",
                f"Waiting for {arm_name} robot joint state before publishing hold commands.",
            )
            return False

        now_s = self._now_s()
        reference = self._hold_reference_samples[arm_name]
        if reference is None or len(reference.values) != len(sample.values):
            self._hold_reference_samples[arm_name] = JointSample(
                values=list(sample.values),
                stamp_s=sample.stamp_s,
            )
            self._hold_reference_start_s[arm_name] = now_s
            return False

        max_delta = max(
            abs(current - previous)
            for current, previous in zip(sample.values, reference.values, strict=True)
        )
        if max_delta > self.deployment_hold_stability_tolerance:
            self._hold_reference_samples[arm_name] = JointSample(
                values=list(sample.values),
                stamp_s=sample.stamp_s,
            )
            self._hold_reference_start_s[arm_name] = now_s
            self._log_deployment_warning(
                f"{arm_name}_hold_unstable",
                f"Waiting for stable {arm_name} robot joint state before publishing hold commands "
                f"(max delta {max_delta:.6f} rad).",
            )
            return False

        stable_duration_s = now_s - self._hold_reference_start_s[arm_name]
        if stable_duration_s < self.deployment_hold_warmup_sec:
            self._log_deployment_warning(
                f"{arm_name}_hold_warmup",
                f"Waiting for {arm_name} robot joint state to stay stable for "
                f"{self.deployment_hold_warmup_sec:.2f}s before publishing hold commands.",
            )
            return False

        return True

    def _all_arm_states_are_stable_for_hold(self) -> bool:
        active_arms = [
            arm_name
            for arm_name in self._required_arms()
            if self._deployment_arm_controller_available.get(arm_name, True)
        ]
        if not active_arms:
            return False
        return all(self._arm_state_is_stable_for_hold(arm_name) for arm_name in active_arms)

    def _publish_hold_joint_commands(self) -> bool:
        if not self._all_arm_states_are_stable_for_hold():
            return False

        published_any = False
        for arm_name in self._required_arms():
            if not self._deployment_arm_controller_available.get(arm_name, True):
                continue
            sample = self.latest_robot_arm_samples[arm_name]
            if sample is None:
                continue
            publisher = self._deployment_joint_command_publishers.get(arm_name)
            if publisher is None:
                continue
            publisher.publish(self._joint_state_message(arm_name, list(sample.values)))
            published_any = True
        return published_any

    def _publish_deployment_commands(self) -> None:
        self._process_pending_deployment_controller_transition()
        self._drain_deployment_command_socket()
        if not self._deployment_active:
            self._publish_hold_joint_commands()
            return

        if not self._deployment_command_active:
            if (
                self._latest_deployment_command is not None
                and self._command_has_payload(self._latest_deployment_command)
            ):
                self._set_deployment_controller_enabled(True)
                if self._deployment_controller_enabled:
                    self._deployment_command_active = True
                    self.get_logger().info(
                        "Deployment controller is ready; switching from hold to command publishing."
                    )

        if not self._deployment_command_active:
            self._publish_hold_joint_commands()
            return

        for arm_name in self._required_arms():
            if not self._deployment_arm_controller_available.get(arm_name, True):
                continue
            target = self._current_or_command_joint_target(arm_name)
            if not target:
                continue
            publisher = self._deployment_joint_command_publishers.get(arm_name)
            if publisher is not None:
                publisher.publish(self._joint_state_message(arm_name, target))

        if not self.include_gripper or not self._command_is_fresh():
            return

        command = self._latest_deployment_command or {}
        for arm_name in self._required_arms():
            if not self._deployment_arm_controller_available.get(arm_name, True):
                continue
            command_key = f"{arm_name}_gripper_command"
            raw_value = command.get(command_key)
            if raw_value is None:
                continue

            clamped_value = max(0.0, min(1.0, float(raw_value)))
            last_value = self._last_published_gripper_command[arm_name]
            if last_value is not None and abs(clamped_value - last_value) < 1e-6:
                continue

            publisher = self._deployment_gripper_command_publishers.get(arm_name)
            if publisher is None:
                continue

            msg = Float32()
            msg.data = clamped_value
            publisher.publish(msg)
            self._last_published_gripper_command[arm_name] = clamped_value

    def _publish_deployment_hold_commands_on_shutdown(self) -> None:
        if not self.deployment_mode:
            return

        published_any = False
        for _ in range(5):
            published_this_cycle = False
            for arm_name in self._required_arms():
                if not self._deployment_arm_controller_available.get(arm_name, True):
                    continue
                sample = self.latest_robot_arm_samples.get(arm_name)
                publisher = self._deployment_joint_command_publishers.get(arm_name)
                if sample is None or publisher is None:
                    continue
                publisher.publish(self._joint_state_message(arm_name, list(sample.values)))
                published_this_cycle = True
                published_any = True
            if not published_this_cycle:
                break
            time.sleep(0.02)

        if published_any:
            self.get_logger().info(
                "Published final deployment hold command(s) before shutting down the bridge."
            )

    def _arm_action_sample(self, arm_name: str) -> JointSample | None:
        if self.arm_action_source in {"topic", "topic_delta"}:
            return self.latest_arm_action_samples[arm_name]
        return self.latest_robot_arm_samples[arm_name]

    def _arm_action_values(self, arm_name: str) -> list[float]:
        if self.arm_action_source == "topic":
            sample = self.latest_arm_action_samples[arm_name]
            if sample is None:
                return []
            return list(sample.values)

        source_sample = self._arm_action_sample(arm_name)
        if source_sample is None:
            return []

        if self.arm_action_source == "robot_state":
            return list(source_sample.values)

        previous_sample = self.last_published_arm_action_samples[arm_name]
        if previous_sample is None:
            return [0.0] * len(source_sample.values)

        return [
            current_value - previous_value
            for current_value, previous_value in zip(
                source_sample.values, previous_sample.values, strict=True
            )
        ]

    def _deployment_arm_action_values(self, arm_name: str) -> list[float]:
        sample = self.latest_robot_arm_samples[arm_name]
        if sample is None:
            return []
        return [0.0] * len(sample.values)

    def _arm_action_representation(self) -> str:
        if self.deployment_mode:
            return "absolute_joint_position"
        if self.arm_action_source in {"topic_delta", "robot_delta"}:
            return "delta_joint_position"
        return "absolute_joint_position"

    def _gripper_action_representation(self) -> str:
        return "binary_open_close"

    def _sample_is_ready(self) -> tuple[bool, str]:
        for arm_name in self._required_arms():
            if self.latest_robot_arm_samples[arm_name] is None:
                return False, f"waiting for {arm_name} robot joint states"
            if not self.deployment_mode and self._arm_action_sample(arm_name) is None:
                return False, f"waiting for {arm_name} action joint states"
            if self.include_gripper and self.latest_robot_gripper_samples[arm_name] is None:
                return False, f"waiting for {arm_name} robot gripper states"
            if (
                self.include_gripper
                and not self.deployment_mode
                and self.latest_gripper_action_samples[arm_name] is None
            ):
                return False, f"waiting for {arm_name} gripper action"

        missing_cameras = [
            self.camera_names[idx]
            for idx, sample in enumerate(self.latest_camera_samples)
            if sample is None
        ]
        if missing_cameras:
            return False, f"waiting for camera images from {', '.join(missing_cameras)}"

        reference_time_s = self.get_clock().now().nanoseconds * 1e-9
        stale_sources: list[str] = []
        for arm_name in self._required_arms():
            if reference_time_s - self.latest_robot_arm_samples[arm_name].stamp_s > self.max_data_age_sec:
                stale_sources.append(f"{arm_name} robot joint states")
            if not self.deployment_mode:
                arm_action_sample = self._arm_action_sample(arm_name)
                if arm_action_sample is not None and (
                    reference_time_s - arm_action_sample.stamp_s > self.max_data_age_sec
                ):
                    stale_sources.append(f"{arm_name} action joint states")
            if self.include_gripper and (self.require_gripper_freshness or self.deployment_mode):
                if (
                    reference_time_s - self.latest_robot_gripper_samples[arm_name].stamp_s
                    > self.max_data_age_sec
                ):
                    stale_sources.append(f"{arm_name} robot gripper states")
                if (
                    not self.deployment_mode
                    and self.latest_gripper_action_samples[arm_name] is not None
                    and (
                    reference_time_s - self.latest_gripper_action_samples[arm_name].stamp_s
                    > self.max_data_age_sec
                    )
                ):
                    stale_sources.append(f"{arm_name} gripper action")

        for camera_name, sample in zip(self.camera_names, self.latest_camera_samples, strict=True):
            if sample is not None and reference_time_s - sample.stamp_s > self.max_data_age_sec:
                stale_sources.append(f"{camera_name} images")

        if stale_sources:
            return False, f"waiting for fresh data from {', '.join(stale_sources)}"

        return True, ""

    def _publish_sample(self) -> None:
        if self.deployment_mode and self.deployment_state_source == "pylibfranka":
            self._refresh_deployment_samples()
        if self.deployment_mode:
            self._publish_deployment_commands()
            if not self._deployment_active:
                return

        ready, reason = self._sample_is_ready()
        if not ready:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if reason != self._last_wait_reason or (now_s - self._last_wait_reason_log_time_s) > 5.0:
                self.get_logger().warning(f"LeRobot bridge not publishing yet: {reason}")
                self._last_wait_reason = reason
                self._last_wait_reason_log_time_s = now_s
            return

        self._last_wait_reason = None

        robot_state: list[float] = []
        action: list[float] = []
        for arm_name in self._required_arms():
            robot_state.extend(self.latest_robot_arm_samples[arm_name].values)
            if self.deployment_mode:
                action.extend(self._deployment_arm_action_values(arm_name))
            else:
                action.extend(self._arm_action_values(arm_name))
            if self.include_gripper:
                robot_state.extend(self.latest_robot_gripper_samples[arm_name].values)
                if self.deployment_mode:
                    action.append(self.deployment_gripper_action_value)
                else:
                    action.extend(self.latest_gripper_action_samples[arm_name].values)

        camera_payload = {}
        for name, sample in zip(self.camera_names, self.latest_camera_samples, strict=True):
            if sample is None:
                return
            camera_payload[name] = {
                "rgb": sample.image,
                "shape": [sample.height, sample.width, 3],
                "stamp_s": sample.stamp_s,
            }

        packet: dict[str, Any] = {
            "task": self.task_name,
            "state": robot_state,
            "action": action,
            "arm_action_representation": self._arm_action_representation(),
            "gripper_action_representation": self._gripper_action_representation(),
            "camera_names": self.camera_names,
            "cameras": camera_payload,
            "robot_state_dim": len(robot_state),
            "action_dim": len(action),
            "include_right_arm": self.include_right_arm,
            "include_gripper": self.include_gripper,
        }
        self._socket.send_pyobj(packet)

        if self.deployment_mode:
            return

        for arm_name in self._required_arms():
            action_sample = self._arm_action_sample(arm_name)
            if action_sample is None:
                continue
            self.last_published_arm_action_samples[arm_name] = JointSample(
                values=list(action_sample.values),
                stamp_s=action_sample.stamp_s,
            )

    def destroy_node(self) -> bool:
        self._set_deployment_controller_enabled(False)
        self._publish_deployment_hold_commands_on_shutdown()
        if self._command_socket is not None:
            self._command_socket.close(0)
        self._socket.close(0)
        self._zmq_context.term()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LeRobotDataBridge()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
