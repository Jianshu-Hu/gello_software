from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
import zmq
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32


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
        self.task_name = str(self.get_parameter("task_name").value)

        self.include_gripper = bool(self.get_parameter("include_gripper").value)
        self.include_right_arm = bool(self.get_parameter("include_right_arm").value)
        self.require_gripper_freshness = bool(
            self.get_parameter("require_gripper_freshness").value
        )
        self.gripper_binary_open_threshold = float(
            self.get_parameter("gripper_binary_open_threshold").value
        )
        self.gripper_binary_close_threshold = float(
            self.get_parameter("gripper_binary_close_threshold").value
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
        self.latched_gripper_action_values: dict[str, float | None] = {"left": None, "right": None}
        self.last_published_arm_action_samples: dict[str, JointSample | None] = {
            "left": None,
            "right": None,
        }
        self.latest_camera_samples: list[ImageSample | None] = [None] * len(self.camera_topics)

        self._zmq_context = zmq.Context()
        self._socket = self._zmq_context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.bind(f"tcp://{self.publish_host}:{self.publish_port}")
        self._last_wait_reason: str | None = None
        self._last_wait_reason_log_time_s = 0.0

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
        self.get_logger().info(f"Arm action source mode: {self.arm_action_source}")

    def _declare_parameters(self) -> None:
        self.declare_parameter("sample_rate_hz", 15.0)
        self.declare_parameter("max_data_age_sec", 0.5)
        self.declare_parameter("publish_host", "127.0.0.1")
        self.declare_parameter("publish_port", 5555)
        self.declare_parameter("task_name", "franka_gello_teleop")

        self.declare_parameter("include_gripper", True)
        self.declare_parameter("include_right_arm", True)
        self.declare_parameter("require_gripper_freshness", False)
        self.declare_parameter("gripper_binary_open_threshold", 0.6)
        self.declare_parameter("gripper_binary_close_threshold", 0.4)
        self.declare_parameter("arm_action_source", "robot_state")

        self.declare_parameter("left_robot_joint_state_topic", "/left/franka/joint_states")
        self.declare_parameter("left_arm_action_topic", "/left/franka/commanded_joint_states")
        self.declare_parameter(
            "left_robot_gripper_state_topic", "/left/franka_gripper/joint_states"
        )
        self.declare_parameter(
            "left_gripper_action_topic",
            "/left/gripper/gripper_client/target_gripper_width_percent",
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

        for idx, default_name in enumerate(["cam_left", "cam_front", "cam_right"], start=1):
            self.declare_parameter(f"camera_{idx}_enabled", True)
            self.declare_parameter(f"camera_{idx}_name", default_name)
            self.declare_parameter(f"camera_{idx}_topic", f"/cameras/{default_name}/image_raw")

    def _required_arms(self) -> list[str]:
        return ["left", "right"] if self.include_right_arm else ["left"]

    def _parse_arm_action_source(self, source: str) -> str:
        normalized_source = source.strip().lower()
        if normalized_source not in self._ARM_ACTION_SOURCE_ALIASES:
            supported_sources = ", ".join(sorted(self._ARM_ACTION_SOURCE_ALIASES))
            raise ValueError(
                f"Unsupported arm_action_source '{source}'. Supported values: {supported_sources}."
            )
        return self._ARM_ACTION_SOURCE_ALIASES[normalized_source]

    def _extract_ordered_arm_joint_values(self, msg: JointState) -> list[float]:
        expected_size = 7
        if len(msg.position) < expected_size:
            return [float(value) for value in msg.position]

        if len(msg.name) < expected_size:
            return [float(value) for value in msg.position[:expected_size]]

        ordered_values: list[float | None] = [None] * expected_size
        for joint_name, joint_value in zip(msg.name, msg.position, strict=True):
            match = re.search(r"(\d+)$", joint_name)
            if match is None:
                continue
            joint_index = int(match.group(1)) - 1
            if 0 <= joint_index < expected_size:
                ordered_values[joint_index] = float(joint_value)

        if all(value is not None for value in ordered_values):
            return [float(value) for value in ordered_values]

        self.get_logger().warning(
            "Could not fully reorder JointState by joint name suffix. "
            "Falling back to raw position order for this sample."
        )
        return [float(value) for value in msg.position[:expected_size]]

    def _store_joint_sample(
        self, storage: dict[str, JointSample | None], arm_name: str, msg: JointState
    ) -> None:
        storage[arm_name] = JointSample(
            values=self._extract_ordered_arm_joint_values(msg),
            stamp_s=_stamp_to_float_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec),
        )

    def _store_gripper_action_sample(self, arm_name: str, msg: Float32) -> None:
        now = self.get_clock().now().to_msg()
        command_value = float(msg.data)
        latched_value = self.latched_gripper_action_values[arm_name]
        if latched_value is None:
            latched_value = 1.0 if command_value >= 0.5 else 0.0
        elif command_value >= self.gripper_binary_open_threshold:
            latched_value = 1.0
        elif command_value <= self.gripper_binary_close_threshold:
            latched_value = 0.0

        self.latched_gripper_action_values[arm_name] = latched_value
        self.latest_gripper_action_samples[arm_name] = JointSample(
            values=[latched_value],
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

    def _arm_action_representation(self) -> str:
        if self.arm_action_source in {"topic_delta", "robot_delta"}:
            return "delta_joint_position"
        return "absolute_joint_position"

    def _gripper_action_representation(self) -> str:
        return "binary_open_close"

    def _sample_is_ready(self) -> tuple[bool, str]:
        for arm_name in self._required_arms():
            if self.latest_robot_arm_samples[arm_name] is None:
                return False, f"waiting for {arm_name} robot joint states"
            if self._arm_action_sample(arm_name) is None:
                return False, f"waiting for {arm_name} action joint states"
            if self.include_gripper and self.latest_robot_gripper_samples[arm_name] is None:
                return False, f"waiting for {arm_name} robot gripper states"
            if self.include_gripper and self.latest_gripper_action_samples[arm_name] is None:
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
            arm_action_sample = self._arm_action_sample(arm_name)
            if arm_action_sample is not None and (
                reference_time_s - arm_action_sample.stamp_s > self.max_data_age_sec
            ):
                stale_sources.append(f"{arm_name} action joint states")
            if self.include_gripper and self.require_gripper_freshness:
                if (
                    reference_time_s - self.latest_robot_gripper_samples[arm_name].stamp_s
                    > self.max_data_age_sec
                ):
                    stale_sources.append(f"{arm_name} robot gripper states")
                if (
                    reference_time_s - self.latest_gripper_action_samples[arm_name].stamp_s
                    > self.max_data_age_sec
                ):
                    stale_sources.append(f"{arm_name} gripper action")

        for camera_name, sample in zip(self.camera_names, self.latest_camera_samples, strict=True):
            if sample is not None and reference_time_s - sample.stamp_s > self.max_data_age_sec:
                stale_sources.append(f"{camera_name} images")

        if stale_sources:
            return False, f"waiting for fresh data from {', '.join(stale_sources)}"

        return True, ""

    def _publish_sample(self) -> None:
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
            action.extend(self._arm_action_values(arm_name))
            if self.include_gripper:
                robot_state.extend(self.latest_robot_gripper_samples[arm_name].values)
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

        for arm_name in self._required_arms():
            action_sample = self._arm_action_sample(arm_name)
            if action_sample is None:
                continue
            self.last_published_arm_action_samples[arm_name] = JointSample(
                values=list(action_sample.values),
                stamp_s=action_sample.stamp_s,
            )

    def destroy_node(self) -> bool:
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
