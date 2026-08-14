from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

import franka_gello_state_publisher.gello_publisher as publisher_module
from utils.limit import FR3_SAFE_POSITION_LOWER_RAD, FR3_SAFE_POSITION_UPPER_RAD


class FakeGelloHardware:
    def __init__(self, _params, _logger) -> None:
        self._reads = 0
        self._initial = 0.5 * (
            FR3_SAFE_POSITION_LOWER_RAD + FR3_SAFE_POSITION_UPPER_RAD
        )

    def read_joint_states(self) -> tuple[np.ndarray, float]:
        self._reads += 1
        target = self._initial.copy()
        if self._reads >= 2:
            target[0] += 0.2
        return target, 0.5

    def update_dynamixel_control_parameter(self, _name, _value) -> None:
        pass

    def disable_torque(self) -> None:
        pass


def test_raw_waypoint_and_interpolated_reference_topics(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(publisher_module, "GelloHardware", FakeGelloHardware)
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros-logs"))
    rclpy.init()
    publisher = publisher_module.GelloPublisher()
    observer = Node("gello_pipeline_test_observer")
    raw_messages: list[JointState] = []
    reference_messages: list[JointState] = []
    observer.create_subscription(
        JointState,
        "gello/raw_joint_states",
        raw_messages.append,
        100,
    )
    observer.create_subscription(
        JointState,
        "gello/joint_states",
        reference_messages.append,
        1000,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(publisher)
    executor.add_node(observer)
    try:
        deadline = time.monotonic() + 0.35
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.005)
    finally:
        executor.remove_node(observer)
        executor.remove_node(publisher)
        observer.destroy_node()
        publisher.destroy_node()
        rclpy.shutdown()

    assert 3 <= len(raw_messages) <= 7
    assert len(reference_messages) >= 10 * len(raw_messages)
    assert raw_messages[-1].position[0] - raw_messages[0].position[0] > 0.19

    reference_positions = np.asarray(
        [message.position for message in reference_messages], dtype=np.float64
    )
    assert np.all(np.diff(reference_positions[:, 0]) >= -1e-8)
    assert np.max(np.diff(reference_positions[:, 0])) < 0.02
