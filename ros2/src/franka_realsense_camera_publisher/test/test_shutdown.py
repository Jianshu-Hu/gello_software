from types import SimpleNamespace
import threading

import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy

from franka_realsense_camera_publisher.realsense_camera_publisher import (
    CameraPublisher,
    RealSenseCameraPublisher,
)


class _Frame:
    def get_width(self):
        return 2

    def get_height(self):
        return 1

    def get_data(self):
        return np.zeros((1, 2, 3), dtype=np.uint8)


class _Pipeline:
    def wait_for_frames(self, *, timeout_ms):
        assert timeout_ms == 1000
        return SimpleNamespace(get_color_frame=_Frame)


class _Clock:
    def now(self):
        stamp = Time(sec=1, nanosec=0)
        return SimpleNamespace(to_msg=lambda: stamp)


def _camera(publisher):
    return CameraPublisher(
        name="cam_front",
        topic="/cameras/cam_front/image_raw",
        serial="test",
        flip=False,
        pipeline=_Pipeline(),
        publisher=publisher,
    )


def _node():
    return SimpleNamespace(
        publish_rate_hz=30.0,
        camera_fps=30,
        frame_id_prefix="realsense",
        _shutdown_event=threading.Event(),
        get_clock=lambda: _Clock(),
    )


@pytest.fixture(autouse=True)
def ros_context():
    rclpy.init()
    try:
        yield
    finally:
        rclpy.try_shutdown()


def test_capture_loop_suppresses_publish_error_during_ros_shutdown():
    class ShutdownPublisher:
        def publish(self, _msg):
            rclpy.try_shutdown()
            raise _rclpy.RCLError("publisher's context is invalid")

    RealSenseCameraPublisher._camera_capture_loop(_node(), _camera(ShutdownPublisher()))


def test_capture_loop_propagates_publish_error_while_ros_is_active():
    class FailingPublisher:
        def publish(self, _msg):
            raise _rclpy.RCLError("publish failed")

    with pytest.raises(_rclpy.RCLError, match="publish failed"):
        RealSenseCameraPublisher._camera_capture_loop(_node(), _camera(FailingPublisher()))
