from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


@dataclass
class CameraPublisher:
    name: str
    topic: str
    serial: str
    flip: bool
    pipeline: Any
    publisher: Any
    worker_thread: threading.Thread | None = None
    warned_no_frame: bool = False
    last_publish_time_s: float = 0.0


def list_realsense_serials() -> list[str]:
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyrealsense2 is not installed in the ROS 2 Python environment. "
            "Install it with `python3.10 -m pip install pyrealsense2` "
            "or rerun `gello_software/ros2/install_workspace_dependencies.bash`."
        ) from exc

    context = rs.context()
    devices = context.query_devices()
    return sorted(dev.get_info(rs.camera_info.serial_number) for dev in devices)


class RealSenseCameraPublisher(Node):
    """Publish RGB image streams from multiple RealSense cameras."""

    def __init__(self) -> None:
        super().__init__("realsense_camera_publisher")
        self._declare_parameters()

        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.frame_id_prefix = str(self.get_parameter("frame_id_prefix").value)
        self._shutdown_event = threading.Event()

        requested_configs = []
        for idx in range(1, 4):
            enabled = bool(self.get_parameter(f"camera_{idx}_enabled").value)
            if not enabled:
                continue
            requested_configs.append(
                {
                    "name": str(self.get_parameter(f"camera_{idx}_name").value),
                    "topic": str(self.get_parameter(f"camera_{idx}_topic").value),
                    "serial": str(self.get_parameter(f"camera_{idx}_serial").value).strip(),
                    "flip": bool(self.get_parameter(f"camera_{idx}_flip").value),
                }
            )

        if not requested_configs:
            raise ValueError("At least one camera must be enabled.")

        available_serials = list_realsense_serials()
        if len(available_serials) < len(requested_configs):
            raise RuntimeError(
                f"Found {len(available_serials)} RealSense cameras but {len(requested_configs)} are enabled."
            )

        assigned_serials = set()
        for config in requested_configs:
            if config["serial"]:
                if config["serial"] not in available_serials:
                    raise RuntimeError(
                        f"Configured RealSense serial '{config['serial']}' for camera "
                        f"'{config['name']}' was not detected. Available: {available_serials}"
                    )
                assigned_serials.add(config["serial"])

        unassigned_serials = [serial for serial in available_serials if serial not in assigned_serials]
        for config in requested_configs:
            if not config["serial"]:
                config["serial"] = unassigned_serials.pop(0)
                self.get_logger().warning(
                    f"Camera '{config['name']}' has no serial configured. Auto-assigned {config['serial']}."
                )

        self.camera_publishers: list[CameraPublisher] = [
            self._create_camera_publisher(config) for config in requested_configs
        ]
        for camera in self.camera_publishers:
            camera.worker_thread = threading.Thread(
                target=self._camera_capture_loop,
                args=(camera,),
                name=f"realsense-{camera.name}",
                daemon=True,
            )
            camera.worker_thread.start()

        camera_descriptions = ", ".join(
            f"{camera.name}({camera.serial} -> {camera.topic})" for camera in self.camera_publishers
        )
        self.get_logger().info(f"Publishing RealSense RGB images for {camera_descriptions}")

    def _declare_parameters(self) -> None:
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("frame_id_prefix", "realsense")

        for idx, default_name in enumerate(["cam_left", "cam_right", "cam_front"], start=1):
            self.declare_parameter(f"camera_{idx}_enabled", True)
            self.declare_parameter(f"camera_{idx}_name", default_name)
            self.declare_parameter(f"camera_{idx}_serial", "")
            self.declare_parameter(f"camera_{idx}_flip", False)
            self.declare_parameter(f"camera_{idx}_topic", f"/cameras/{default_name}/image_raw")

    def _create_camera_publisher(self, config: dict[str, Any]) -> CameraPublisher:
        try:
            import pyrealsense2 as rs
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed in the ROS 2 Python environment. "
                "Install it with `python3.10 -m pip install pyrealsense2` "
                "or rerun `gello_software/ros2/install_workspace_dependencies.bash`."
            ) from exc

        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(config["serial"])
        rs_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.camera_fps)
        try:
            profile = pipeline.start(rs_config)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to start RealSense camera '{config['name']}' "
                f"(serial {config['serial']}) on topic {config['topic']}: {exc}"
            ) from exc

        for _ in range(5):
            try:
                pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break

        publisher = self.create_publisher(Image, config["topic"], qos_profile_sensor_data)
        return CameraPublisher(
            name=config["name"],
            topic=config["topic"],
            serial=config["serial"],
            flip=bool(config["flip"]),
            pipeline=pipeline,
            publisher=publisher,
        )

    def _camera_capture_loop(self, camera: CameraPublisher) -> None:
        enforce_publish_rate = 0.0 < self.publish_rate_hz < float(self.camera_fps)
        publish_period_s = 0.0 if not enforce_publish_rate else 1.0 / self.publish_rate_hz

        while rclpy.ok() and not self._shutdown_event.is_set():
            try:
                frames = camera.pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                if not camera.warned_no_frame:
                    self.get_logger().warning(
                        f"No fresh frame available yet for {camera.name} on {camera.topic}"
                    )
                    camera.warned_no_frame = True
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                if not camera.warned_no_frame:
                    self.get_logger().warning(
                        f"RealSense returned frames for {camera.name} without a color image on {camera.topic}"
                    )
                    camera.warned_no_frame = True
                continue

            now_msg = self.get_clock().now().to_msg()
            now_s = float(now_msg.sec) + (float(now_msg.nanosec) * 1e-9)
            if publish_period_s > 0.0:
                elapsed_s = now_s - camera.last_publish_time_s
                if elapsed_s + 0.002 < publish_period_s:
                    continue

            camera.warned_no_frame = False
            camera.last_publish_time_s = now_s

            frame_width = int(color_frame.get_width())
            frame_height = int(color_frame.get_height())
            if camera.flip:
                color_rgb = np.ascontiguousarray(np.rot90(np.asanyarray(color_frame.get_data()), 2))
                image_bytes = color_rgb.tobytes()
            else:
                image_bytes = bytes(color_frame.get_data())

            msg = Image()
            msg.header.stamp = now_msg
            msg.header.frame_id = f"{self.frame_id_prefix}/{camera.name}"
            msg.height = frame_height
            msg.width = frame_width
            msg.encoding = "rgb8"
            msg.is_bigendian = False
            msg.step = frame_width * 3
            msg.data = image_bytes
            camera.publisher.publish(msg)

    def destroy_node(self) -> bool:
        self._shutdown_event.set()
        for camera in getattr(self, "camera_publishers", []):
            if camera.worker_thread is not None and camera.worker_thread.is_alive():
                camera.worker_thread.join(timeout=1.5)
        for camera in getattr(self, "camera_publishers", []):
            camera.pipeline.stop()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RealSenseCameraPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
