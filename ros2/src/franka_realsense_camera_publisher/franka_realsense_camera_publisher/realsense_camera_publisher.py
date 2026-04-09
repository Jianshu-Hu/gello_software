from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


@dataclass
class CameraPublisher:
    name: str
    topic: str
    serial: str
    flip: bool
    pipeline: Any
    publisher: Any
    warned_no_frame: bool = False


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
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_images)

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

        for idx, default_name in enumerate(["cam_left", "cam_front", "cam_right"], start=1):
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
        rs_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.camera_fps)
        try:
            pipeline.start(rs_config)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to start RealSense camera '{config['name']}' "
                f"(serial {config['serial']}) on topic {config['topic']}: {exc}"
            ) from exc

        for _ in range(15):
            pipeline.wait_for_frames()

        publisher = self.create_publisher(Image, config["topic"], 10)
        return CameraPublisher(
            name=config["name"],
            topic=config["topic"],
            serial=config["serial"],
            flip=bool(config["flip"]),
            pipeline=pipeline,
            publisher=publisher,
        )

    def _publish_images(self) -> None:
        now = self.get_clock().now().to_msg()
        for camera in self.camera_publishers:
            frames = camera.pipeline.poll_for_frames()
            if not frames:
                if not camera.warned_no_frame:
                    self.get_logger().warning(
                        f"No fresh frame available yet for {camera.name} on {camera.topic}"
                    )
                    camera.warned_no_frame = True
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            camera.warned_no_frame = False

            color_bgr = np.asanyarray(color_frame.get_data())
            if camera.flip:
                color_bgr = np.rot90(color_bgr, 2)

            color_rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])

            msg = Image()
            msg.header.stamp = now
            msg.header.frame_id = f"{self.frame_id_prefix}/{camera.name}"
            msg.height = color_rgb.shape[0]
            msg.width = color_rgb.shape[1]
            msg.encoding = "rgb8"
            msg.is_bigendian = False
            msg.step = color_rgb.shape[1] * color_rgb.shape[2]
            msg.data = color_rgb.tobytes()
            camera.publisher.publish(msg)

    def destroy_node(self) -> bool:
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
