from __future__ import annotations

from dataclasses import dataclass
import threading
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
                    "width": int(self.get_parameter(f"camera_{idx}_width").value),
                    "height": int(self.get_parameter(f"camera_{idx}_height").value),
                    "fps": int(self.get_parameter(f"camera_{idx}_fps").value),
                    "exposure_mode": str(
                        self.get_parameter(f"camera_{idx}_exposure_mode").value
                    ).strip().lower(),
                    "exposure_value": float(self.get_parameter(f"camera_{idx}_exposure_value").value),
                    "gain_value": float(self.get_parameter(f"camera_{idx}_gain_value").value),
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
            self.declare_parameter(f"camera_{idx}_width", 0)
            self.declare_parameter(f"camera_{idx}_height", 0)
            self.declare_parameter(f"camera_{idx}_fps", 0)
            self.declare_parameter(f"camera_{idx}_exposure_mode", "keep")
            self.declare_parameter(f"camera_{idx}_exposure_value", 8000.0)
            self.declare_parameter(f"camera_{idx}_gain_value", -1.0)

    def _camera_width(self, config: dict[str, Any]) -> int:
        return int(config["width"]) if int(config["width"]) > 0 else self.width

    def _camera_height(self, config: dict[str, Any]) -> int:
        return int(config["height"]) if int(config["height"]) > 0 else self.height

    def _camera_fps_value(self, config: dict[str, Any]) -> int:
        return int(config["fps"]) if int(config["fps"]) > 0 else self.camera_fps

    def _format_option_range(self, option_range: Any) -> str:
        return (
            f"[min={float(option_range.min):.3f}, max={float(option_range.max):.3f}, "
            f"step={float(option_range.step):.3f}, default={float(option_range.default):.3f}]"
        )

    def _log_sensor_option(self, sensor: Any, option: Any, *, camera_name: str, option_name: str) -> None:
        if not sensor.supports(option):
            self.get_logger().info(f"Camera '{camera_name}' does not support {option_name}.")
            return

        try:
            option_range = sensor.get_option_range(option)
            range_text = self._format_option_range(option_range)
        except RuntimeError as exc:
            range_text = f"range unavailable: {exc}"

        try:
            current_value = float(sensor.get_option(option))
            current_text = f"{current_value:.3f}"
        except RuntimeError as exc:
            current_text = f"unavailable: {exc}"

        self.get_logger().info(
            f"Camera '{camera_name}' color {option_name}: current={current_text}, {range_text}"
        )

    def _log_color_exposure_controls(self, rs: Any, sensor: Any, config: dict[str, Any]) -> None:
        camera_name = f"{config['name']} ({config['serial']})"
        for option, option_name in (
            (rs.option.enable_auto_exposure, "enable_auto_exposure"),
            (rs.option.exposure, "exposure"),
            (rs.option.gain, "gain"),
        ):
            self._log_sensor_option(
                sensor,
                option,
                camera_name=camera_name,
                option_name=option_name,
            )

    def _set_sensor_option(
        self,
        sensor: Any,
        option: Any,
        value: float,
        *,
        camera_name: str,
        option_name: str,
    ) -> bool:
        if not sensor.supports(option):
            self.get_logger().warning(
                f"Camera '{camera_name}' does not support the RealSense option '{option_name}'."
            )
            return False

        try:
            option_range = sensor.get_option_range(option)
        except RuntimeError:
            option_range = None

        if option_range is not None:
            min_value = float(option_range.min)
            max_value = float(option_range.max)
            if value < min_value or value > max_value:
                self.get_logger().warning(
                    f"Camera '{camera_name}' rejected {option_name}={value:.3f}. "
                    f"Valid range is [{min_value:.3f}, {max_value:.3f}]. Skipping this override."
                )
                return False

        try:
            sensor.set_option(option, value)
        except RuntimeError as exc:
            if option_range is not None:
                min_value = float(option_range.min)
                max_value = float(option_range.max)
                self.get_logger().warning(
                    f"Failed to set {option_name}={value:.3f} for camera '{camera_name}': {exc}. "
                    f"Valid range is [{min_value:.3f}, {max_value:.3f}]."
                )
            else:
                self.get_logger().warning(
                    f"Failed to set {option_name}={value:.3f} for camera '{camera_name}': {exc}."
                )
            return False
        return True

    def _sensor_matches_active_color_stream(self, rs: Any, sensor: Any, config: dict[str, Any]) -> bool:
        expected_width = self._camera_width(config)
        expected_height = self._camera_height(config)
        expected_fps = self._camera_fps_value(config)
        for stream_profile in sensor.get_stream_profiles():
            try:
                if stream_profile.stream_type() != rs.stream.color:
                    continue
                video_profile = stream_profile.as_video_stream_profile()
                if (
                    video_profile.width() == expected_width
                    and video_profile.height() == expected_height
                    and stream_profile.fps() == expected_fps
                ):
                    return True
            except RuntimeError:
                continue
        return False

    def _find_active_color_sensor(self, rs: Any, profile: Any, config: dict[str, Any]) -> Any | None:
        sensors = profile.get_device().query_sensors()

        for sensor in sensors:
            if self._sensor_matches_active_color_stream(rs, sensor, config):
                return sensor

        for sensor in sensors:
            for stream_profile in sensor.get_stream_profiles():
                try:
                    if stream_profile.stream_type() == rs.stream.color:
                        return sensor
                except RuntimeError:
                    continue

        self.get_logger().warning(
            f"Could not find a color-capable sensor for camera '{config['name']}' ({config['serial']})."
        )
        return None

    def _configure_color_sensor(self, rs: Any, profile: Any, config: dict[str, Any]) -> None:
        exposure_mode = config["exposure_mode"]
        if exposure_mode not in {"keep", "auto", "manual"}:
            raise ValueError(
                f"Invalid exposure mode '{exposure_mode}' for camera '{config['name']}'. "
                "Use one of: keep, auto, manual."
            )

        if exposure_mode == "keep":
            return

        sensor = self._find_active_color_sensor(rs, profile, config)
        if sensor is None:
            return

        self._log_color_exposure_controls(rs, sensor, config)

        camera_name = f"{config['name']} ({config['serial']})"
        if exposure_mode == "auto":
            changed = self._set_sensor_option(
                sensor,
                rs.option.enable_auto_exposure,
                1.0,
                camera_name=camera_name,
                option_name="enable_auto_exposure",
            )
            if changed:
                self.get_logger().info(
                    f"Enabled auto exposure on the color sensor for camera '{config['name']}'."
                )
            self._configure_color_gain(rs, sensor, config)
            return

        exposure_value = float(config["exposure_value"])
        if exposure_value <= 0.0:
            raise ValueError(
                f"Camera '{config['name']}' uses manual exposure mode, but exposure_value "
                f"must be > 0. Received {exposure_value}."
            )

        auto_disabled = self._set_sensor_option(
            sensor,
            rs.option.enable_auto_exposure,
            0.0,
            camera_name=camera_name,
            option_name="enable_auto_exposure",
        )
        exposure_set = self._set_sensor_option(
            sensor,
            rs.option.exposure,
            exposure_value,
            camera_name=camera_name,
            option_name="exposure",
        )
        if auto_disabled and exposure_set:
            self.get_logger().info(
                f"Set manual exposure to {exposure_value:.0f} on camera '{config['name']}'."
            )

        self._configure_color_gain(rs, sensor, config)

    def _configure_color_gain(self, rs: Any, sensor: Any, config: dict[str, Any]) -> None:
        gain_value = float(config["gain_value"])
        if gain_value < 0.0:
            return

        camera_name = f"{config['name']} ({config['serial']})"
        changed = self._set_sensor_option(
            sensor,
            rs.option.gain,
            gain_value,
            camera_name=camera_name,
            option_name="gain",
        )
        if changed:
            self.get_logger().info(
                f"Set color gain to {gain_value:.0f} for camera '{config['name']}'."
            )

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
        rs_config.enable_stream(
            rs.stream.color,
            self._camera_width(config),
            self._camera_height(config),
            rs.format.rgb8,
            self._camera_fps_value(config),
        )
        try:
            profile = pipeline.start(rs_config)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to start RealSense camera '{config['name']}' "
                f"(serial {config['serial']}) on topic {config['topic']}: {exc}"
            ) from exc

        self._configure_color_sensor(rs, profile, config)

        for _ in range(5):
            try:
                pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break

        publisher = self.create_publisher(Image, config["topic"], 10)
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
