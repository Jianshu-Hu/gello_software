#!/usr/bin/env python3
"""Quest UDP packet to Cartesian end-effector target publisher.

This is intentionally separate from franka_vr_teleop/vr_teleop_node.py. It
does not run IK and does not publish joint targets; it republishes the tracked
controller pose for CartesianEndEffectorController.
"""

from __future__ import annotations

import math
import socket
import struct
import threading

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

_PACK_FMT = "<18f"
_PACK_SIZE = struct.calcsize(_PACK_FMT)


def _normalize_quaternion(x: float, y: float, z: float, w: float) -> tuple[float, float, float, float]:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9 or not math.isfinite(norm):
        return 0.0, 0.0, 0.0, 1.0
    return x / norm, y / norm, z / norm, w / norm


class VRCartesianTeleopNode(Node):
    """Bridge-compatible UDP receiver for the Cartesian controller path."""

    def __init__(self) -> None:
        super().__init__("vr_cartesian_teleop_node")

        self.declare_parameter("udp_port", 9876)
        self.declare_parameter("target_pose_topic", "/cartesian_end_effector_controller/target_pose")
        self.declare_parameter("hand", "right")

        self.udp_port = int(self.get_parameter("udp_port").value)
        self.hand = str(self.get_parameter("hand").value).lower()
        if self.hand not in ("right", "left"):
            raise ValueError("hand must be 'right' or 'left'")

        target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.pose_pub = self.create_publisher(PoseStamped, target_pose_topic, 10)

        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.udp_port))
        self.sock.settimeout(1.0)
        self.thread = threading.Thread(target=self._udp_loop, daemon=True)
        self.thread.start()

        self.get_logger().info(
            f"VR Cartesian teleop listening on UDP {self.udp_port}, publishing {target_pose_topic}"
        )

    def _udp_loop(self) -> None:
        while self.running and rclpy.ok():
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                self.get_logger().info("Waiting for VR UDP packets...", throttle_duration_sec=5.0)
                continue
            except OSError:
                break

            if len(data) != _PACK_SIZE:
                self.get_logger().warn(
                    f"Malformed UDP packet ({len(data)} bytes)", throttle_duration_sec=2.0
                )
                continue

            vals = struct.unpack(_PACK_FMT, data)
            offset = 0 if self.hand == "right" else 9
            tracked = vals[offset + 8] > 0.5
            if not tracked:
                msg = PoseStamped()
                msg.header.frame_id = "fr3_link0"
                msg.pose.orientation.w = 1.0
                self.pose_pub.publish(msg)
                continue

            qx, qy, qz, qw = _normalize_quaternion(
                vals[offset + 3], vals[offset + 4], vals[offset + 5], vals[offset + 6]
            )

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "fr3_link0"
            msg.pose.position.x = float(vals[offset + 0])
            msg.pose.position.y = float(vals[offset + 1])
            msg.pose.position.z = float(vals[offset + 2])
            msg.pose.orientation.x = qx
            msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz
            msg.pose.orientation.w = qw
            self.pose_pub.publish(msg)

    def destroy_node(self) -> None:
        self.running = False
        self.sock.close()
        self.thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VRCartesianTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
