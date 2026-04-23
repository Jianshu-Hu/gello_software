import os
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
import transforms3d as tf3d

# Franka configuration
FR3_JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]
NUM_JOINTS = 7

# VR Protocol config
_PACK_FMT = "18f"  # 18 floats = 72 bytes
_PACK_SIZE = struct.calcsize(_PACK_FMT)
UDP_PORT = 9876

# Position scaling
VR_POSITION_SCALE = 1.0  # 1-to-1 metric mapping


class VRTeleopNode(Node):
    """ROS 2 node: VR UDP → IK → /gello/joint_states."""

    def __init__(self):
        super().__init__("vr_teleop_node")

        # ── Publishers ──────────────────────────────────────────────
        self.joint_pub = self.create_publisher(
            JointState, "gello/joint_states", 10
        )
        self.gripper_pub = self.create_publisher(
            Float32, "gello/gripper_cmd", 10
        )

        # ── Subscribers ─────────────────────────────────────────────
        # We need the robot's current pose for delta-based control
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10
        )

        # ── MuJoCo / IK Setup ───────────────────────────────────────
        import mujoco
        from dm_control import mujoco as dm_mujoco
        from dm_control.utils import inverse_kinematics

        modelfile = self._find_fr3_xml()
        self.physics = dm_mujoco.Physics.from_xml_path(modelfile)
        self.site_name = "attachment_site"

        # Check if site exists
        site_names = [
            self.physics.model.id2name(i, "site")
            for i in range(self.physics.model.nsite)
        ]
        assert self.site_name in site_names, (
            f"Site '{self.site_name}' not found in model. Available: {site_names}"
        )
        self.get_logger().info(f"IK target site: {self.site_name}")

        # ── Safety / smoothing state ───────────────────────────────
        self.last_target_pos = None
        self.last_target_quat = None
        self.alpha_pos = 0.3       # EMA smoothing factor for position
        self.alpha_rot = 0.5       # EMA smoothing factor for rotation
        self.max_delta_pos = 0.02  # max 2 cm per IK cycle
        self.last_q_goal = None    # last successfully solved joint angles
        self.last_grasp = 0.0      # last received grasp value

        # ── Control mode state ─────────────────────────────────────
        self.control_active = False
        self.reference_vr_pos = None
        self.reference_vr_rot = None
        self.reference_robot_pos = None
        self.reference_robot_quat = None

        # Robot state
        self.current_q = None

        # ── Networking setup ────────────────────────────────────────
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", UDP_PORT))
        self.sock.settimeout(1.0)
        self.running = True

        self.get_logger().info(f"VR Teleop IK Node started. Listening on UDP port {UDP_PORT}")

        self.thread = threading.Thread(target=self._udp_loop, daemon=True)
        self.thread.start()

        # ── Heartbeat Timer (100Hz) ───────────────────────────────
        # This keeps the robot 'fed' even if no VR packets arrive,
        # preventing the 'diving' behavior.
        self.timer = self.create_timer(0.01, self._timer_cb)

    # ── Locate the FR3 MuJoCo XML ─────────────────────────────────
    @staticmethod
    def _find_fr3_xml() -> str:
        # Priority 1: Check in current workspace
        ws_path = os.path.expanduser("~/real-exp-work-branch/gello_software/third_party/mujoco_menagerie/franka_fr3/fr3.xml")
        if os.path.exists(ws_path):
            return ws_path
        # Priority 2: Fallback to common search paths
        search_paths = [
            "third_party/mujoco_menagerie/franka_fr3/fr3.xml",
            "../third_party/mujoco_menagerie/franka_fr3/fr3.xml",
        ]
        for p in search_paths:
            if os.path.exists(p):
                return os.path.abspath(p)
        raise FileNotFoundError("Could not find fr3.xml Mujoco model! Check third_party path.")

    # ── Joint Callback ────────────────────────────────────────────
    def _joint_cb(self, msg: JointState):
        # Extract FR3 joints in order
        try:
            indices = [msg.name.index(name) for name in FR3_JOINT_NAMES]
            self.current_q = np.array([msg.position[i] for i in indices])
        except ValueError:
            pass # Joint name mismatch, ignore

    # ── UDP receive loop ──────────────────────────────────────────
    def _udp_loop(self):
        while self.running and rclpy.ok():
            try:
                data, _ = self.sock.recvfrom(4096)
                if len(data) == _PACK_SIZE:
                    vals = struct.unpack(_PACK_FMT, data)

                    # Right hand data
                    r_px, r_py, r_pz = vals[0], vals[1], vals[2]
                    r_qx, r_qy, r_qz, r_qw = vals[3], vals[4], vals[5], vals[6]
                    r_grasp = vals[7]
                    r_tracked = vals[8] > 0.5
                    self.last_grasp = r_grasp

                    # Clutching logic: Only move if trigger is held
                    if r_grasp < 0.5:
                        if self.control_active:
                            self.get_logger().info("Disengaged (Trigger released)")
                        self.control_active = False
                        self.reference_vr_pos = None
                        self.reference_vr_rot = None
                        continue

                    # If just engaged, capture reference
                    if not self.control_active and r_tracked:
                        if self.current_q is not None:
                            self.get_logger().info("Engaged (Trigger held)")
                            self.control_active = True
                            self.reference_vr_pos = np.array([r_px, r_py, r_pz])
                            self.reference_vr_rot = [r_qw, r_qx, r_qy, r_qz]
                            
                            # Solve current forward kinematics to get starting pose
                            self.physics.data.qpos[:NUM_JOINTS] = self.current_q
                            self.physics.forward()
                            self.reference_robot_pos = self.physics.named.data.site_xpos[self.site_name].copy()
                            self.reference_robot_quat = tf3d.quaternions.mat2quat(
                                self.physics.named.data.site_xmat[self.site_name].reshape(3,3)
                            )
                        else:
                            self.get_logger().warn("Waiting for robot joint states before engaging...", throttle_duration_sec=2.0)
                        continue

                    if self.control_active and r_tracked:
                        self._process_vr_pose(
                            np.array([r_px, r_py, r_pz]),
                            [r_qw, r_qx, r_qy, r_qz],
                            r_grasp
                        )

            except socket.timeout:
                self.get_logger().info("Waiting for VR UDP packets...", throttle_duration_sec=5.0)
                continue
            except Exception as e:
                self.get_logger().error(f"UDP Loop Error: {e}")

    # ── Process VR Pose → IK ──────────────────────────────────────
    def _process_vr_pose(self, vr_pos, vr_quat, grasp):
        from dm_control.utils import inverse_kinematics

        # 1. Delta Position (1-to-1 metric)
        delta_vr_pos = (vr_pos - self.reference_vr_pos) * VR_POSITION_SCALE
        target_pos = self.reference_robot_pos + delta_vr_pos

        # 2. Delta Rotation
        # inv(ref) * current
        q_rel = tf3d.quaternions.qmult(
            tf3d.quaternions.qinverse(self.reference_vr_rot),
            vr_quat
        )
        target_quat = tf3d.quaternions.qmult(self.reference_robot_quat, q_rel)

        # 3. Smoothing (EMA)
        if self.last_target_pos is not None:
            target_pos = (1 - self.alpha_pos) * self.last_target_pos + self.alpha_pos * target_pos
            target_quat = tf3d.quaternions.qnorm(
                (1 - self.alpha_rot) * self.last_target_quat + self.alpha_rot * target_quat
            )
        
        # 4. Velocity Limiting (Snap prevention)
        if self.last_target_pos is not None:
            diff = target_pos - self.last_target_pos
            dist = np.linalg.norm(diff)
            if dist > self.max_delta_pos:
                target_pos = self.last_target_pos + (diff / dist) * self.max_delta_pos

        self.last_target_pos = target_pos
        self.last_target_quat = target_quat

        # 5. Inverse Kinematics
        # Use dm_control qpos_from_site_pose
        self.physics.data.qpos[:NUM_JOINTS] = self.current_q
        ik_result = inverse_kinematics.qpos_from_site_pose(
            self.physics,
            site_name=self.site_name,
            target_pos=target_pos,
            target_quat=target_quat,
            tol=1e-4, # Relaxed from 1e-12 for better reachability
            max_steps=400, # Increased for more robust solving
        )
        self.physics.reset()

        if ik_result.success:
            self.last_q_goal = ik_result.qpos[:NUM_JOINTS].copy()
        else:
            self.get_logger().warn(
                "IK failed — holding last valid joint goal",
                throttle_duration_sec=1.0,
            )

    # ── Heartbeat Timer Callback ──────────────────────────────
    def _timer_cb(self):
        """Always publish a goal to keep the robot from diving."""
        if self.current_q is None:
            return # Wait for robot to be online

        # Goal selection
        if self.control_active and self.last_q_goal is not None:
            q_target = self.last_q_goal
            grasp_val = self.last_grasp
        else:
            # Idle Mode: Goal is current physical state
            q_target = self.current_q
            grasp_val = 0.0

        # ── Publish joint states (mimic GelloPublisher) ───────
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        msg.name = FR3_JOINT_NAMES
        msg.position = q_target.tolist()
        self.joint_pub.publish(msg)

        # ── Publish gripper command ───────────────────────────
        grip_msg = Float32()
        if self.control_active:
            grip_msg.data = 1.0 - grasp_val
        else:
            grip_msg.data = 1.0 # Open in idle
        self.gripper_pub.publish(grip_msg)

    # ── Cleanup ───────────────────────────────────────────────
    def destroy_node(self):
        self.running = False
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VRTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
