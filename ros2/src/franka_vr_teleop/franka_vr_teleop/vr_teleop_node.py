"""VR Teleop Node — IK-based Joint Command Publisher.

Receives VR controller pose data via UDP (from bridge.py), solves Inverse
Kinematics using dm_control + MuJoCo to produce 7 joint angles for the
Franka FR3, then publishes those angles to ``gello/joint_states`` so the
existing JointImpedanceController can track them.

This node replaces the physical GELLO device: instead of reading joint
encoders from a puppet arm, it computes target joints from the VR hand pose.
"""

import os
import math
import socket
import struct
import threading

import numpy as np
import transforms3d
from dm_control import mjcf
from dm_control.utils.inverse_kinematics import qpos_from_site_pose

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

# ---------- UDP packet layout (matches bridge.py) ----------
# 18 × float32 = 72 bytes
# R_pos(3), R_quat(4), R_grasp(1), R_tracked(1),
# L_pos(3), L_quat(4), L_grasp(1), L_tracked(1)
_PACK_FMT = "<18f"
_PACK_SIZE = struct.calcsize(_PACK_FMT)

# ---------- Joint names (must match JointImpedanceController) ----------
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

# ---------- VR-to-Robot coordinate frame transform ----------
# Bridge outputs right-handed Z-up: x=front, y=left, z=up
# Franka base frame is also right-handed Z-up with x-forward
# We start with identity and can adjust axes/offset after first test
#
# Format: 4×4 homogeneous transform
VR_TO_ROBOT = np.eye(4)

# Translation offset: maps VR origin to Franka workspace center
# Adjust these after first test to calibrate
VR_ORIGIN_OFFSET = np.array([0.4, 0.0, 0.4])  # meters — roughly center of FR3 workspace

# Position scaling: VR hand movement range → robot workspace range
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
            Float32, "gripper/gripper_client/target_gripper_width_percent", 10
        )

        # ── Subscribe to actual robot joint states (warm-start seed) ──
        self.current_q = None  # filled by subscriber
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10
        )

        # ── Load MuJoCo model for IK ───────────────────────────────
        fr3_xml = self._find_fr3_xml()
        self.get_logger().info(f"Loading MuJoCo model from {fr3_xml}")
        self.physics = mjcf.Physics.from_xml_path(fr3_xml)
        self.site_name = "attachment_site"

        # Verify IK site exists
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
        self.alpha_pos = 0.8       # INCREASED responsiveness
        self.alpha_rot = 0.8       # INCREASED responsiveness
        self.max_delta_pos = 0.05  # Increased max speed
        self.last_q_goal = None    # last successfully solved joint angles
        self.startup_q = None      # The 'Home Base' captured at launch
        self.last_udp_time = self.get_clock().now()

        # ── Control mode state ─────────────────────────────────────
        self.control_active = False
        self.reference_vr_pos = None
        self.reference_vr_rot = None
        self.reference_robot_pos = None
        self.reference_robot_rot = None

        # ── UDP receiver ───────────────────────────────────────────
        self.udp_port = 9876
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.udp_port))
        self.sock.settimeout(1.0)
        self.get_logger().info(
            f"VR Teleop IK Node started. Listening on UDP port {self.udp_port}"
        )

        self.thread = threading.Thread(target=self._udp_loop, daemon=True)
        self.thread.start()

    # ── Locate the FR3 MuJoCo XML ─────────────────────────────────
    @staticmethod
    def _find_fr3_xml() -> str:
        """Walk up from this file to find the fr3.xml MuJoCo model."""
        # Try relative to the workspace root
        candidates = [
            # From the installed ROS workspace
            os.path.expanduser(
                "~/real-exp-work-branch/gello_software/third_party/"
                "mujoco_menagerie/franka_fr3/fr3.xml"
            ),
            # From the dev machine workspace
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..", "..",
                "third_party", "mujoco_menagerie", "franka_fr3", "fr3.xml"
            ),
        ]
        for p in candidates:
            p = os.path.abspath(p)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(
            f"Cannot find fr3.xml. Searched: {candidates}"
        )

    # ── Joint state subscriber callback ───────────────────────────
    def _joint_state_cb(self, msg):
        """Update the current robot joint positions."""
        q = [0.0] * NUM_JOINTS
        # Map ROS msg to our order (fr3_joint1...7)
        try:
            for i, name in enumerate(msg.name):
                if name in FR3_JOINT_NAMES:
                    idx = FR3_JOINT_NAMES.index(name)
                    q[idx] = msg.position[i]
            self.current_q = np.array(q)

            # Capture initial joint state once at startup
            if self.startup_q is None:
                self.startup_q = self.current_q.copy()
                self.get_logger().info(f"Initial joint state captured as Home Base: {self.startup_q.tolist()}")
        except Exception:
            pass

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

                    # Clutching logic: Only move if trigger is held
                    if r_grasp < 0.5:
                        if self.control_active:
                            self.get_logger().info("Trigger released — Holding position (Clutching)")
                            self.control_active = False
                            self.reference_vr_pos = None
                            self.last_target_pos = None
                        continue

                    if r_tracked:
                        self._process_vr_pose(
                            r_px, r_py, r_pz,
                            r_qx, r_qy, r_qz, r_qw,
                            r_grasp,
                        )
                    else:
                        if self.control_active:
                            self.get_logger().info(
                                "VR tracking lost — holding position (dead-man switch)",
                                throttle_duration_sec=2.0,
                            )
                            self.control_active = False
                            self.reference_vr_pos = None
                else:
                    self.get_logger().warn(
                        f"Malformed UDP packet ({len(data)} bytes)",
                        throttle_duration_sec=2.0,
                    )
            except socket.timeout:
                self.get_logger().info(
                    "Waiting for VR UDP packets…",
                    throttle_duration_sec=5.0,
                )
            except Exception as e:
                self.get_logger().error(f"UDP error: {e}")

    # ── Core VR → IK → Publish pipeline ──────────────────────────
    def _process_vr_pose(self, px, py, pz, qx, qy, qz, qw, grasp):
        """Convert a VR controller pose into joint angles and publish."""

        # Need current robot state for IK seed and reference pose
        if self.current_q is None:
            self.get_logger().info(
                "Waiting for /joint_states before starting IK…",
                throttle_duration_sec=2.0,
            )
            return

        vr_pos = np.array([px, py, pz])

        # Convert VR quaternion to rotation matrix
        # UDP sends (x,y,z,w), transforms3d uses (w,x,y,z)
        vr_rot = transforms3d.quaternions.quat2mat([qw, qx, qy, qz])

        # ── First-press activation (relative control) ─────────
        # ── First-press activation (relative control) ─────────
        if not self.control_active:
            # IMPORTANT: Wait for a fresh joint state to avoid using stale robot pose
            if self.current_q is None:
                return

            self.control_active = True
            self.reference_vr_pos = vr_pos.copy()
            self.reference_vr_rot = vr_rot.copy()

            # Run FK one last time on the LATEST joint angles to get reference
            self.physics.data.qpos[:NUM_JOINTS] = self.current_q
            mj.mj_forward(self.model, self.physics.data)
            self.reference_robot_pos = np.array(
                self.physics.named.data.site_xpos[self.site_name]
            ).copy()
            self.reference_robot_rot = np.array(
                self.physics.named.data.site_xmat[self.site_name]
            ).reshape(3, 3).copy()

            self.last_target_pos = self.reference_robot_pos.copy()
            self.get_logger().info("VR control activated — reference pose captured")
            return

        # ── Compute delta from VR reference ───────────────────
        delta_pos = (vr_pos - self.reference_vr_pos) * VR_POSITION_SCALE
        delta_rot = vr_rot @ np.linalg.inv(self.reference_vr_rot)

        # Apply VR-to-Robot frame transform to the delta
        R_vr2robot = VR_TO_ROBOT[:3, :3]
        delta_pos_robot = R_vr2robot @ delta_pos
        delta_rot_robot = R_vr2robot @ delta_rot @ R_vr2robot.T

        # Compute absolute target in robot frame
        target_pos = self.reference_robot_pos + delta_pos_robot
        target_rot = delta_rot_robot @ self.reference_robot_rot

        # --- Movement Diagnostic ---
        dist_vr = np.linalg.norm(vr_pos - self.reference_vr_pos)
        self.get_logger().info(
            f"CLUTCH HELD - VR Delta: {dist_vr:.3f}m | Target: {target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}",
            throttle_duration_sec=0.5
        )

        # ── Safety: position clamping ─────────────────────────
        if self.last_target_pos is not None:
            diff = target_pos - self.last_target_pos
            norm = np.linalg.norm(diff)
            if norm > self.max_delta_pos:
                diff = diff * (self.max_delta_pos / norm)
                target_pos = self.last_target_pos + diff

        # ── Safety: Z floor clamp ─────────────────────────────
        target_pos[2] = max(target_pos[2], 0.05)

        # ── EMA smoothing ─────────────────────────────────────
        if self.last_target_pos is not None:
            target_pos = (
                self.alpha_pos * target_pos
                + (1 - self.alpha_pos) * self.last_target_pos
            )

        self.last_target_pos = target_pos.copy()

        # Convert target rotation to quaternion for MuJoCo IK
        # transforms3d.mat2quat returns (w,x,y,z) — same as MuJoCo
        target_quat = transforms3d.quaternions.mat2quat(target_rot)

        # ── Solve IK ──────────────────────────────────────────
        # Seed the physics with current robot state for warm start
        self.physics.data.qpos[:NUM_JOINTS] = self.current_q
        self.physics.step()

        ik_result = qpos_from_site_pose(
            self.physics,
            self.site_name,
            target_pos=target_pos,
            target_quat=target_quat,
            tol=1e-4, # Relaxed from 1e-12 for better reachability
            max_steps=400, # Increased for more robust solving
        )
        self.physics.reset()

        if ik_result.success:
            q_goal = ik_result.qpos[:NUM_JOINTS]
            self.last_q_goal = q_goal.copy()
        else:
            self.get_logger().warn(
                "IK failed — holding last valid joint goal",
                throttle_duration_sec=1.0,
            )
            if self.last_q_goal is not None:
                q_goal = self.last_q_goal
            else:
                q_goal = self.current_q # Fallback to actual
                if q_goal is None: return

        # ── Publish joint states ──────────────────────────────
        # Follow pseudo-code: If inactive, stay at startup_q. If active, follow IK.
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        msg.name = FR3_JOINT_NAMES

        if self.control_active:
            msg.position = q_goal.tolist()
        else:
            # Stay at (or go to) the initial startup pose
            if self.startup_q is not None:
                msg.position = self.startup_q.tolist()
            else:
                # Still waiting for robot to report its state
                return

        # ── Debug Logging (Throttle to 1Hz) ──────────────────
        now = self.get_clock().now()
        if not hasattr(self, '_last_debug_time'): self._last_debug_time = now
        
        if (now - self._last_debug_time).nanoseconds > 1e9: # Every 1s
            status = "ACTIVE" if self.control_active else "IDLE"
            q_out = q_goal if self.control_active else self.startup_q
            q_str = [round(x, 3) for x in q_out.tolist()] if q_out is not None else "None"
            vr_status = "CONNECTED" if (now - self.last_udp_time).nanoseconds < 1e9 else "WAITING"
            
            self.get_logger().info(
                f"\n--- VR TELEOP STATUS ---\n"
                f"Mode: {status} | VR Stream: {vr_status}\n"
                f"Startup Q (Home): {self.startup_q.tolist() if self.startup_q is not None else 'WAITING'}\n"
                f"Current Publish: {q_str}\n"
                f"------------------------"
            )
            self._last_debug_time = now

        self.joint_pub.publish(msg)

        # ── Publish gripper command ───────────────────────────
        grip_msg = Float32()
        if self.control_active:
            grip_msg.data = 1.0 - grasp  # 1.0=open, 0.0=closed
        else:
            grip_msg.data = 1.0 # Keep open in idle period
        self.gripper_pub.publish(grip_msg)

    # ── Cleanup ───────────────────────────────────────────────
    def destroy_node(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
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
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
