import os
import socket
import struct
import numpy as np
import transforms3d
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from dm_control import mujoco
from ament_index_python.packages import get_package_share_directory

# ── Setup ─────────────────────────────────────────────────────
# FR3 joint names in standard order
FR3_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]
NUM_JOINTS = 7

# FR3 home position (User specified initial state)
HOME_Q = np.array([0.0, 0.5, 0.6, -1.9, -1.0, 1.7, 1.0])

# Position scaling: VR hand movement range -> robot workspace range
VR_POSITION_SCALE = 1.0  # 1-to-1 metric mapping

# Axis Mapping: [Hand_X, Hand_Y, Hand_Z] -> [Robot_Forward, Robot_Left, Robot_Up]
# Standard Quest 3 (OpenXR) -> MuJoCo Franka:
# Hand X (Right)  -> Robot -Y (Right)
# Hand Y (Up)     -> Robot Z  (Up)
# Hand Z (Forward)-> Robot X  (Forward)
# VR Z usually is +Forward in some streamers, or +Back in others. 
# We'll assume +Forward for now.
VR_TO_ROBOT_AXES = np.array([
    [0.0,  0.0,  1.0], # Robot X (Forward) is VR Z (Forward)
    [-1.0, 0.0,  0.0], # Robot Y (Left) is VR -X (Right -> Left)
    [0.0,  1.0,  0.0], # Robot Z (Up) is VR Y (Up)
])

class VRTeleopNode(Node):
    """ROS 2 node: VR UDP -> IK -> /gello/joint_states."""

    def __init__(self):
        super().__init__("vr_teleop_node")

        # ── Publishers ──────────────────────────────────────────────
        self.joint_pub = self.create_publisher(
            JointState, "gello/joint_states", 10
        )
        self.gripper_pub = self.create_publisher(
            Float32, "gello/gripper_action", 10
        )

        # ── Subscribers ─────────────────────────────────────────────
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10
        )

        # ── Robot Model (MuJoCo) ────────────────────────────────────
        # Using MuJoCo for Inverse Kinematics
        mujoco_pkg = get_package_share_directory("franka_vr_teleop")
        model_path = os.path.join(
            mujoco_pkg, "third_party", "mujoco_menagerie", "franka_fr3", "fr3.xml"
        )
        self.get_logger().info(f"Loading MuJoCo model from {model_path}")
        self.physics = mujoco.Physics.from_xml_path(model_path)
        self.site_name = "attachment_site"  # End-effector site in XML
        self.get_logger().info(f"IK target site: {self.site_name}")

        # ── UDP Server ──────────────────────────────────────────────
        self.udp_ip = "0.0.0.0"
        self.udp_port = 9876
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.settimeout(1.0)
        self.get_logger().info(f"VR Teleop IK Node started. Listening on UDP port {self.udp_port}")

        # ── State Variables ─────────────────────────────────────────
        self.current_q = None      # live joint states from robot
        self.startup_q = None      # Home base captured at launch
        self.reference_vr_pos = None# VR controller pose at trigger-press
        self.reference_vr_rot = None
        self.reference_robot_pos = None # Robot EE pose at trigger-press
        self.reference_robot_rot = None
        
        self.last_target_pos = None
        self.alpha_pos = 0.5       # EMA smoothing factor for position
        self.alpha_rot = 0.5       # EMA smoothing factor for rotation
        self.max_delta_pos = 0.05  # max change per step (safety)
        self.last_q_goal = None    # last successfully solved joint angles
        self.last_udp_time = self.get_clock().now()

        # ── Control mode state ─────────────────────────────────────
        self.control_active = False
        self.running = True
        self.thread = self.create_timer(0.001, self._udp_loop) # Fast loop

    def _joint_state_cb(self, msg):
        """Update the current robot joint positions."""
        q = [0.0] * NUM_JOINTS
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

    def _udp_loop(self):
        if not self.running or not rclpy.ok():
            return
            
        try:
            data, _ = self.sock.recvfrom(1024)
            if len(data) < 72: # 18 floats * 4 bytes
                return
            
            self.last_udp_time = self.get_clock().now()
            self._process_vr_pose(data)
            
        except socket.timeout:
            # We use publish logic even on timeout to keep robot in Safe Idle
            self._process_vr_pose(None)
        except Exception as e:
            self.get_logger().error(f"UDP error: {e}")

    def _process_vr_pose(self, data):
        if self.current_q is None:
            return  # Need robot state first

        if data is not None:
            # Struct: 18 floats (72 bytes)
            # [0-2]: L_pos, [3-6]: L_rot, [7]: L_grasp
            # [8-10]: R_pos, [11-14]: R_rot, [15]: R_grasp
            # [16]: index, [17]: menu
            floats = struct.unpack('18f', data)
            
            # Use Right Hand (index 8-15)
            vr_pos = np.array(floats[8:11])
            vr_rot_quat = np.array(floats[11:15]) # (w,x,y,z) hopefully
            grasp = floats[15] # Right Trigger
            
            # Convert VR Quat to Rotation Matrix
            vr_rot = transforms3d.quaternions.quat2mat(vr_rot_quat)
            
            # Teleop Engagement (Hold-to-Move)
            if not self.control_active:
                if grasp > 0.5:
                    self.control_active = True
                    self.reference_vr_pos = vr_pos.copy()
                    self.reference_vr_rot = vr_rot.copy()
                    
                    # Capture current robot EE pose as start point
                    self.physics.data.qpos[:NUM_JOINTS] = self.current_q
                    self.physics.step()
                    self.reference_robot_pos = np.array(self.physics.named.data.site_xpos[self.site_name]).copy()
                    self.reference_robot_rot = np.array(self.physics.named.data.site_xmat[self.site_name]).reshape(3,3).copy()
                    
                    self.last_target_pos = self.reference_robot_pos.copy()
                    self.get_logger().info("VR control activated — reference pose captured")
                return
            else:
                if grasp < 0.3:
                    self.control_active = False
                    self.get_logger().info("Trigger released — Holding position")
                    return
            
            # ── Compute Goal Pose ───────────────────────────────────
            # Calculate VR delta
            delta_pos_vr = vr_pos - self.reference_vr_pos
            delta_rot_vr = vr_rot @ np.linalg.inv(self.reference_vr_rot)
            
            # Map VR delta to Robot frame
            target_delta_pos = VR_TO_ROBOT_AXES @ delta_pos_vr
            target_pos = self.reference_robot_pos + target_delta_pos * VR_POSITION_SCALE
            
            # Rotation is more complex; for now we'll do a simple transform
            target_rot = VR_TO_ROBOT_AXES @ delta_rot_vr @ VR_TO_ROBOT_AXES.T @ self.reference_robot_rot
            
            # Safety: Position Clamping
            if self.last_target_pos is not None:
                diff = target_pos - self.last_target_pos
                dist = np.linalg.norm(diff)
                if dist > self.max_delta_pos:
                    target_pos = self.last_target_pos + (diff / dist) * self.max_delta_pos
            
            # Safety: Floor
            target_pos[2] = max(target_pos[2], 0.05)
            
            # Smoothing
            if self.last_target_pos is not None:
                target_pos = 0.5 * target_pos + 0.5 * self.last_target_pos
            self.last_target_pos = target_pos.copy()
            
            # Solve IK
            self.physics.data.qpos[:NUM_JOINTS] = self.current_q
            target_quat = transforms3d.quaternions.mat2quat(target_rot)
            
            # dm_control IK call
            from dm_control.utils import inverse_kinematics
            result = inverse_kinematics.qpos_from_site_pose(
                physics=self.physics,
                site_name=self.site_name,
                target_pos=target_pos,
                target_quat=target_quat,
                joint_names=FR3_JOINT_NAMES,
                tol=1e-3,
                max_steps=200
            )
            q_goal = self.physics.data.qpos[:NUM_JOINTS].copy()
            self.last_q_goal = q_goal.copy()

        # ── Publishing ──────────────────────────────────────────
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        msg.name = FR3_JOINT_NAMES
        
        if self.control_active and self.last_q_goal is not None:
            msg.position = self.last_q_goal.tolist()
        else:
            # Idle: Stay at startup position
            if self.startup_q is not None:
                msg.position = self.startup_q.tolist()
            else:
                return

        # Status Debug logging
        now = self.get_clock().now()
        if not hasattr(self, '_last_log'): self._last_log = now
        if (now - self._last_log).nanoseconds > 1e9:
            status = "ACTIVE" if self.control_active else "IDLE"
            vr_status = "CONNECTED" if (now - self.last_udp_time).nanoseconds < 1e9 else "WAITING"
            self.get_logger().info(f"Mode: {status} | VR: {vr_status} | Joints: {[round(x,2) for x in msg.position]}")
            self._last_log = now

        self.joint_pub.publish(msg)
        
        grip_msg = Float32()
        if self.control_active and data is not None:
            grip_msg.data = 1.0 - floats[15]
        else:
            grip_msg.data = 1.0
        self.gripper_pub.publish(grip_msg)

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
