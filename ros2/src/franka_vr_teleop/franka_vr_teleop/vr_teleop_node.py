import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import socket
import struct
import threading
import time
import math

_PACK_FMT = "<18f"
_PACK_SIZE = struct.calcsize(_PACK_FMT)

class VRTeleopNode(Node):
    def __init__(self):
        super().__init__('vr_teleop_node')
        
        # Publish to the Cartesian Controller's expected topic
        self.pose_pub = self.create_publisher(PoseStamped, '/cartesian_impedance_example_controller/equilibrium_pose', 10)
        
        # Publish to the standard Gripper client topic
        self.gripper_pub = self.create_publisher(Float32, '/gripper/gripper_client/target_gripper_width_percent', 10)

        # Safety / Smoothing internal state
        self.last_pose = None
        self.alpha_pos = 0.2  # EMA smoothing factor for position
        self.alpha_rot = 0.5  # EMA smoothing factor for rotation
        self.max_delta_pos = 0.05  # clamp max 5cm per update
        
        self.udp_port = 9876
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.udp_port))
        self.sock.settimeout(1.0)
        
        self.get_logger().info(f"VR Teleop Node started. Listening on UDP port {self.udp_port}")

        self.thread = threading.Thread(target=self.udp_loop, daemon=True)
        self.thread.start()

    def clamp(self, val, min_v, max_v):
        return max(min_v, min(max_v, val))

    def udp_loop(self):
        while self.running and rclpy.ok():
            try:
                data, _ = self.sock.recvfrom(4096)
                if len(data) == _PACK_SIZE:
                    unpacked = struct.unpack(_PACK_FMT, data)
                    
                    # Layout: R_pos(3), R_quat(4), R_grasp, R_tracked, L_pos(3), L_quat(4), L_grasp, L_tracked
                    # Assuming we are controlling right arm for this instance
                    r_px, r_py, r_pz = unpacked[0], unpacked[1], unpacked[2]
                    r_qx, r_qy, r_qz, r_qw = unpacked[3], unpacked[4], unpacked[5], unpacked[6]
                    r_grasp = unpacked[7]
                    r_tracked = unpacked[8] > 0.5

                    if r_tracked:
                        self.process_target(r_px, r_py, r_pz, r_qx, r_qy, r_qz, r_qw, r_grasp)
                    else:
                        # Dead-man switch: we stop sending new poses if tracking is lost,
                        # robot stays at last pose
                        self.get_logger().info("Packet received, but tracking is LOST (Dead-man switch active)", throttle_duration_sec=2.0)
                else:
                    self.get_logger().warn(f"Received malformed UDP packet of size {len(data)}", throttle_duration_sec=2.0)
            except socket.timeout:
                self.get_logger().info("Waiting for UDP packets... (None received)", throttle_duration_sec=2.0)
                continue
            except Exception as e:
                self.get_logger().error(f"UDP Error: {e}")

    def process_target(self, px, py, pz, qx, qy, qz, qw, grasp):
        # 1. Coordinate transformation for Franka Base (if needed from Bridge Frame to Robot Frame)
        # Note: bridge.py already outputs in right-handed Z-up. 
        # But we may need an offset or rotation to match Franka's fr3_link0.
        # Let's apply a basic offset if necessary, or just use it as is if it's world calibrated.
        # For safety, you normally apply an offset from an initial pose. Here we do raw.
        tx, ty, tz = px, py, pz
        
        # 2. Safety: Velocity Clamping
        if self.last_pose is not None:
            dx = self.clamp(tx - self.last_pose['x'], -self.max_delta_pos, self.max_delta_pos)
            dy = self.clamp(ty - self.last_pose['y'], -self.max_delta_pos, self.max_delta_pos)
            dz = self.clamp(tz - self.last_pose['z'], -self.max_delta_pos, self.max_delta_pos)
            
            tx = self.last_pose['x'] + dx
            ty = self.last_pose['y'] + dy
            tz = self.last_pose['z'] + dz
            
            # 3. EMA Smoothing
            tx = self.alpha_pos * tx + (1 - self.alpha_pos) * self.last_pose['x']
            ty = self.alpha_pos * ty + (1 - self.alpha_pos) * self.last_pose['y']
            tz = self.alpha_pos * tz + (1 - self.alpha_pos) * self.last_pose['z']
        
        self.last_pose = {'x': tx, 'y': ty, 'z': tz}

        # 4. Construct and Publisher PoseStamped
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        
        msg.pose.position.x = tx
        msg.pose.position.y = ty
        # Apply a safety bound to z to not crash into the floor
        msg.pose.position.z = max(tz, 0.05) 
        
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        
        self.pose_pub.publish(msg)

        # 5. Gripper Command
        # The bridge.py sends grasp as float 1.0 (closed) to 0.0 (open)
        # The gripper manager expects width_percent. Let's map it.
        # Assuming grasp 1.0 -> width 0.0 (closed), and grasp 0.0 -> width 1.0 (opened)
        grip_msg = Float32()
        grip_msg.data = 1.0 - grasp
        self.gripper_pub.publish(grip_msg)

    def destroy_node(self):
        self.running = False
        if self.thread:
            self.thread.join()
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

if __name__ == '__main__':
    main()
