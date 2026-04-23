#!/usr/bin/env python3
"""Standalone VR → IK pipeline test.

Runs the full VR teleoperation pipeline WITHOUT a physical robot and
WITHOUT ROS 2. This verifies:
  1. UDP reception from bridge.py
  2. IK solving via dm_control + MuJoCo
  3. Joint angle output (ROS-style formatting)

Usage:
  Terminal 1:  python test_vr_ik_pipeline.py
  Terminal 2:  python bridge.py ...
"""

import math
import os
import socket
import struct
import time

import numpy as np
import transforms3d
from dm_control import mjcf
from dm_control.utils.inverse_kinematics import qpos_from_site_pose

# ---------- Config ----------
UDP_PORT = 9876
_PACK_FMT = "<18f"
_PACK_SIZE = struct.calcsize(_PACK_FMT)

FR3_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]
NUM_JOINTS = 7

# FR3 home position (User specified initial state)
HOME_Q = np.array([0.0, 0.5, 0.6, -1.9, -1.0, 1.7, 1.0])

# Position scaling (1.0 = 1-to-1 metric mapping)
VR_POSITION_SCALE = 1.0


def _quat_xyzw_to_rotmat(qx, qy, qz, qw):
    return transforms3d.quaternions.quat2mat([qw, qx, qy, qz])


def _rotmat_to_quat_wxyz(rotmat):
    return transforms3d.quaternions.mat2quat(rotmat)


def _print_ros_style(q, q_vel, grasp):
    """Prints joint states in ROS-style YAML format."""
    now = time.time()
    sec = int(now)
    nanosec = int((now - sec) * 1e9)

    print("\n---")
    print("header:")
    print("  stamp:")
    print(f"    sec: {sec}")
    print(f"    nanosec: {nanosec}")
    print("  frame_id: base_link")
    print("name:")
    for name in FR3_JOINT_NAMES:
        print(f"- {name}")
    print("position:")
    for val in q:
        print(f"- {val:.10f}")
    print("velocity:")
    for val in q_vel:
        print(f"- {val:.10f}")
    print("effort:")
    for _ in range(NUM_JOINTS):
        print("- 0.0")
    print(f"grasp: {grasp:.3f}")
    print("---\n")


def _find_fr3_xml():
    candidates = [
        os.path.expanduser("~/real-exp-work-branch/gello_software/third_party/mujoco_menagerie/franka_fr3/fr3.xml"),
        os.path.expanduser("~/real_experiment_on _franka/gello_software/third_party/mujoco_menagerie/franka_fr3/fr3.xml"),
    ]
    for p in candidates:
        if os.path.isfile(p): return p
    raise FileNotFoundError("Cannot find fr3.xml")


def main():
    xml_path = _find_fr3_xml()
    print(f"[INIT] Loading FR3 model: {xml_path}")
    physics = mjcf.Physics.from_xml_path(xml_path)

    # State tracking
    current_q = HOME_Q.copy()
    last_q = HOME_Q.copy()
    
    # Relative control state
    control_active = False
    reference_vr_pos = None
    reference_vr_rot = None
    reference_robot_pos = None
    reference_robot_rot = None

    # Smoothing/Safety
    last_target_pos = None
    max_delta_pos = 0.02
    alpha_pos = 0.3
    max_joint_jump = np.deg2rad(45) # 45 degree limit per packet

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(2.0)
    print(f"\n[READY] Listening on UDP {UDP_PORT}. Press Ctrl+C to stop.\n")

    ik_count = 0
    start_time = time.time()

    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue

            if len(data) != _PACK_SIZE: continue

            vals = struct.unpack(_PACK_FMT, data)
            vr_pos = np.array([vals[0], vals[1], vals[2]])
            vr_rot = _quat_xyzw_to_rotmat(vals[3], vals[4], vals[5], vals[6])
            grasp = vals[7]
            tracked = vals[8] > 0.5

            # Clutching logic: Only move if trigger is held
            if grasp < 0.5:
                if control_active:
                    print("[INFO] Trigger released — Holding position (Clutching)")
                    control_active = False
                    reference_vr_pos = None
                    last_target_pos = None # Reset smoothing
                continue

            if not tracked:
                control_active = False
                continue

            # ── Capture Reference ─────────────────────────────
            if not control_active:
                control_active = True
                reference_vr_pos, reference_vr_rot = vr_pos.copy(), vr_rot.copy()
                physics.data.qpos[:NUM_JOINTS] = current_q
                physics.step()
                reference_robot_pos = np.array(physics.named.data.site_xpos["attachment_site"]).copy()
                reference_robot_rot = np.array(physics.named.data.site_xmat["attachment_site"]).reshape(3, 3).copy()
                print("[ACTIVE] Reference captured.")
                continue

            # ── Compute Target ────────────────────────────────
            delta_pos = (vr_pos - reference_vr_pos) * VR_POSITION_SCALE
            delta_rot = vr_rot @ np.linalg.inv(reference_vr_rot)
            target_pos = reference_robot_pos + delta_pos
            target_rot = delta_rot @ reference_robot_rot

            # ── Safety Filters ────────────────────────────────
            if last_target_pos is not None:
                d = target_pos - last_target_pos
                if np.linalg.norm(d) > max_delta_pos:
                    target_pos = last_target_pos + d * (max_delta_pos / np.linalg.norm(d))
            target_pos[2] = max(target_pos[2], 0.05)
            if last_target_pos is not None:
                target_pos = alpha_pos * target_pos + (1 - alpha_pos) * last_target_pos
            last_target_pos = target_pos.copy()

            # ── IK Solve ──────────────────────────────────────
            target_quat = _rotmat_to_quat_wxyz(target_rot)
            physics.data.qpos[:NUM_JOINTS] = current_q
            physics.step()
            ik_result = qpos_from_site_pose(physics, "attachment_site", target_pos=target_pos, target_quat=target_quat,
                tol=1e-4, # Relaxed from 1e-12 for better reachability
                max_steps=400, # Increased for more robust solving
            )
            physics.reset()

            ik_count += 1

            if ik_result.success:
                new_q = ik_result.qpos[:NUM_JOINTS]
                
                # Check for large joint jumps (safety)
                if np.any(np.abs(new_q - current_q) > max_joint_jump):
                    print(f"[WARN] IK jump detected on IK #{ik_count} - holding position")
                else:
                    q_vel = (new_q - last_q) / (1.0/30.0) # Assumed 30Hz
                    last_q = current_q.copy()
                    current_q = new_q.copy()

                    # Throttled ROS-style print (once per sec/30 packets)
                    if ik_count % 30 == 0:
                        _print_ros_style(current_q, q_vel, grasp)
                    else:
                        q_deg = np.degrees(current_q)
                        print(f"[IK #{ik_count:5d}] OK | Joints: {np.round(q_deg, 1)}")

    except KeyboardInterrupt:
        print("\n[DONE]")

if __name__ == "__main__":
    main()
