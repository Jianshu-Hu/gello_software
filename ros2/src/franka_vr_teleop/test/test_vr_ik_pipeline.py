#!/usr/bin/env python3
"""Standalone VR → IK pipeline test.

Runs the full VR teleoperation pipeline WITHOUT a physical robot and
WITHOUT ROS 2. This verifies:
  1. UDP reception from bridge.py
  2. IK solving via dm_control + MuJoCo
  3. Joint angle output

Quaternion conventions used in this codebase:
  - Bridge / UDP packets:   (x, y, z, w)
  - MuJoCo / dm_control:    (w, x, y, z)
  - transforms3d:           (w, x, y, z)

We use transforms3d for quaternion ↔ rotation-matrix conversions.
This is the same library used throughout the GELLO codebase
(via transforms3d._gohlketransforms in conversion_utils.py).

Usage:
  Terminal 1:  python test_vr_ik_pipeline.py
  Terminal 2:  python bridge.py --in-protocol tcp --in-port 8000 --out-port 9876
  (Move your VR controller and observe joint angles updating in Terminal 1)
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

# FR3 home position
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Position scaling
VR_POSITION_SCALE = 1.0


def _quat_xyzw_to_rotmat(qx, qy, qz, qw):
    """Convert quaternion (x,y,z,w) from UDP to 3×3 rotation matrix.

    transforms3d uses (w,x,y,z) order, so we reorder the input.
    """
    return transforms3d.quaternions.quat2mat([qw, qx, qy, qz])


def _rotmat_to_quat_wxyz(rotmat):
    """Convert 3×3 rotation matrix to quaternion in MuJoCo (w,x,y,z) order.

    transforms3d.quaternions.mat2quat returns (w,x,y,z) which matches
    MuJoCo's convention directly — no reordering needed.
    """
    return transforms3d.quaternions.mat2quat(rotmat)


def _find_fr3_xml():
    """Locate the fr3.xml model."""
    candidates = [
        # Robot PC path
        os.path.expanduser(
            "~/real-exp-work-branch/gello_software/third_party/"
            "mujoco_menagerie/franka_fr3/fr3.xml"
        ),
        # Dev machine path
        os.path.expanduser(
            "~/real_experiment_on _franka/gello_software/third_party/"
            "mujoco_menagerie/franka_fr3/fr3.xml"
        ),
        # Relative to this script
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


def main():
    # ── Load FR3 MuJoCo model ─────────────────────────────────
    xml_path = _find_fr3_xml()
    print(f"[INIT] Loading FR3 model from: {xml_path}")
    physics = mjcf.Physics.from_xml_path(xml_path)

    # Get home EE pose
    physics.data.qpos[:NUM_JOINTS] = HOME_Q
    physics.step()
    home_ee_pos = np.array(
        physics.named.data.site_xpos["attachment_site"]
    ).copy()
    home_ee_rot = np.array(
        physics.named.data.site_xmat["attachment_site"]
    ).reshape(3, 3).copy()

    print(f"[INIT] FR3 Home EE position: {np.round(home_ee_pos, 4)}")
    print(f"[INIT] FR3 Home joint angles: {np.round(np.degrees(HOME_Q), 1)} deg")

    # Current simulated joint state (starts at home)
    current_q = HOME_Q.copy()
    last_q_goal = HOME_Q.copy()

    # Relative control state
    control_active = False
    reference_vr_pos = None
    reference_vr_rot = None
    reference_robot_pos = None
    reference_robot_rot = None

    # Smoothing
    last_target_pos = None
    max_delta_pos = 0.02  # 2cm clamp
    alpha_pos = 0.3

    # ── Bind UDP ──────────────────────────────────────────────
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(2.0)
    print(f"\n[READY] Listening on UDP port {UDP_PORT}")
    print("        Run bridge.py in another terminal and move your VR controller.")
    print("        Joint angles will be printed below as they are solved.\n")
    print("=" * 80)

    ik_count = 0
    ik_fail_count = 0

    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                print("[WAIT] No UDP packets received…")
                continue

            if len(data) != _PACK_SIZE:
                print(f"[WARN] Bad packet size: {len(data)} (expected {_PACK_SIZE})")
                continue

            vals = struct.unpack(_PACK_FMT, data)
            r_px, r_py, r_pz = vals[0], vals[1], vals[2]
            r_qx, r_qy, r_qz, r_qw = vals[3], vals[4], vals[5], vals[6]
            r_grasp = vals[7]
            r_tracked = vals[8] > 0.5

            if not r_tracked:
                if control_active:
                    print("[INFO] Tracking lost — dead-man switch active, holding position")
                    control_active = False
                    reference_vr_pos = None
                continue

            # VR pose
            vr_pos = np.array([r_px, r_py, r_pz])
            vr_rot = _quat_xyzw_to_rotmat(r_qx, r_qy, r_qz, r_qw)

            # ── First-press: capture reference ────────────────
            if not control_active:
                control_active = True
                reference_vr_pos = vr_pos.copy()
                reference_vr_rot = vr_rot.copy()

                physics.data.qpos[:NUM_JOINTS] = current_q
                physics.step()
                reference_robot_pos = np.array(
                    physics.named.data.site_xpos["attachment_site"]
                ).copy()
                reference_robot_rot = np.array(
                    physics.named.data.site_xmat["attachment_site"]
                ).reshape(3, 3).copy()

                print(f"\n[ACTIVATED] VR reference captured")
                print(f"  VR ref pos:   {np.round(reference_vr_pos, 4)}")
                print(f"  Robot ref pos: {np.round(reference_robot_pos, 4)}")
                continue

            # ── Compute delta ─────────────────────────────────
            delta_pos = (vr_pos - reference_vr_pos) * VR_POSITION_SCALE
            delta_rot = vr_rot @ np.linalg.inv(reference_vr_rot)

            # identity VR-to-Robot transform for now
            target_pos = reference_robot_pos + delta_pos
            target_rot = delta_rot @ reference_robot_rot

            # ── Safety: velocity clamp ────────────────────────
            if last_target_pos is not None:
                diff = target_pos - last_target_pos
                norm = np.linalg.norm(diff)
                if norm > max_delta_pos:
                    diff = diff * (max_delta_pos / norm)
                    target_pos = last_target_pos + diff

            # ── Safety: Z floor ───────────────────────────────
            target_pos[2] = max(target_pos[2], 0.05)

            # ── EMA smoothing ─────────────────────────────────
            if last_target_pos is not None:
                target_pos = alpha_pos * target_pos + (1 - alpha_pos) * last_target_pos

            last_target_pos = target_pos.copy()

            # ── IK Solve ──────────────────────────────────────
            # transforms3d.mat2quat returns (w,x,y,z) which matches MuJoCo
            target_quat = _rotmat_to_quat_wxyz(target_rot)

            physics.data.qpos[:NUM_JOINTS] = current_q
            physics.step()

            ik_result = qpos_from_site_pose(
                physics,
                "attachment_site",
                target_pos=target_pos,
                target_quat=target_quat,
                tol=1e-12,
                max_steps=200,
            )
            physics.reset()

            ik_count += 1

            if ik_result.success:
                q_goal = ik_result.qpos[:NUM_JOINTS]
                last_q_goal = q_goal.copy()
                current_q = q_goal.copy()  # update simulated state

                # Print results
                q_deg = np.round(np.degrees(q_goal), 1)
                delta_mm = np.round(delta_pos * 1000, 1)
                print(
                    f"[IK #{ik_count:4d}] OK | "
                    f"Δpos(mm): [{delta_mm[0]:6.1f}, {delta_mm[1]:6.1f}, {delta_mm[2]:6.1f}] | "
                    f"Joints(°): [{q_deg[0]:6.1f}, {q_deg[1]:6.1f}, {q_deg[2]:6.1f}, "
                    f"{q_deg[3]:6.1f}, {q_deg[4]:6.1f}, {q_deg[5]:6.1f}, {q_deg[6]:6.1f}] | "
                    f"Grasp: {r_grasp:.1f}"
                )
            else:
                ik_fail_count += 1
                print(
                    f"[IK #{ik_count:4d}] FAILED (total fails: {ik_fail_count}) | "
                    f"target_pos: {np.round(target_pos, 4)} — using last valid"
                )

    except KeyboardInterrupt:
        print(f"\n\n[DONE] Total IK calls: {ik_count}, failures: {ik_fail_count}")
        if ik_count > 0:
            print(f"       Success rate: {100*(ik_count-ik_fail_count)/ik_count:.1f}%")


if __name__ == "__main__":
    main()
