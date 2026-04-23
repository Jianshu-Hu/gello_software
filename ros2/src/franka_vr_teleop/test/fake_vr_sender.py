#!/usr/bin/env python3
"""Fake VR sender — simulates VR controller movement for testing.

Sends synthetic UDP packets that mimic bridge.py output, moving the
virtual controller in a smooth circle pattern. Use this to test the
IK pipeline without a Meta Quest headset.

Usage:
  Terminal 1:  python test_vr_ik_pipeline.py
  Terminal 2:  python fake_vr_sender.py
"""

import math
import socket
import struct
import time

_PACK_FMT = "<18f"
TARGET = ("127.0.0.1", 9876)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Sending fake VR data to {TARGET[0]}:{TARGET[1]}")
    print("The virtual controller will trace a small circle in front of the robot.")
    print("Press Ctrl+C to stop.\n")

    t = 0.0
    dt = 1.0 / 30.0  # 30 Hz like bridge.py

    # Base position: roughly where a hand would be in front of the robot
    base_x = 0.4   # forward
    base_y = 0.0   # centered
    base_z = 0.5   # half-meter height

    # Circle radius
    radius = 0.08  # 8cm circle

    try:
        while True:
            # Smooth circle in the Y-Z plane
            px = base_x
            py = base_y + radius * math.cos(t * 0.5)
            pz = base_z + radius * math.sin(t * 0.5)

            # Identity quaternion (no rotation change)
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

            # No grasp
            grasp = 0.0

            # Tracked = True
            tracked = 1.0

            packet = struct.pack(
                _PACK_FMT,
                # Right hand
                px, py, pz,
                qx, qy, qz, qw,
                grasp, tracked,
                # Left hand (zeros)
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
                0.0, 0.0,
            )

            sock.sendto(packet, TARGET)

            if int(t / dt) % 30 == 0:  # Print once per second
                print(f"  t={t:5.1f}s  pos=({px:.3f}, {py:.3f}, {pz:.3f})")

            t += dt
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
