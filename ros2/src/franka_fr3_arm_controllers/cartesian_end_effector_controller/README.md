# Cartesian End-Effector VR Teleoperation Controller

This directory contains the direct Cartesian teleoperation path for moving a
Franka FR3 end effector from a tracked VR controller pose. It is intentionally
separate from the older `franka_vr_teleop` package: this path does not solve
inverse kinematics in Python and does not publish joint targets. Instead, the
VR controller pose is converted into a Cartesian target pose, and a ROS 2
control controller sends that target through Franka's Cartesian pose command
interface.

The main goal of this implementation is to reduce the instability that came
from sending discrete IK-generated joint targets into the joint impedance
controller. Here, Cartesian target generation happens inside the controller
loop, with filtering, velocity limits, workspace limits, stale-command
handling, and tracking-loss handling.

## Files

```text
cartesian_end_effector_controller/
- config/cartesian_controllers.yaml
  Runtime parameters for controller_manager, the Cartesian controller, and
  the VR UDP bridge node.
- include/franka_fr3_arm_controllers/cartesian_end_effector_controller.hpp
  Controller class declaration and controller state.
- launch/franka_vr_cartesian_end_effector.launch.py
  Single launch file for robot description, ros2_control_node, controller
  spawners, and the VR Cartesian UDP bridge.
- scripts/vr_cartesian_teleop_node.py
  UDP receiver that converts Quest bridge packets into PoseStamped messages.
- src/cartesian_end_effector_controller.cpp
  ROS 2 control controller implementation.
```

The plugin is exported from the parent package as:

```text
franka_fr3_arm_controllers/CartesianEndEffectorController
```

The executable installed from `scripts/vr_cartesian_teleop_node.py` is:

```text
vr_cartesian_teleop_node
```

## High-Level System

The teleoperation system has four stages:

```text
Meta Quest / VR bridge
        |
        | UDP, port 9876, 18 little-endian float32 values
        v
vr_cartesian_teleop_node.py
        |
        | geometry_msgs/msg/PoseStamped
        | /cartesian_end_effector_controller/target_pose
        v
CartesianEndEffectorController
        |
        | Franka Cartesian pose command interfaces
        v
franka_ros2 hardware interface / FR3 robot
```

The Python node is only a transport adapter. It receives the tracked VR
controller pose and republishes it into ROS. The C++ controller owns the actual
robot command behavior.

## UDP Packet Contract

`vr_cartesian_teleop_node.py` expects the same packet layout used by the
existing VR bridge:

```python
struct format: "<18f"
packet size: 72 bytes
```

The 18 floats are:

```text
right_x, right_y, right_z,
right_qx, right_qy, right_qz, right_qw,
right_grasp, right_tracked,
left_x, left_y, left_z,
left_qx, left_qy, left_qz, left_qw,
left_grasp, left_tracked
```

The `hand` parameter selects which half of the packet is used:

```yaml
hand: right
```

If the selected controller is tracked, the node publishes:

```text
topic: /cartesian_end_effector_controller/target_pose
type:  geometry_msgs/msg/PoseStamped
frame: fr3_link0
```

If the selected controller is not tracked, the node publishes a `PoseStamped`
with a zero timestamp and identity orientation. The C++ controller treats a
zero timestamp as an explicit invalid command signal.

The `grasp` value is currently ignored by this Cartesian arm controller.
Gripper control is not implemented in this path yet.

## Controller Lifecycle

The C++ controller is a standard ROS 2 control lifecycle controller.

### Initialization

`on_init()` declares these parameters:

```yaml
arm_id: ""
target_pose_topic: "~/target_pose"
vr_position_scale: 1.0
command_timeout_sec: 0.25
filter_time_constant_sec: 0.08
max_linear_velocity: 0.12
max_angular_velocity: 0.70
workspace_radius: 0.35
min_z: 0.05
vr_to_robot_rotation_rpy: [0.0, 0.0, 0.0]
```

It also creates a `FrankaCartesianPoseInterface`. This is the semantic
component that exposes the Franka Cartesian pose command and state interfaces
to the controller.

### Configuration

`on_configure()` validates parameters, reads the target pose topic, and creates
a subscription for `geometry_msgs/msg/PoseStamped`.

The subscriber callback does not command the robot directly. It stores the
latest target pose into atomic variables behind a sequence counter so that the
realtime `update()` loop can read a coherent snapshot without using ROS
message objects.

### Activation

`on_activate()` assigns the loaned command and state interfaces from
ros2_control to `FrankaCartesianPoseInterface`. It then initializes the
commanded pose to the robot's current measured end-effector pose.

This matters because the controller should not jump to a stale target when it
is first activated. Until a valid VR target arrives, it commands the current
pose.

### Update Loop

`update()` runs at the controller manager update rate. The provided YAML uses:

```yaml
update_rate: 1000
```

Each update cycle does the following:

1. Read the latest VR pose from atomics.
2. Read the current robot end-effector pose from Franka state interfaces.
3. If this is the first update, initialize the commanded pose to the current
   robot pose.
4. Check whether the VR command is valid and fresh.
5. If no valid fresh command exists, reset the teleoperation reference and hold
   the current robot pose through the filter.
6. If a valid command exists and no reference is active, capture a new relative
   reference between VR pose and current robot pose.
7. Convert the relative VR motion into a target robot end-effector pose.
8. Filter and velocity-limit the target.
9. Send the final pose command to Franka's Cartesian pose command interfaces.

## Relative Mapping Method

This controller does not command the absolute VR pose directly. It uses a
relative mapping to avoid a jump when teleoperation starts or resumes after
tracking loss.

When the first valid VR command arrives after activation, timeout, or tracking
loss, the controller captures:

```text
reference_vr_position      = current VR controller position
reference_vr_orientation   = current VR controller orientation
reference_ee_position      = current robot end-effector position
reference_ee_orientation   = current robot end-effector orientation
```

After that, every new VR pose is interpreted as a delta from this captured
reference.

### Position

The position command is:

```text
delta_position = (vr_position - reference_vr_position) * vr_position_scale
delta_position = vr_to_robot_rotation * delta_position
target_position = reference_ee_position + delta_position
```

Then two safety limits are applied:

```text
if norm(delta_position) > workspace_radius:
    delta_position is clamped to workspace_radius

target_position.z = max(target_position.z, min_z)
```

With the default config:

```yaml
vr_position_scale: 1.0
workspace_radius: 0.30
min_z: 0.05
```

That means a 10 cm VR hand movement requests a 10 cm robot end-effector
movement, but the total relative motion from the captured reference is limited
to a 30 cm radius and cannot command below `z = 0.05`.

`vr_to_robot_rotation_rpy` is a fixed roll, pitch, yaw rotation in radians that
maps VR-frame deltas into `fr3_link0`. Leave it at `[0.0, 0.0, 0.0]` only if
the Quest bridge frame is already aligned with the Franka base frame.

### Orientation

The orientation command is also relative:

```text
delta_orientation = vr_orientation * inverse(reference_vr_orientation)
rotated_delta = vr_to_robot_rotation * delta_orientation * inverse(vr_to_robot_rotation)
target_orientation = rotated_delta * reference_ee_orientation
```

Quaternions are normalized before use. If an invalid quaternion is received,
the controller falls back to identity orientation.

## Filtering and Velocity Limits

Raw VR input can be jittery and can arrive at a lower rate than the robot
controller loop. `filterTargetPose_()` smooths the command before it reaches
the hardware interface.

The timestep used by the filter is clamped:

```text
dt in [0.001, 0.02] seconds
```

The low-pass filter coefficient is:

```text
alpha = dt / (filter_time_constant_sec + dt)
```

Then the position step is limited by:

```text
max_position_step = max_linear_velocity * dt
```

The orientation is filtered with quaternion slerp, then limited by:

```text
max_angular_step = max_angular_velocity * dt
```

With the default config:

```yaml
filter_time_constant_sec: 0.12
max_linear_velocity: 0.10
max_angular_velocity: 0.60
```

At a 1 kHz controller rate, the robot command changes gradually even if the VR
pose jumps.

## Tracking Loss and Timeout Behavior

There are two ways the command becomes invalid:

1. The Python node publishes a zero timestamp because the selected VR controller
   is not tracked.
2. The latest valid command is older than `command_timeout_sec`.

When either condition happens, the controller:

```text
reference_initialized_ = false
```

Then it filters the command back toward the robot's current measured
end-effector pose and sends that as the command. This behaves like a hold
instead of continuing stale VR motion.

When tracking returns, the next valid VR pose creates a new reference at the
current robot pose. This prevents a large jump caused by the operator's hand
moving while tracking was lost.

## Launching

Build the ROS 2 workspace from the `ros2` directory:

```bash
cd /home/pair/real_experiment_on\ _franka/gello_software/ros2
colcon build --packages-select franka_fr3_arm_controllers
source install/setup.bash
```

Launch the direct Cartesian VR teleoperation stack:

```bash
ros2 launch franka_fr3_arm_controllers franka_vr_cartesian_end_effector.launch.py \
  robot_ip:=172.16.0.2
```

If the VR frame is rotated relative to the robot base, pass the fixed frame
rotation as roll, pitch, yaw radians:

```bash
ros2 launch franka_fr3_arm_controllers franka_vr_cartesian_end_effector.launch.py \
  robot_ip:=172.16.0.2 \
  vr_to_robot_rotation_rpy:=0.0,0.0,1.5708
```

For fake hardware:

```bash
ros2 launch franka_fr3_arm_controllers franka_vr_cartesian_end_effector.launch.py \
  use_fake_hardware:=true \
  fake_sensor_commands:=true
```

The launch file starts:

```text
robot_state_publisher
controller_manager / ros2_control_node
franka_robot_state_broadcaster, unless fake hardware is enabled
joint_state_broadcaster
cartesian_end_effector_controller
vr_cartesian_teleop_node
```

The launch file uses:

```text
config/cartesian_controllers.yaml
```

## Useful Runtime Checks

Check that the controller is loaded and active:

```bash
ros2 control list_controllers
```

Check that the VR bridge node is receiving UDP and publishing poses:

```bash
ros2 topic echo /cartesian_end_effector_controller/target_pose
```

Check the command topic type:

```bash
ros2 topic info /cartesian_end_effector_controller/target_pose
```

If no VR packets are arriving, the Python node logs:

```text
Waiting for VR UDP packets...
```

If packets have the wrong size, it logs:

```text
Malformed UDP packet (<N> bytes)
```

## Parameters

### Controller Parameters

| Parameter | Default in YAML | Meaning |
| --- | ---: | --- |
| `arm_id` | `fr3` | Robot arm id used by the Franka interfaces. Must not be empty. |
| `target_pose_topic` | `/cartesian_end_effector_controller/target_pose` | ROS topic containing the VR target pose. |
| `vr_position_scale` | `1.0` | Multiplier from VR translation delta to robot translation delta. |
| `command_timeout_sec` | `0.25` | Maximum age of a valid VR command before the controller treats it as stale. |
| `filter_time_constant_sec` | `0.12` | Low-pass filter time constant for target pose smoothing. |
| `max_linear_velocity` | `0.10` | Maximum commanded Cartesian translation speed in m/s. |
| `max_angular_velocity` | `0.60` | Maximum commanded angular speed in rad/s. |
| `workspace_radius` | `0.30` | Maximum relative translation from the captured robot reference pose. |
| `min_z` | `0.05` | Minimum allowed target `z` position in the robot base frame. |
| `vr_to_robot_rotation_rpy` | `[0.0, 0.0, 0.0]` | Fixed roll, pitch, yaw rotation that maps VR-frame deltas into `fr3_link0`. |

### VR Bridge Node Parameters

| Parameter | Default in YAML | Meaning |
| --- | ---: | --- |
| `udp_port` | `9876` | UDP port for Quest bridge packets. |
| `target_pose_topic` | `/cartesian_end_effector_controller/target_pose` | Topic published by the bridge node. |
| `hand` | `right` | Selected controller half of the UDP packet. Must be `right` or `left`. |

## How This Differs From the IK-Based VR Path

The older path in `franka_vr_teleop` is:

```text
VR UDP -> Python IK -> /gello/joint_states -> JointImpedanceController
```

That path computes joint angles outside the realtime controller. It can suffer
from IK discontinuities, joint-space jumps, and target jitter when the VR
stream is not smooth.

This direct Cartesian path is:

```text
VR UDP -> PoseStamped -> CartesianEndEffectorController -> Franka Cartesian pose command
```

It avoids Python IK entirely. The C++ controller keeps the relative reference,
workspace clamp, timeout behavior, low-pass filter, and Cartesian velocity
limits close to the hardware command loop.

## Current Progress

As of the current implementation, the following pieces are in place:

- A ROS 2 control controller plugin exists:
  `franka_fr3_arm_controllers/CartesianEndEffectorController`.
- The controller reads Franka Cartesian pose state and writes Franka Cartesian
  pose commands through `FrankaCartesianPoseInterface`.
- The controller subscribes to `geometry_msgs/msg/PoseStamped` target poses.
- The realtime update loop uses sequence-checked atomic storage for the latest
  received VR pose.
- Relative position and orientation mapping is implemented.
- A configurable fixed VR-to-robot rotation is implemented through
  `vr_to_robot_rotation_rpy`.
- Command timeout and tracking-loss reset behavior are implemented.
- Position low-pass filtering, orientation slerp filtering, linear velocity
  limiting, and angular velocity limiting are implemented.
- Workspace radius and minimum-Z limits are implemented.
- A UDP bridge node exists for Quest-compatible 18-float packets.
- A launch file exists to start robot description, controller manager,
  broadcasters, the Cartesian controller, and the VR UDP bridge node.
- CMake install rules are present for the controller headers, plugin library,
  config file, launch file, and Python executable.

## Current Issues and Limitations

### 1. VR-to-Robot Frame Calibration Still Needs Hardware Tuning

The controller can rotate VR deltas into the robot base frame:

```text
delta_position = vr_to_robot_rotation
                * (vr_position - reference_vr_position)
                * vr_position_scale
target_position = reference_ee_position
                + delta_position
```

The rotation comes from `vr_to_robot_rotation_rpy`. This is enough to correct a
fixed axis mismatch between the Quest bridge frame and `fr3_link0`, but the
actual values still need to be validated on the robot server. If the operator
moves forward in VR and the robot moves sideways or backward, tune this launch
argument first.

There is still no full calibration transform with translation or TF support.
A robust follow-up would be:

```yaml
vr_to_robot_rotation_rpy: [0.0, 0.0, 0.0]
vr_to_robot_translation: [0.0, 0.0, 0.0]
```

or to use a TF frame published by calibration.

### 2. Orientation Mapping May Need a Tool-Frame Offset

The current orientation rule is mathematically relative but simple:

```text
target_orientation =
    (vr_orientation * inverse(reference_vr_orientation))
    * reference_ee_orientation
```

This assumes the VR controller orientation axes are already meaningful for the
FR3 end-effector frame. In practice, a hand controller's natural pointing
direction may not match the gripper or tool frame. A fixed controller-to-tool
orientation offset may be required.

### 3. No Explicit Clutch or Dead-Man Button Yet

Tracking loss acts like an automatic hold, but there is no deliberate operator
clutch button in this path. The UDP packet includes a `grasp` value, but the
controller currently ignores it. A production teleoperation workflow should add
an intentional enable/disable signal so the operator can reposition the VR
controller without moving the robot.

### 4. Gripper Control Is Not Implemented Here

This package controls only the arm Cartesian pose. The UDP `grasp` float is
available in the packet, but `vr_cartesian_teleop_node.py` does not publish a
gripper command and the C++ controller does not use it.

### 5. Safety Limits Are Conservative but Not a Full Safety Layer

The controller limits relative workspace radius, minimum `z`, linear velocity,
and angular velocity. It does not check:

- self-collision
- scene collision
- singularity proximity
- joint limits before the hardware controller resolves Cartesian motion
- task-specific forbidden regions

Run first with fake hardware, then with very low speed limits, and keep the
robot's external safety mechanisms active.

### 6. Current Behavior Has Not Been Documented as Hardware-Validated

The code path is wired and launchable, but this README should be treated as a
description of the implemented software behavior, not proof that the method has
been fully validated on the physical FR3. Before relying on it for experiments,
engineers should verify:

- the package builds cleanly in the target ROS 2 environment
- the controller activates on the target Franka ROS 2 stack
- the Cartesian command interfaces are available for the robot system version
- VR packet timing is stable
- the VR frame is aligned with `fr3_link0`
- tracking loss causes hold behavior without jumps
- the first valid command after tracking returns does not jump

## Recommended Next Engineering Steps

1. Tune `vr_to_robot_rotation_rpy` on the robot server with low speed limits.
2. Add a fixed controller-to-tool orientation offset.
3. Add an operator clutch/dead-man input, likely using the existing `grasp`
   float or another button field from the bridge.
4. Add full frame calibration, either with translation parameters or TF.
5. Add a fake UDP sender or launch-time test for this direct Cartesian path.
6. Add controller tests for timeout, reference reset, workspace clamping,
   minimum-Z clamping, and velocity limiting.
7. Add runtime diagnostics that publish whether the controller is tracking,
   timed out, holding, or actively commanding.
