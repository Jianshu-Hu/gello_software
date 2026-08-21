import numpy as np

from franka_gello_state_publisher.gello_hardware import GelloHardware


OFFSETS = np.array([0.397, 3.188, 2.813, 4.639, 3.230, 4.829, 3.770])
JOINT_SIGNS = np.array([1, -1, 1, 1, 1, -1, 1])


def test_normalizes_joint7_multiturn_encoder_value() -> None:
    raw_positions = OFFSETS.copy()
    raw_positions[6] += -8.2687

    normalized = GelloHardware.normalize_joint_positions(
        raw_positions,
        OFFSETS,
        JOINT_SIGNS,
    )

    assert np.isclose(normalized[6], -8.2687 + 2 * np.pi)
    assert GelloHardware.JOINT_POSITION_LOWER_RAD[6] <= normalized[6]
    assert normalized[6] <= GelloHardware.JOINT_POSITION_UPPER_RAD[6]


def test_normalization_preserves_in_range_joint_positions() -> None:
    expected = np.array([0.5, -0.8, 1.2, -1.4, -0.7, 4.0, 2.6])
    raw_positions = expected * JOINT_SIGNS + OFFSETS

    normalized = GelloHardware.normalize_joint_positions(
        raw_positions,
        OFFSETS,
        JOINT_SIGNS,
    )

    np.testing.assert_allclose(normalized, expected)


def test_incremental_tracking_crosses_encoder_revolution_continuously() -> None:
    hardware = GelloHardware.__new__(GelloHardware)
    hardware._joint_signs = np.ones(7)
    hardware._prev_arm_joints_raw = np.zeros(7)
    hardware._prev_arm_joints_raw[6] = np.pi - 0.01
    hardware._prev_arm_joints = np.zeros(7)
    next_raw = hardware._prev_arm_joints_raw.copy()
    next_raw[6] = -np.pi + 0.01

    positions = hardware.process_arm_joint_positions(next_raw)

    assert np.isclose(positions[6], 0.02)
