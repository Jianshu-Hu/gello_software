from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np


PACKAGE_SRC = (
    Path(__file__).resolve().parents[1]
    / "ros2"
    / "src"
    / "franka_gello_state_publisher"
)
sys.path.insert(0, str(PACKAGE_SRC))

try:
    from franka_gello_state_publisher.dynamixel.driver import DynamixelDriver
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import the GELLO ROS 2 Dynamixel driver. Run this script from the "
        "real-exp checkout and install/source the gello_software ROS 2 dependencies first."
    ) from exc


ACTIVE_CONFIRMATION = "DRIVE_ONE_GELLO_JOINT"
EXTENDED_ACTIVE_CONFIRMATION = "DRIVE_ONE_GELLO_JOINT_EXTENDED"
CURRENT_BASED_POSITION_MODE = 5
MAX_TEST_DELTA_DEG = 3.0
MAX_TEST_CURRENT_MA = 150
MAX_TEST_KP_P = 100
EXTENDED_MAX_TEST_DELTA_DEG = 20.0
EXTENDED_MAX_TEST_CURRENT_MA = 300
EXTENDED_MAX_TEST_KP_P = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit GELLO Dynamixel hardware and optionally perform a current-limited, "
            "single-joint, small-angle round-trip drive test. Audit mode is the default."
        )
    )
    parser.add_argument(
        "--port",
        required=True,
        help="GELLO serial port, for example /dev/ttyUSB_left.",
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
        help="Dynamixel IDs on this GELLO (default: 1 2 3 4 5 6 7 8).",
    )
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument(
        "--num-arm-joints",
        type=int,
        default=7,
        help="Number of arm joints before the optional gripper (default: 7).",
    )
    parser.add_argument(
        "--active-test",
        action="store_true",
        help="Enable the single-joint motion test. Without this flag all torque stays off.",
    )
    parser.add_argument(
        "--extended-test",
        action="store_true",
        help=(
            "Allow the reviewed extended envelope: up to 20 degrees, 300 mA, and kp=1000. "
            "Requires a separate confirmation phrase."
        ),
    )
    parser.add_argument(
        "--joint-id",
        type=int,
        help="Dynamixel ID to test. Required with --active-test and must be an arm joint.",
    )
    parser.add_argument(
        "--expected-model-number",
        type=int,
        help=(
            "Model number printed by a previous audit for --joint-id. Required with "
            "--active-test so a different actuator cannot be driven accidentally."
        ),
    )
    parser.add_argument(
        "--delta-deg",
        type=float,
        default=1.0,
        help="Raw motor-angle excursion before returning to start (default: 1 degree).",
    )
    parser.add_argument(
        "--goal-current-ma",
        type=int,
        default=50,
        help=(
            "Goal-current limit for the tested joint (default: 50 mA; "
            "normal maximum: 150 mA; extended maximum: 300 mA)."
        ),
    )
    parser.add_argument(
        "--kp-p",
        type=int,
        default=20,
        help=(
            "Position proportional gain used only for the tested joint "
            f"(default: 20, maximum: {MAX_TEST_KP_P})."
        ),
    )
    parser.add_argument(
        "--travel-time-sec",
        type=float,
        default=2.0,
        help="Time for each outbound/return interpolation (default: 2 seconds).",
    )
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=25.0,
        help="Command and monitoring frequency (default: 25 Hz).",
    )
    parser.add_argument(
        "--max-temperature-c",
        type=int,
        default=50,
        help="Abort threshold for the tested servo temperature (default: 50 C).",
    )
    parser.add_argument(
        "--abort-current-ma",
        type=int,
        default=150,
        help="Observed-current abort threshold (default: 150 mA).",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help=f"Required with --active-test. Must equal {ACTIVE_CONFIRMATION!r}.",
    )
    return parser.parse_args()


def only_at(index: int, value: int, count: int) -> list[int | None]:
    values: list[int | None] = [None] * count
    values[index] = int(value)
    return values


def signed_value(raw_value: int, bits: int) -> int:
    raw_value &= (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    full_range = 1 << bits
    return raw_value - full_range if raw_value & sign_bit else raw_value


def read_signed_current_ma(driver: DynamixelDriver) -> list[int]:
    return [signed_value(int(value), 16) for value in driver.read_value_by_name("present_current")]


def read_snapshot(driver: DynamixelDriver) -> dict[str, list[int]]:
    return {
        "model_number": [int(v) for v in driver.read_value_by_name("model_number")],
        "operating_mode": [int(v) for v in driver.read_value_by_name("operating_mode")],
        "torque_enable": [int(v) for v in driver.read_value_by_name("torque_enable")],
        "current_limit": [int(v) for v in driver.read_value_by_name("current_limit")],
        "goal_current": [int(v) for v in driver.read_value_by_name("goal_current")],
        "kp_p": [int(v) for v in driver.read_value_by_name("kp_p")],
        "kp_i": [int(v) for v in driver.read_value_by_name("kp_i")],
        "kp_d": [int(v) for v in driver.read_value_by_name("kp_d")],
        "present_input_voltage": [
            int(v) for v in driver.read_value_by_name("present_input_voltage")
        ],
        "present_temperature": [
            int(v) for v in driver.read_value_by_name("present_temperature")
        ],
        "present_position": [int(v) for v in driver.read_value_by_name("present_position")],
        "present_current": read_signed_current_ma(driver),
    }


def print_snapshot(ids: Sequence[int], driver: DynamixelDriver, snapshot: dict[str, list[int]]) -> None:
    print("GELLO Dynamixel audit")
    print("---------------------")
    print(f"Port: {driver._port}")
    print(f"Baudrate: {driver._baudrate}")
    print()
    print(
        " ID | model | mode | torque | voltage | temp | current | position"
    )
    print("----+-------+------+--------+---------+------+---------+----------------")
    for index, dxl_id in enumerate(ids):
        position_rad = driver._pulses_to_rad(snapshot["present_position"])[index]
        print(
            f"{dxl_id:>3} |"
            f" {snapshot['model_number'][index]:>5} |"
            f" {snapshot['operating_mode'][index]:>4} |"
            f" {snapshot['torque_enable'][index]:>6} |"
            f" {snapshot['present_input_voltage'][index] / 10.0:>6.1f} V |"
            f" {snapshot['present_temperature'][index]:>3} C |"
            f" {snapshot['present_current'][index]:>5} mA |"
            f" {position_rad:>8.4f} rad ({math.degrees(position_rad):>7.2f} deg)"
        )


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.ids)) != len(args.ids):
        raise SystemExit("--ids contains duplicate Dynamixel IDs.")
    if args.num_arm_joints < 1 or args.num_arm_joints > len(args.ids):
        raise SystemExit("--num-arm-joints must be between 1 and the number of IDs.")
    if args.extended_test and not args.active_test:
        raise SystemExit("--extended-test requires --active-test.")
    if not args.active_test:
        return
    required_confirmation = (
        EXTENDED_ACTIVE_CONFIRMATION if args.extended_test else ACTIVE_CONFIRMATION
    )
    if args.confirm != required_confirmation:
        raise SystemExit(
            "Refusing active motion: --active-test requires "
            f"--confirm {required_confirmation}."
        )
    if args.joint_id is None:
        raise SystemExit("--joint-id is required with --active-test.")
    if args.expected_model_number is None:
        raise SystemExit("--expected-model-number is required with --active-test.")
    arm_ids = args.ids[: args.num_arm_joints]
    if args.joint_id not in arm_ids:
        raise SystemExit(f"--joint-id must be one of the arm IDs: {arm_ids}")
    max_delta_deg = EXTENDED_MAX_TEST_DELTA_DEG if args.extended_test else MAX_TEST_DELTA_DEG
    max_current_ma = (
        EXTENDED_MAX_TEST_CURRENT_MA if args.extended_test else MAX_TEST_CURRENT_MA
    )
    max_kp_p = EXTENDED_MAX_TEST_KP_P if args.extended_test else MAX_TEST_KP_P
    if not 0.0 < abs(args.delta_deg) <= max_delta_deg:
        raise SystemExit(
            f"Absolute --delta-deg must be greater than 0 and at most {max_delta_deg}."
        )
    if not 1 <= args.goal_current_ma <= max_current_ma:
        raise SystemExit(
            f"--goal-current-ma must be between 1 and {max_current_ma}."
        )
    if args.abort_current_ma < args.goal_current_ma:
        raise SystemExit("--abort-current-ma must be at least --goal-current-ma.")
    if args.travel_time_sec < 1.0:
        raise SystemExit("--travel-time-sec must be at least 1 second.")
    if not 5.0 <= args.sample_hz <= 50.0:
        raise SystemExit("--sample-hz must be between 5 and 50 Hz.")
    if not 0 <= args.kp_p <= max_kp_p:
        raise SystemExit(
            f"--kp-p must be between 0 and {max_kp_p} for this feasibility test."
        )


def monitor_joint(
    driver: DynamixelDriver,
    index: int,
    start_pulse: int,
    max_excursion_pulses: int,
    max_temperature_c: int,
    abort_current_ma: int,
) -> tuple[int, int, int]:
    position = int(driver.read_value_by_name("present_position")[index])
    current = int(read_signed_current_ma(driver)[index])
    temperature = int(driver.read_value_by_name("present_temperature")[index])
    if abs(position - start_pulse) > max_excursion_pulses:
        raise RuntimeError(
            "Position excursion exceeded the feasibility-test envelope: "
            f"start={start_pulse}, current={position}, limit={max_excursion_pulses} pulses."
        )
    if abs(current) > abort_current_ma:
        raise RuntimeError(
            f"Observed current {current} mA exceeded abort limit {abort_current_ma} mA."
        )
    if temperature >= max_temperature_c:
        raise RuntimeError(
            f"Servo temperature {temperature} C reached abort limit {max_temperature_c} C."
        )
    return position, current, temperature


def interpolate_goal(
    driver: DynamixelDriver,
    index: int,
    start_goal: int,
    end_goal: int,
    args: argparse.Namespace,
    initial_position: int,
    max_excursion_pulses: int,
) -> tuple[int, int]:
    sample_count = max(1, int(math.ceil(args.travel_time_sec * args.sample_hz)))
    period = 1.0 / args.sample_hz
    peak_current = 0
    last_position = initial_position
    for step in range(1, sample_count + 1):
        fraction = step / sample_count
        goal = int(round(start_goal + fraction * (end_goal - start_goal)))
        driver.write_value_by_name("goal_position", only_at(index, goal, len(args.ids)))
        time.sleep(period)
        last_position, current, _ = monitor_joint(
            driver,
            index,
            initial_position,
            max_excursion_pulses,
            args.max_temperature_c,
            args.abort_current_ma,
        )
        peak_current = max(peak_current, abs(current))
    return last_position, peak_current


def run_active_test(
    driver: DynamixelDriver,
    args: argparse.Namespace,
    initial_snapshot: dict[str, list[int]],
) -> None:
    index = args.ids.index(args.joint_id)
    servo_count = len(args.ids)
    initial_position = int(initial_snapshot["present_position"][index])
    voltage = initial_snapshot["present_input_voltage"][index] / 10.0
    current_limit = int(initial_snapshot["current_limit"][index])
    model_number = int(initial_snapshot["model_number"][index])

    if model_number != args.expected_model_number:
        raise SystemExit(
            f"Refusing active test: joint ID {args.joint_id} reports model {model_number}, "
            f"not expected model {args.expected_model_number}."
        )

    if not 3.7 <= voltage <= 6.0:
        raise SystemExit(
            f"Refusing active test: servo voltage is {voltage:.1f} V; expected XL330 range is 3.7-6.0 V."
        )
    if current_limit < args.goal_current_ma:
        raise SystemExit(
            f"Refusing active test: configured current limit {current_limit} mA is below "
            f"requested goal current {args.goal_current_ma} mA."
        )

    delta_pulses = int(round(math.radians(args.delta_deg) / (2.0 * math.pi) * 4095))
    if delta_pulses == 0:
        raise SystemExit("Requested movement rounds to zero encoder pulses.")
    target_position = initial_position + delta_pulses
    safety_margin_pulses = int(round(math.radians(1.0) / (2.0 * math.pi) * 4095))
    max_excursion_pulses = 2 * abs(delta_pulses) + safety_margin_pulses

    print()
    print("ACTIVE SINGLE-JOINT TEST")
    print("------------------------")
    print(f"Joint ID: {args.joint_id}")
    print(f"Commanded excursion: {args.delta_deg:.3f} deg ({delta_pulses} pulses)")
    print(f"Goal current: {args.goal_current_ma} mA")
    print(f"Start pulse: {initial_position}; outbound target pulse: {target_position}")
    print("The tested GELLO joint may move now.")

    all_torque_off = [0] * servo_count
    driver.write_value_by_name("torque_enable", all_torque_off)
    driver.write_value_by_name(
        "operating_mode",
        only_at(index, CURRENT_BASED_POSITION_MODE, servo_count),
    )
    driver.write_value_by_name(
        "goal_current",
        only_at(index, args.goal_current_ma, servo_count),
    )
    driver.write_value_by_name("kp_i", only_at(index, 0, servo_count))
    driver.write_value_by_name("kp_d", only_at(index, 0, servo_count))
    driver.write_value_by_name("kp_p", only_at(index, args.kp_p, servo_count))

    # Establish a hold target at the measured position before enabling any torque.
    driver.write_value_by_name(
        "goal_position",
        only_at(index, initial_position, servo_count),
    )
    driver.write_value_by_name("torque_enable", only_at(index, 1, servo_count))
    time.sleep(0.5)
    monitor_joint(
        driver,
        index,
        initial_position,
        max_excursion_pulses,
        args.max_temperature_c,
        args.abort_current_ma,
    )

    outbound_position, outbound_peak_current = interpolate_goal(
        driver,
        index,
        initial_position,
        target_position,
        args,
        initial_position,
        max_excursion_pulses,
    )
    time.sleep(0.5)
    outbound_position, current, _ = monitor_joint(
        driver,
        index,
        initial_position,
        max_excursion_pulses,
        args.max_temperature_c,
        args.abort_current_ma,
    )
    outbound_peak_current = max(outbound_peak_current, abs(current))

    _, return_peak_current = interpolate_goal(
        driver,
        index,
        target_position,
        initial_position,
        args,
        initial_position,
        max_excursion_pulses,
    )
    time.sleep(0.5)
    final_position, final_current, final_temperature = monitor_joint(
        driver,
        index,
        initial_position,
        max_excursion_pulses,
        args.max_temperature_c,
        args.abort_current_ma,
    )

    observed_delta_deg = math.degrees(
        float(driver._pulses_to_rad([outbound_position - initial_position])[0])
    )
    return_error_deg = math.degrees(
        float(driver._pulses_to_rad([final_position - initial_position])[0])
    )
    peak_current = max(outbound_peak_current, return_peak_current, abs(final_current))

    print()
    print("Test result")
    print("-----------")
    print(f"Observed outbound movement: {observed_delta_deg:.3f} deg")
    print(f"Return error: {return_error_deg:.3f} deg")
    print(f"Peak observed current: {peak_current} mA")
    print(f"Final temperature: {final_temperature} C")
    if abs(observed_delta_deg) >= 0.5 * abs(args.delta_deg):
        print("RESULT: active-drive feasibility demonstrated for this joint at this current limit.")
    else:
        print(
            "RESULT: inconclusive. The command path worked, but movement was below half of the "
            "requested excursion; do not assume the motor is absent or raise current without review."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    driver: DynamixelDriver | None = None
    snapshot: dict[str, list[int]] | None = None
    try:
        driver = DynamixelDriver(
            args.ids,
            port=args.port,
            baudrate=args.baudrate,
            motor_type="xl330",
            autostart_polling=False,
        )
        # The driver constructor disables torque; repeat explicitly before the audit.
        driver.write_value_by_name("torque_enable", [0] * len(args.ids))
        snapshot = read_snapshot(driver)
        print_snapshot(args.ids, driver, snapshot)

        if not args.active_test:
            print()
            print("AUDIT COMPLETE: communication works and all listed servos remain torque-off.")
            print(
                "This does not yet prove active-drive feasibility. Use --active-test only after "
                "isolating Franka, clearing the GELLO workspace, and verifying its power system."
            )
            return

        run_active_test(driver, args, snapshot)
    except KeyboardInterrupt:
        print("\nInterrupted by operator; disabling all GELLO torque.")
        raise SystemExit(130) from None
    finally:
        if driver is not None:
            try:
                driver.write_value_by_name("torque_enable", [0] * len(args.ids))
                print("All GELLO Dynamixel torque disabled.")
            except Exception as exc:
                print(f"WARNING: failed to confirm torque-off over serial: {exc}", file=sys.stderr)
            if snapshot is not None and args.active_test and args.joint_id in args.ids:
                index = args.ids.index(args.joint_id)
                try:
                    driver.write_value_by_name(
                        "operating_mode",
                        only_at(index, snapshot["operating_mode"][index], len(args.ids)),
                    )
                    driver.write_value_by_name(
                        "goal_current",
                        only_at(index, snapshot["goal_current"][index], len(args.ids)),
                    )
                    driver.write_value_by_name(
                        "kp_i", only_at(index, snapshot["kp_i"][index], len(args.ids))
                    )
                    driver.write_value_by_name(
                        "kp_d", only_at(index, snapshot["kp_d"][index], len(args.ids))
                    )
                    driver.write_value_by_name(
                        "kp_p", only_at(index, snapshot["kp_p"][index], len(args.ids))
                    )
                    print("Tested servo operating mode and gains restored; torque remains off.")
                except Exception as exc:
                    print(
                        f"WARNING: failed to restore tested-servo control parameters: {exc}",
                        file=sys.stderr,
                    )
            try:
                driver.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
