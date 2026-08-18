// Copyright (c) 2026 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace franka_fr3_arm_controllers {

/**
 * Generates a position reference from absolute joint waypoints.
 *
 * The generator is deliberately independent of ROS. A non-real-time callback
 * may publish a new target, while the controller's 1 kHz update loop calls
 * setTarget() and advance(). All state is fixed-size and the hot path does not
 * allocate memory.
 */
class JointReferenceGenerator {
 public:
  static constexpr std::size_t kJoints = 7;
  using Vector = std::array<double, kJoints>;

  struct State {
    Vector position{};
    Vector velocity{};
    Vector acceleration{};
  };

  // These are the shared operational limits used by collection and deployment.
  static constexpr Vector kMaxVelocity = {0.60, 0.60, 0.60, 0.60, 0.75, 0.75, 0.75};
  static constexpr Vector kMaxAcceleration = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  static constexpr Vector kPositionLower = {-2.6937, -1.7337, -2.8507, -2.9921,
                                            -2.7565, 0.5945,  -2.9659};
  static constexpr Vector kPositionUpper = {2.6937, 1.7337, 2.8507, -0.2018,
                                            2.7565, 4.4669, 2.9659};

  explicit JointReferenceGenerator(double minimum_duration_s = 0.1)
      : minimum_duration_s_(minimum_duration_s) {}

  void reset(const Vector& position);

  [[nodiscard]] bool initialized() const { return initialized_; }
  [[nodiscard]] const State& state() const { return state_; }
  [[nodiscard]] const Vector& target() const { return target_; }

  // Returns false for a non-finite target. Position targets are clipped to the
  // shared operational envelope before they are accepted.
  bool setTarget(const Vector& target);

  // Generate one reference sample. dt is normally the controller period.
  void advance(double dt);

  // Replan to the current reference position with zero terminal velocity and
  // acceleration. This gives the watchdog a bounded stop rather than a jump
  // to the measured robot position.
  void requestStop();

  static bool sanitizeTarget(const Vector& input, Vector* output);

 private:
  using Coefficients = std::array<std::array<double, 6>, kJoints>;

  static double evaluate(const std::array<double, 6>& c, double t);
  static double evaluateVelocity(const std::array<double, 6>& c, double t);
  static double evaluateAcceleration(const std::array<double, 6>& c, double t);
  static Coefficients makeCoefficients(const State& start, const Vector& target, double duration);
  bool limitsSatisfied(const Coefficients& coefficients, double duration) const;
  void replan(const Vector& target);

  State state_{};
  Vector target_{};
  Coefficients coefficients_{};
  double elapsed_s_{0.0};
  double duration_s_{0.0};
  double minimum_duration_s_{0.1};
  bool initialized_{false};
  bool target_pending_{false};
};

}  // namespace franka_fr3_arm_controllers
