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

#include "franka_fr3_arm_controllers/joint_reference_generator.hpp"

#include <algorithm>
#include <limits>

namespace franka_fr3_arm_controllers {

void JointReferenceGenerator::reset(const Vector& position) {
  for (const double value : position) {
    if (!std::isfinite(value)) {
      return;
    }
  }
  // The measured robot state is authoritative at initialization. Do not clip
  // it to the operational target envelope and create an activation jump.
  state_.position = position;
  state_.velocity.fill(0.0);
  state_.acceleration.fill(0.0);
  target_ = position;
  coefficients_ = Coefficients{};
  elapsed_s_ = 0.0;
  duration_s_ = 0.0;
  initialized_ = true;
  target_pending_ = false;
}

bool JointReferenceGenerator::sanitizeTarget(const Vector& input, Vector* output) {
  if (output == nullptr) {
    return false;
  }
  for (std::size_t i = 0; i < kJoints; ++i) {
    if (!std::isfinite(input[i])) {
      return false;
    }
    (*output)[i] = std::clamp(input[i], kPositionLower[i], kPositionUpper[i]);
  }
  return true;
}

bool JointReferenceGenerator::setTarget(const Vector& target) {
  Vector safe_target{};
  if (!sanitizeTarget(target, &safe_target)) {
    return false;
  }
  if (!initialized_) {
    reset(safe_target);
    return true;
  }
  if (safe_target == target_) {
    return true;
  }
  target_ = safe_target;
  target_pending_ = true;
  return true;
}

double JointReferenceGenerator::evaluate(const std::array<double, 6>& c, double t) {
  return ((((c[5] * t + c[4]) * t + c[3]) * t + c[2]) * t + c[1]) * t + c[0];
}

double JointReferenceGenerator::evaluateVelocity(const std::array<double, 6>& c, double t) {
  return ((((5.0 * c[5] * t) + 4.0 * c[4]) * t + 3.0 * c[3]) * t + 2.0 * c[2]) * t + c[1];
}

double JointReferenceGenerator::evaluateAcceleration(const std::array<double, 6>& c, double t) {
  return (((20.0 * c[5] * t) + 12.0 * c[4]) * t + 6.0 * c[3]) * t + 2.0 * c[2];
}

JointReferenceGenerator::Coefficients JointReferenceGenerator::makeCoefficients(
    const State& start,
    const Vector& target,
    double duration) {
  Coefficients coefficients{};
  const double t2 = duration * duration;
  const double t3 = t2 * duration;
  const double t4 = t3 * duration;
  const double t5 = t4 * duration;
  for (std::size_t i = 0; i < kJoints; ++i) {
    const double delta = target[i] - start.position[i];
    coefficients[i][0] = start.position[i];
    coefficients[i][1] = start.velocity[i];
    coefficients[i][2] = 0.5 * start.acceleration[i];
    // Terminal velocity and acceleration are zero. These are the closed-form
    // quintic boundary-condition coefficients; no matrix allocation is needed.
    coefficients[i][3] =
        (20.0 * delta - 12.0 * start.velocity[i] * duration - 3.0 * start.acceleration[i] * t2) /
        (2.0 * t3);
    coefficients[i][4] =
        (-30.0 * delta + 16.0 * start.velocity[i] * duration + 3.0 * start.acceleration[i] * t2) /
        (2.0 * t4);
    coefficients[i][5] =
        (12.0 * delta - 6.0 * start.velocity[i] * duration - start.acceleration[i] * t2) /
        (2.0 * t5);
  }
  return coefficients;
}

bool JointReferenceGenerator::limitsSatisfied(const Coefficients& coefficients,
                                              double duration) const {
  // A dense fixed grid is conservative for these low-order polynomials and
  // keeps planning deterministic and allocation-free. The 2% margin covers
  // the small gap between sampled extrema and exact polynomial extrema.
  constexpr std::size_t kChecks = 256;
  for (std::size_t i = 0; i < kJoints; ++i) {
    const double start_position = evaluate(coefficients[i], 0.0);
    const bool starts_below = start_position < kPositionLower[i];
    const bool starts_above = start_position > kPositionUpper[i];
    bool entered_envelope = !(starts_below || starts_above);
    double previous_position = start_position;
    for (std::size_t sample = 0; sample <= kChecks; ++sample) {
      const double t = duration * static_cast<double>(sample) / static_cast<double>(kChecks);
      const double position = evaluate(coefficients[i], t);
      const double velocity = std::abs(evaluateVelocity(coefficients[i], t));
      const double acceleration = std::abs(evaluateAcceleration(coefficients[i], t));
      const bool inside_position_envelope =
          position >= kPositionLower[i] && position <= kPositionUpper[i];
      if (inside_position_envelope) {
        entered_envelope = true;
      }
      // A measured activation state can be outside the operational envelope.
      // Allow only the initial monotonic recovery back into the envelope;
      // after entry, every sample must remain bounded.
      const bool initial_recovery =
          !entered_envelope && ((starts_below && position >= previous_position) ||
                                (starts_above && position <= previous_position));
      const bool outside_position_envelope = !inside_position_envelope && !initial_recovery;
      if (outside_position_envelope || velocity > 0.98 * kMaxVelocity[i] ||
          acceleration > 0.98 * kMaxAcceleration[i]) {
        return false;
      }
      previous_position = position;
    }
  }
  return true;
}

void JointReferenceGenerator::replan(const Vector& target) {
  const State start = state_;
  Vector distance{};
  double duration = minimum_duration_s_;
  for (std::size_t i = 0; i < kJoints; ++i) {
    distance[i] = std::abs(target[i] - start.position[i]);
    duration = std::max(duration, 1.875 * distance[i] / kMaxVelocity[i]);
    duration = std::max(duration, std::sqrt(5.773503 * distance[i] / kMaxAcceleration[i]));
  }

  for (std::size_t attempt = 0; attempt < 40; ++attempt) {
    const Coefficients candidate = makeCoefficients(start, target, duration);
    if (limitsSatisfied(candidate, duration)) {
      coefficients_ = candidate;
      elapsed_s_ = 0.0;
      duration_s_ = duration;
      target_pending_ = false;
      return;
    }
    duration *= 1.25;
  }

  // This should be unreachable for finite, bounded input. Keep the previous
  // reference if a pathological state ever makes planning impossible.
  target_pending_ = false;
}

void JointReferenceGenerator::advance(double dt) {
  if (!initialized_) {
    return;
  }
  dt = std::clamp(dt, 0.0, 0.02);
  if (target_pending_) {
    replan(target_);
  }
  if (duration_s_ <= 0.0 || dt <= 0.0) {
    return;
  }

  elapsed_s_ = std::min(elapsed_s_ + dt, duration_s_);
  for (std::size_t i = 0; i < kJoints; ++i) {
    state_.position[i] = evaluate(coefficients_[i], elapsed_s_);
    state_.velocity[i] = evaluateVelocity(coefficients_[i], elapsed_s_);
    state_.acceleration[i] = evaluateAcceleration(coefficients_[i], elapsed_s_);
  }
  if (elapsed_s_ >= duration_s_) {
    state_.position = target_;
    state_.velocity.fill(0.0);
    state_.acceleration.fill(0.0);
    duration_s_ = 0.0;
    elapsed_s_ = 0.0;
  }
}

void JointReferenceGenerator::requestStop() {
  if (initialized_) {
    target_ = state_.position;
    target_pending_ = true;
  }
}

}  // namespace franka_fr3_arm_controllers
