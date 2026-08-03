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

#include "franka_fr3_arm_controllers/safe_joint_target_limiter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace franka_fr3_arm_controllers {
namespace {

constexpr double kComparisonTolerance = 1e-12;

double clamp(double value, double lower, double upper) {
  return std::max(lower, std::min(value, upper));
}

double direction(double value) {
  if (value > 0.0) {
    return 1.0;
  }
  if (value < 0.0) {
    return -1.0;
  }
  return 0.0;
}

}  // namespace

bool SafeJointTargetLimiter::Result::anyLimited() const {
  return position_limited || velocity_limited || acceleration_limited || tracking_error_limited;
}

SafeJointTargetLimiter::SafeJointTargetLimiter() : config_(defaultConfig()) {}

SafeJointTargetLimiter::SafeJointTargetLimiter(const Config& config) : config_(config) {
  validateConfig_(config_);
}

SafeJointTargetLimiter::Config SafeJointTargetLimiter::defaultConfig(double position_margin_rad) {
  if (!std::isfinite(position_margin_rad) || position_margin_rad < 0.0) {
    throw std::invalid_argument("position margin must be finite and non-negative");
  }

  // FR3 joint limits from the Franka documentation. The position margin keeps
  // q_goal away from the hard boundary; velocity defaults are 80% of the
  // documented maximum joint velocities.
  const Vector7d hard_lower{-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508};
  const Vector7d hard_upper{2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508};

  Config config;
  config.lower_position = hard_lower.array() + position_margin_rad;
  config.upper_position = hard_upper.array() - position_margin_rad;
  config.max_velocity << 2.096, 2.096, 2.096, 2.096, 4.208, 3.344, 4.208;
  config.max_acceleration.setConstant(8.0);
  config.max_tracking_error << 0.12, 0.12, 0.12, 0.12, 0.20, 0.25, 0.30;
  validateConfig_(config);
  return config;
}

void SafeJointTargetLimiter::configure(const Config& config) {
  validateConfig_(config);
  config_ = config;
  clear();
}

void SafeJointTargetLimiter::reset(const Vector7d& position) {
  if (!position.allFinite()) {
    throw std::invalid_argument("safe target reset position must be finite");
  }
  target_position_ = clampToPositionRange(position);
  target_velocity_.setZero();
  initialized_ = true;
}

void SafeJointTargetLimiter::clear() {
  target_position_.setZero();
  target_velocity_.setZero();
  initialized_ = false;
}

bool SafeJointTargetLimiter::initialized() const {
  return initialized_;
}

SafeJointTargetLimiter::Vector7d SafeJointTargetLimiter::clampToPositionRange(
    const Vector7d& target,
    bool* limited) const {
  Vector7d clamped = target;
  bool was_limited = false;
  for (Eigen::Index i = 0; i < target.size(); ++i) {
    clamped(i) = clamp(target(i), config_.lower_position(i), config_.upper_position(i));
    was_limited = was_limited || std::abs(clamped(i) - target(i)) > kComparisonTolerance;
  }
  if (limited != nullptr) {
    *limited = was_limited;
  }
  return clamped;
}

SafeJointTargetLimiter::Result SafeJointTargetLimiter::update(const Vector7d& raw_target,
                                                              const Vector7d& measured_position,
                                                              double dt_seconds) {
  Result result;
  result.target = initialized_ ? target_position_ : measured_position;
  result.target_velocity = initialized_ ? target_velocity_ : Vector7d::Zero();

  if (!raw_target.allFinite() || !measured_position.allFinite() || !std::isfinite(dt_seconds) ||
      dt_seconds <= 0.0) {
    return result;
  }

  if (!initialized_) {
    reset(measured_position);
  }

  bool position_limited = false;
  Vector7d destination = clampToPositionRange(raw_target, &position_limited);
  result.position_limited = position_limited;

  for (Eigen::Index i = 0; i < destination.size(); ++i) {
    const double tracking_lower = measured_position(i) - config_.max_tracking_error(i);
    const double tracking_upper = measured_position(i) + config_.max_tracking_error(i);
    const double tracking_limited_destination =
        clamp(destination(i), tracking_lower, tracking_upper);
    if (std::abs(tracking_limited_destination - destination(i)) > kComparisonTolerance) {
      result.tracking_error_limited = true;
    }
    destination(i) = tracking_limited_destination;

    const double error = destination(i) - target_position_(i);
    const double acceleration = config_.max_acceleration(i);
    const double velocity = config_.max_velocity(i);

    // This discrete stopping-speed bound reserves enough distance to decelerate
    // on subsequent control cycles without snapping the target at the endpoint.
    const double stopping_velocity = std::max(
        0.0,
        std::sqrt(std::pow(acceleration * dt_seconds, 2.0) + 2.0 * acceleration * std::abs(error)) -
            acceleration * dt_seconds);
    const double desired_velocity = direction(error) * std::min(velocity, stopping_velocity);
    if (std::abs(stopping_velocity) > velocity + kComparisonTolerance) {
      result.velocity_limited = true;
    }

    const double max_velocity_change = acceleration * dt_seconds;
    const double velocity_change = desired_velocity - target_velocity_(i);
    const double limited_velocity_change =
        clamp(velocity_change, -max_velocity_change, max_velocity_change);
    if (std::abs(limited_velocity_change - velocity_change) > kComparisonTolerance) {
      result.acceleration_limited = true;
    }

    double next_velocity = target_velocity_(i) + limited_velocity_change;
    if (std::abs(next_velocity) > velocity) {
      next_velocity = clamp(next_velocity, -velocity, velocity);
      result.velocity_limited = true;
    }

    double next_position = target_position_(i) + next_velocity * dt_seconds;
    const double safe_lower =
        std::max(config_.lower_position(i), measured_position(i) - config_.max_tracking_error(i));
    const double safe_upper =
        std::min(config_.upper_position(i), measured_position(i) + config_.max_tracking_error(i));
    const double bounded_position = clamp(next_position, safe_lower, safe_upper);
    if (std::abs(bounded_position - next_position) > kComparisonTolerance) {
      result.tracking_error_limited = result.tracking_error_limited ||
                                      bounded_position == tracking_lower ||
                                      bounded_position == tracking_upper;
      result.position_limited = result.position_limited ||
                                bounded_position == config_.lower_position(i) ||
                                bounded_position == config_.upper_position(i);
      next_position = bounded_position;
      next_velocity = (next_position - target_position_(i)) / dt_seconds;
    }

    target_position_(i) = next_position;
    target_velocity_(i) = next_velocity;
  }

  result.target = target_position_;
  result.target_velocity = target_velocity_;
  result.max_raw_to_safe_error_rad = (raw_target - target_position_).cwiseAbs().maxCoeff();
  result.valid = true;
  return result;
}

void SafeJointTargetLimiter::validateConfig_(const Config& config) {
  if (!config.lower_position.allFinite() || !config.upper_position.allFinite() ||
      !config.max_velocity.allFinite() || !config.max_acceleration.allFinite() ||
      !config.max_tracking_error.allFinite()) {
    throw std::invalid_argument("safe target limiter parameters must be finite");
  }

  for (Eigen::Index i = 0; i < config.lower_position.size(); ++i) {
    if (config.lower_position(i) >= config.upper_position(i)) {
      throw std::invalid_argument("safe target lower position must be below upper position");
    }
    if (config.max_velocity(i) <= 0.0 || config.max_acceleration(i) <= 0.0 ||
        config.max_tracking_error(i) <= 0.0) {
      throw std::invalid_argument("safe target dynamic limits must be positive");
    }
  }
}

}  // namespace franka_fr3_arm_controllers
