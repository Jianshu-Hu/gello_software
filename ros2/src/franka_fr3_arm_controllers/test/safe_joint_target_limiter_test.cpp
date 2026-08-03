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

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "franka_fr3_arm_controllers/safe_joint_target_limiter.hpp"

namespace franka_fr3_arm_controllers {
namespace {

using Vector7d = SafeJointTargetLimiter::Vector7d;

const Vector7d kValidStart{0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0};

TEST(SafeJointTargetLimiterTest, DefaultConfigurationProducesFiniteTarget) {
  SafeJointTargetLimiter limiter;
  limiter.reset(kValidStart);

  const Vector7d raw_target{1.0, -1.0, 1.0, -2.0, 1.0, 2.0, -1.0};
  const auto result = limiter.update(raw_target, kValidStart, 0.001);

  EXPECT_TRUE(result.valid);
  EXPECT_TRUE(result.target.allFinite());
  EXPECT_TRUE(result.target_velocity.allFinite());
}

TEST(SafeJointTargetLimiterTest, ObeysVelocityAndAccelerationDuringTrackingAndReversal) {
  const auto config = SafeJointTargetLimiter::defaultConfig();
  SafeJointTargetLimiter limiter(config);
  limiter.reset(kValidStart);

  constexpr double kDt = 0.001;
  Vector7d measured = kValidStart;
  Vector7d previous_target = kValidStart;
  Vector7d previous_velocity = Vector7d::Zero();

  for (int step = 0; step < 5000; ++step) {
    const Vector7d raw_target = step < 2500 ? Vector7d{1.0, -1.0, 1.0, -2.0, 1.0, 2.0, -1.0}
                                            : Vector7d{-1.0, 1.0, -1.0, -1.0, -1.0, 3.0, 1.0};
    const auto result = limiter.update(raw_target, measured, kDt);
    ASSERT_TRUE(result.valid);

    const Vector7d velocity = (result.target - previous_target) / kDt;
    const Vector7d acceleration = (velocity - previous_velocity) / kDt;
    for (Eigen::Index joint = 0; joint < velocity.size(); ++joint) {
      EXPECT_LE(std::abs(velocity(joint)), config.max_velocity(joint) + 1e-8);
      EXPECT_LE(std::abs(acceleration(joint)), config.max_acceleration(joint) + 1e-6);
      EXPECT_GE(result.target(joint), config.lower_position(joint) - 1e-12);
      EXPECT_LE(result.target(joint), config.upper_position(joint) + 1e-12);
    }

    measured = result.target;
    previous_target = result.target;
    previous_velocity = velocity;
  }
}

TEST(SafeJointTargetLimiterTest, ReportsRawPositionOutsideSafeRange) {
  const auto config = SafeJointTargetLimiter::defaultConfig();
  SafeJointTargetLimiter limiter(config);
  limiter.reset(kValidStart);
  Vector7d raw_target = kValidStart;
  raw_target(0) = config.upper_position(0) + 1.0;

  const auto result = limiter.update(raw_target, kValidStart, 0.001);

  ASSERT_TRUE(result.valid);
  EXPECT_TRUE(result.position_limited);
  EXPECT_LE(result.target(0), config.upper_position(0));
}

TEST(SafeJointTargetLimiterTest, CapsTargetErrorWhenRobotDoesNotMove) {
  const auto config = SafeJointTargetLimiter::defaultConfig();
  SafeJointTargetLimiter limiter(config);
  limiter.reset(kValidStart);
  const Vector7d raw_target{2.0, 1.0, -2.0, -2.5, 2.0, 4.0, 2.0};

  SafeJointTargetLimiter::Result result;
  for (int step = 0; step < 5000; ++step) {
    result = limiter.update(raw_target, kValidStart, 0.001);
    ASSERT_TRUE(result.valid);
    for (Eigen::Index joint = 0; joint < result.target.size(); ++joint) {
      EXPECT_LE(std::abs(result.target(joint) - kValidStart(joint)),
                config.max_tracking_error(joint) + 1e-12);
    }
  }
  EXPECT_TRUE(result.tracking_error_limited);
}

TEST(SafeJointTargetLimiterTest, RejectsNonFiniteInput) {
  SafeJointTargetLimiter limiter;
  limiter.reset(kValidStart);
  Vector7d raw_target = kValidStart;
  raw_target(2) = std::numeric_limits<double>::quiet_NaN();

  const auto result = limiter.update(raw_target, kValidStart, 0.001);

  EXPECT_FALSE(result.valid);
  EXPECT_TRUE(result.target.allFinite());
}

}  // namespace
}  // namespace franka_fr3_arm_controllers
