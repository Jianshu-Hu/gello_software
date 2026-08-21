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

#include <chrono>
#include <cmath>

#include <gtest/gtest.h>

#include "franka_fr3_arm_controllers/joint_reference_generator.hpp"

namespace franka_fr3_arm_controllers {

TEST(JointReferenceGeneratorTest, KeepsReferenceWithinConfiguredLimits) {
  JointReferenceGenerator generator;
  JointReferenceGenerator::Vector initial = {0.0, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0};
  generator.reset(initial);

  JointReferenceGenerator::Vector target{};
  target[0] = 0.1;
  ASSERT_TRUE(generator.setTarget(target));

  for (int cycle = 0; cycle < 2000; ++cycle) {
    generator.advance(0.001);
    const auto& state = generator.state();
    for (std::size_t joint = 0; joint < JointReferenceGenerator::kJoints; ++joint) {
      EXPECT_GE(state.position[joint], JointReferenceGenerator::kPositionLower[joint]);
      EXPECT_LE(state.position[joint], JointReferenceGenerator::kPositionUpper[joint]);
      EXPECT_LE(std::abs(state.velocity[joint]),
                JointReferenceGenerator::kMaxVelocity[joint] * 1.001);
      EXPECT_LE(std::abs(state.acceleration[joint]),
                JointReferenceGenerator::kMaxAcceleration[joint] * 1.001);
    }
  }
  EXPECT_NEAR(generator.state().position[0], 0.1, 1e-3);
}

TEST(JointReferenceGeneratorTest, ClipsUnreachableTargetToNearestSafePoint) {
  JointReferenceGenerator generator;
  JointReferenceGenerator::Vector initial = {0.0, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0};
  generator.reset(initial);

  JointReferenceGenerator::Vector requested = initial;
  requested[0] = JointReferenceGenerator::kPositionUpper[0] + 1.0;
  requested[1] = JointReferenceGenerator::kPositionLower[1] - 1.0;
  ASSERT_TRUE(generator.setTarget(requested));

  EXPECT_DOUBLE_EQ(generator.target()[0], JointReferenceGenerator::kPositionUpper[0]);
  EXPECT_DOUBLE_EQ(generator.target()[1], JointReferenceGenerator::kPositionLower[1]);
  for (std::size_t joint = 2; joint < JointReferenceGenerator::kJoints; ++joint) {
    EXPECT_DOUBLE_EQ(generator.target()[joint], requested[joint]);
  }
}

TEST(JointReferenceGeneratorTest, RecoversFromMeasuredStateOutsideEnvelope) {
  JointReferenceGenerator generator;
  JointReferenceGenerator::Vector initial = {0.0, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0};
  initial[0] = JointReferenceGenerator::kPositionUpper[0] + 0.1;
  generator.reset(initial);

  JointReferenceGenerator::Vector target = initial;
  target[0] = JointReferenceGenerator::kPositionUpper[0];
  ASSERT_TRUE(generator.setTarget(target));
  for (int cycle = 0; cycle < 2000; ++cycle) {
    generator.advance(0.001);
  }
  EXPECT_NEAR(generator.state().position[0], JointReferenceGenerator::kPositionUpper[0], 1e-3);
}

TEST(JointReferenceGeneratorTest, ReplanningPreservesReferenceState) {
  JointReferenceGenerator generator;
  JointReferenceGenerator::Vector initial = {0.0, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0};
  generator.reset(initial);
  JointReferenceGenerator::Vector first{};
  first[0] = 0.2;
  ASSERT_TRUE(generator.setTarget(first));
  generator.advance(0.12);
  const auto before = generator.state();

  JointReferenceGenerator::Vector second{};
  second[0] = -0.1;
  ASSERT_TRUE(generator.setTarget(second));
  generator.advance(0.0);
  const auto after = generator.state();
  EXPECT_DOUBLE_EQ(after.position[0], before.position[0]);
  EXPECT_DOUBLE_EQ(after.velocity[0], before.velocity[0]);
  EXPECT_DOUBLE_EQ(after.acceleration[0], before.acceleration[0]);
}

TEST(JointReferenceGeneratorTest, OneKHzAdvanceFitsRealtimeBudget) {
  JointReferenceGenerator generator;
  JointReferenceGenerator::Vector initial = {0.0, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0};
  generator.reset(initial);
  JointReferenceGenerator::Vector target{};
  target[0] = 0.3;
  ASSERT_TRUE(generator.setTarget(target));

  constexpr int kCycles = 100000;
  const auto start = std::chrono::steady_clock::now();
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    if (cycle % 67 == 0) {
      target[0] = (target[0] > 0.0) ? -0.3 : 0.3;
      ASSERT_TRUE(generator.setTarget(target));
    }
    generator.advance(0.001);
  }
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
  // This is a host-side regression guard, not a hard real-time guarantee. The
  // generator must leave ample margin below one second for 100,000 1 kHz ticks.
  EXPECT_LT(elapsed.count(), 1.0);
}

}  // namespace franka_fr3_arm_controllers
