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

#include <Eigen/Eigen>

namespace franka_fr3_arm_controllers {

class SafeJointTargetLimiter {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;

  struct Config {
    Vector7d lower_position;
    Vector7d upper_position;
    Vector7d max_velocity;
    Vector7d max_acceleration;
    Vector7d max_tracking_error;
  };

  struct Result {
    Vector7d target{Vector7d::Zero()};
    Vector7d target_velocity{Vector7d::Zero()};
    bool valid{false};
    bool position_limited{false};
    bool velocity_limited{false};
    bool acceleration_limited{false};
    bool tracking_error_limited{false};
    double max_raw_to_safe_error_rad{0.0};

    [[nodiscard]] bool anyLimited() const;
  };

  SafeJointTargetLimiter();
  explicit SafeJointTargetLimiter(const Config& config);

  [[nodiscard]] static Config defaultConfig(double position_margin_rad = 0.02);
  void configure(const Config& config);
  void reset(const Vector7d& position);
  void clear();
  [[nodiscard]] bool initialized() const;
  [[nodiscard]] Vector7d clampToPositionRange(const Vector7d& target,
                                              bool* limited = nullptr) const;
  [[nodiscard]] Result update(const Vector7d& raw_target,
                              const Vector7d& measured_position,
                              double dt_seconds);

 private:
  static void validateConfig_(const Config& config);

  Config config_;
  Vector7d target_position_{Vector7d::Zero()};
  Vector7d target_velocity_{Vector7d::Zero()};
  bool initialized_{false};
};

}  // namespace franka_fr3_arm_controllers
