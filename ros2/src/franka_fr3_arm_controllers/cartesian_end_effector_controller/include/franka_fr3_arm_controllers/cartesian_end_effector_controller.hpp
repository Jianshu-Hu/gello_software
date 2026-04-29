// Copyright (c) 2026
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
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <franka_semantic_components/franka_cartesian_pose_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_fr3_arm_controllers {

class CartesianEndEffectorController : public controller_interface::ControllerInterface {
 public:
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  static constexpr size_t kPoseValues = 7;

  void targetPoseCallback_(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  bool readLatestVrPose_(Eigen::Vector3d& position,
                         Eigen::Quaterniond& orientation,
                         double& stamp_sec,
                         uint64_t& sequence) const;
  void resetReference_(const Eigen::Vector3d& vr_position,
                       const Eigen::Quaterniond& vr_orientation,
                       const Eigen::Vector3d& current_position,
                       const Eigen::Quaterniond& current_orientation);
  void computeTargetPose_(const Eigen::Vector3d& vr_position,
                          const Eigen::Quaterniond& vr_orientation,
                          Eigen::Vector3d& target_position,
                          Eigen::Quaterniond& target_orientation);
  void filterTargetPose_(const Eigen::Vector3d& target_position,
                         const Eigen::Quaterniond& target_orientation,
                         double dt);
  bool loadVectorParameter_(const std::string& parameter_name,
                            size_t expected_size,
                            std::vector<double>& values) const;

  std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface> franka_cartesian_pose_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_subscriber_;

  std::array<std::atomic<double>, kPoseValues> latest_vr_pose_{};
  std::atomic<bool> latest_vr_pose_valid_{false};
  std::atomic<double> latest_vr_pose_time_sec_{0.0};
  std::atomic<uint64_t> latest_vr_pose_sequence_{0};

  Eigen::Vector3d reference_vr_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond reference_vr_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d reference_ee_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond reference_ee_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d commanded_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond commanded_orientation_{Eigen::Quaterniond::Identity()};

  bool reference_initialized_{false};
  bool command_initialized_{false};
  uint64_t consumed_vr_pose_sequence_{0};

  std::string target_pose_topic_;
  double vr_position_scale_{1.0};
  double command_timeout_sec_{0.25};
  double filter_time_constant_sec_{0.08};
  double max_linear_velocity_{0.12};
  double max_angular_velocity_{0.70};
  double workspace_radius_{0.35};
  double min_z_{0.05};
  Eigen::Matrix3d vr_to_robot_rotation_{Eigen::Matrix3d::Identity()};
};

}  // namespace franka_fr3_arm_controllers
