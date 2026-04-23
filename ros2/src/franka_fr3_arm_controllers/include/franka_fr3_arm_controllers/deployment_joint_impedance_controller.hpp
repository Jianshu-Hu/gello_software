// Copyright (c) 2025 Franka Robotics GmbH
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
#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <string>
#include "franka_fr3_arm_controllers/motion_generator.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_fr3_arm_controllers {

class DeploymentJointImpedanceController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
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
  std::string arm_id_;
  std::string namespace_prefix_;
  const int num_joints = 7;
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  Vector7d hold_position_;
  double k_alpha_;
  double command_timeout_sec_{0.5};
  bool move_to_start_position_finished_{false};
  bool motion_generator_initialized_{false};
  bool hold_position_initialized_{false};
  bool deployment_enabled_{false};
  bool command_values_valid_{false};
  rclcpp::Time start_time_;
  rclcpp::Time command_accept_time_;
  rclcpp::Time last_command_time_;
  std::unique_ptr<MotionGenerator> motion_generator_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_ = nullptr;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr commanded_joint_state_publisher_ =
      nullptr;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr deployment_enable_service_ = nullptr;
  std::array<double, 7> command_values_{0, 0, 0, 0, 0, 0, 0};
  std::array<std::string, 7> joint_names_;

  Vector7d calculateTauDGains_(const Vector7d& q_goal);
  bool validateGains_(const std::vector<double>& gains, const std::string& gains_name);
  bool initializeMotionGenerator_();
  void publishCommandedJointState_(const Vector7d& q_goal);
  void updateJointStates_();
  bool commandValuesAreFresh_() const;
  void holdCurrentPosition_();
  void publishHoldPosition_();
  void resetCommandTracking_(const rclcpp::Time& reference_time);
  void setDeploymentEnabled_(bool enabled);
  void handleSetDeploymentEnabled_(
      const std::shared_ptr<std_srvs::srv::SetBool::Request>& request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response);
  void jointStateCallback_(const sensor_msgs::msg::JointState msg);
};

}  // namespace franka_fr3_arm_controllers
