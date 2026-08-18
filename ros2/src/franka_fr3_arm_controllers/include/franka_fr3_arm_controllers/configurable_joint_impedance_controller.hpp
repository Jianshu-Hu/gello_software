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
#include <array>
#include <atomic>
#include <controller_interface/controller_interface.hpp>
#include <cstdint>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <string>

#include "franka_fr3_arm_controllers/joint_reference_generator.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_fr3_arm_controllers {

/**
 * Shared impedance controller for teleoperation and policy deployment.
 *
 * The two modes differ only in their command topic and whether a service gate
 * is required. The torque law and robot-side reference generation are shared.
 */
class ConfigurableJointImpedanceController : public controller_interface::ControllerInterface {
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

 protected:
  virtual std::string defaultCommandTopic() const = 0;
  virtual std::string defaultAcceptedTargetTopic() const = 0;
  virtual std::string defaultActivationMode() const = 0;

 private:
  struct CommandState {
    std::array<double, 7> target{};
    double received_time_s{0.0};
    std::uint64_t sequence{0};
    bool valid{false};
  };

  std::string arm_id_;
  std::string namespace_prefix_;
  const int num_joints = 7;
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  Vector7d hold_position_;
  double k_alpha_{0.99};
  double command_timeout_sec_{0.5};
  std::string command_topic_;
  std::string accepted_target_topic_;
  std::string activation_mode_;
  bool command_mode_active_{false};
  std::atomic_bool requested_command_mode_active_{false};
  bool hold_position_initialized_{false};
  bool reference_generator_initialized_{false};
  bool reference_stop_requested_{false};
  std::atomic<double> command_accept_time_s_{0.0};
  JointReferenceGenerator reference_generator_;
  realtime_tools::RealtimeBuffer<CommandState> command_buffer_;
  std::uint64_t command_sequence_{0};
  std::uint64_t last_reference_command_sequence_{0};
  std::uint32_t diagnostic_publish_counter_{0};
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_ = nullptr;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr commanded_joint_state_publisher_ =
      nullptr;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr accepted_target_publisher_ = nullptr;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr activation_service_ = nullptr;
  std::array<std::string, 7> joint_names_;

  bool validateGains_(const std::vector<double>& gains, const std::string& gains_name);
  void updateJointStates_();
  void jointStateCallback_(const sensor_msgs::msg::JointState& msg);
  void publishCommandedJointState_(const Vector7d& q_goal);
  void publishAcceptedTarget_(const std::array<double, 7>& target);
  void publishHoldPosition_();
  void setCommandModeActive_(bool enabled);
  void handleSetCommandModeActive_(const std::shared_ptr<std_srvs::srv::SetBool::Request>& request,
                                   std::shared_ptr<std_srvs::srv::SetBool::Response> response);
  void resetCommandTracking_(double reference_time_s);
  bool commandValuesAreFresh_(const CommandState& command, double now_s) const;
  Vector7d calculateTauDGains_(const Vector7d& q_goal);
};

}  // namespace franka_fr3_arm_controllers
