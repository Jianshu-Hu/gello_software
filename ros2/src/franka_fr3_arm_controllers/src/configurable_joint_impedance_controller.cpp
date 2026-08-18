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

#include "franka_fr3_arm_controllers/configurable_joint_impedance_controller.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>

namespace franka_fr3_arm_controllers {

controller_interface::InterfaceConfiguration
ConfigurableJointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
ConfigurableJointImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) +
                           "/position");
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) +
                           "/velocity");
  }
  return config;
}

controller_interface::return_type ConfigurableJointImpedanceController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  updateJointStates_();
  if (activation_mode_ == "service") {
    const bool requested_active = requested_command_mode_active_.load(std::memory_order_acquire);
    if (requested_active != command_mode_active_) {
      command_mode_active_ = requested_active;
      hold_position_initialized_ = false;
      reference_generator_initialized_ = false;
      reference_stop_requested_ = false;
      command_accept_time_s_.store(get_node()->now().seconds(), std::memory_order_release);
    }
  }
  if (!command_mode_active_) {
    publishHoldPosition_();
    return controller_interface::return_type::OK;
  }

  if (!reference_generator_initialized_) {
    JointReferenceGenerator::Vector initial{};
    for (int i = 0; i < num_joints; ++i) {
      initial[i] = q_(i);
    }
    reference_generator_.reset(initial);
    reference_generator_initialized_ = true;
  }

  const auto* command = command_buffer_.readFromRT();
  const double now_s = get_node()->now().seconds();
  if (command->valid && command->sequence != last_reference_command_sequence_ &&
      commandValuesAreFresh_(*command, now_s)) {
    reference_generator_.setTarget(command->target);
    last_reference_command_sequence_ = command->sequence;
    reference_stop_requested_ = false;
  }
  if (!command->valid || !commandValuesAreFresh_(*command, now_s)) {
    if (!reference_stop_requested_) {
      reference_generator_.requestStop();
      reference_stop_requested_ = true;
    }
  }
  reference_generator_.advance(period.seconds());

  Vector7d q_goal;
  for (int i = 0; i < num_joints; ++i) {
    q_goal(i) = reference_generator_.state().position[i];
  }
  const Vector7d tau_d_calculated = calculateTauDGains_(q_goal);
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }
  publishCommandedJointState_(q_goal);
  return controller_interface::return_type::OK;
}

CallbackReturn ConfigurableJointImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<double>("k_alpha", 0.99);
    auto_declare<double>("command_timeout_sec", 0.5);
    auto_declare<std::string>("command_topic", defaultCommandTopic());
    auto_declare<std::string>("accepted_target_topic", defaultAcceptedTargetTopic());
    auto_declare<std::string>("activation_mode", defaultActivationMode());
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn ConfigurableJointImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  namespace_prefix_ = get_node()->get_namespace();
  if (namespace_prefix_ == "/" || namespace_prefix_.empty()) {
    namespace_prefix_.clear();
  } else {
    namespace_prefix_ = namespace_prefix_.substr(1) + "_";
  }

  const auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  const auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  k_alpha_ = get_node()->get_parameter("k_alpha").as_double();
  command_timeout_sec_ = get_node()->get_parameter("command_timeout_sec").as_double();
  command_topic_ = get_node()->get_parameter("command_topic").as_string();
  accepted_target_topic_ = get_node()->get_parameter("accepted_target_topic").as_string();
  activation_mode_ = get_node()->get_parameter("activation_mode").as_string();

  if (!validateGains_(k_gains, "k_gains") || !validateGains_(d_gains, "d_gains")) {
    return CallbackReturn::FAILURE;
  }
  if (k_alpha_ < 0.0 || k_alpha_ > 1.0 || command_timeout_sec_ <= 0.0 || command_topic_.empty() ||
      accepted_target_topic_.empty() ||
      (activation_mode_ != "always" && activation_mode_ != "service")) {
    RCLCPP_FATAL(get_node()->get_logger(), "Invalid impedance controller configuration");
    return CallbackReturn::FAILURE;
  }

  for (int i = 0; i < num_joints; ++i) {
    d_gains_(i) = d_gains.at(i);
    k_gains_(i) = k_gains.at(i);
    joint_names_[i] = arm_id_ + "_joint" + std::to_string(i + 1);
  }
  dq_filtered_.setZero();

  joint_state_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      command_topic_, rclcpp::QoS(1).best_effort(),
      [this](const sensor_msgs::msg::JointState& msg) { jointStateCallback_(msg); });
  commanded_joint_state_publisher_ = get_node()->create_publisher<sensor_msgs::msg::JointState>(
      "franka/commanded_joint_states", 10);
  accepted_target_publisher_ =
      get_node()->create_publisher<sensor_msgs::msg::JointState>(accepted_target_topic_, 10);
  if (activation_mode_ == "service") {
    activation_service_ = get_node()->create_service<std_srvs::srv::SetBool>(
        "~/set_deployment_enabled",
        [this](const std::shared_ptr<std_srvs::srv::SetBool::Request>& request,
               std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
          handleSetCommandModeActive_(request, response);
        });
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn ConfigurableJointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  const rclcpp::Time activation_time = get_node()->now();
  command_mode_active_ = activation_mode_ == "always";
  requested_command_mode_active_.store(command_mode_active_, std::memory_order_release);
  hold_position_initialized_ = false;
  reference_generator_initialized_ = false;
  reference_stop_requested_ = false;
  dq_filtered_.setZero();
  command_sequence_ = 0;
  last_reference_command_sequence_ = 0;
  diagnostic_publish_counter_ = 0;
  resetCommandTracking_(activation_time.seconds());
  return CallbackReturn::SUCCESS;
}

CallbackReturn ConfigurableJointImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  command_mode_active_ = false;
  requested_command_mode_active_.store(false, std::memory_order_release);
  hold_position_initialized_ = false;
  reference_generator_initialized_ = false;
  reference_stop_requested_ = false;
  return CallbackReturn::SUCCESS;
}

void ConfigurableJointImpedanceController::jointStateCallback_(
    const sensor_msgs::msg::JointState& msg) {
  if (msg.position.size() < static_cast<std::size_t>(num_joints)) {
    RCLCPP_WARN(get_node()->get_logger(), "Received joint target has fewer than 7 positions");
    return;
  }
  const rclcpp::Time receive_time = get_node()->now();
  if (receive_time.seconds() < command_accept_time_s_.load(std::memory_order_acquire)) {
    return;
  }
  CommandState command;
  for (int i = 0; i < num_joints; ++i) {
    command.target[i] = msg.position[i];
  }
  if (!JointReferenceGenerator::sanitizeTarget(command.target, &command.target)) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                         "Rejecting non-finite joint target");
    return;
  }
  command.received_time_s = receive_time.seconds();
  command.sequence = ++command_sequence_;
  command.valid = true;
  command_buffer_.writeFromNonRT(command);
  publishAcceptedTarget_(command.target);
}

bool ConfigurableJointImpedanceController::validateGains_(const std::vector<double>& gains,
                                                          const std::string& gains_name) {
  if (gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "%s parameter not set", gains_name.c_str());
    return false;
  }
  if (gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "%s should have size %d", gains_name.c_str(),
                 num_joints);
    return false;
  }
  return true;
}

void ConfigurableJointImpedanceController::updateJointStates_() {
  for (int i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");
    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

bool ConfigurableJointImpedanceController::commandValuesAreFresh_(const CommandState& command,
                                                                  double now_s) const {
  return command.valid &&
         command.received_time_s >= command_accept_time_s_.load(std::memory_order_acquire) &&
         now_s - command.received_time_s <= command_timeout_sec_;
}

void ConfigurableJointImpedanceController::resetCommandTracking_(double reference_time_s) {
  command_accept_time_s_.store(reference_time_s, std::memory_order_release);
  command_buffer_.initRT(CommandState{});
}

void ConfigurableJointImpedanceController::setCommandModeActive_(bool enabled) {
  requested_command_mode_active_.store(enabled, std::memory_order_release);
}

void ConfigurableJointImpedanceController::handleSetCommandModeActive_(
    const std::shared_ptr<std_srvs::srv::SetBool::Request>& request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
  setCommandModeActive_(request->data);
  response->success = true;
  response->message = request->data ? "enabled" : "disabled";
}

void ConfigurableJointImpedanceController::publishHoldPosition_() {
  if (!hold_position_initialized_) {
    hold_position_ = q_;
    hold_position_initialized_ = true;
  }
  const Vector7d tau_d_calculated = calculateTauDGains_(hold_position_);
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }
  publishCommandedJointState_(hold_position_);
}

ConfigurableJointImpedanceController::Vector7d
ConfigurableJointImpedanceController::calculateTauDGains_(const Vector7d& q_goal) {
  dq_filtered_ = (1 - k_alpha_) * dq_filtered_ + k_alpha_ * dq_;
  return k_gains_.cwiseProduct(q_goal - q_) + d_gains_.cwiseProduct(-dq_filtered_);
}

void ConfigurableJointImpedanceController::publishCommandedJointState_(const Vector7d& q_goal) {
  if (!commanded_joint_state_publisher_) {
    return;
  }
  // This topic is diagnostic only. Keep DDS serialization out of most 1 kHz
  // cycles; q_goal itself is consumed directly by the torque calculation.
  if (++diagnostic_publish_counter_ % 10 != 0) {
    return;
  }
  sensor_msgs::msg::JointState msg;
  msg.header.stamp = get_node()->now();
  msg.header.frame_id = arm_id_ + "_link0";
  msg.name.assign(joint_names_.begin(), joint_names_.end());
  msg.position.resize(num_joints);
  for (int i = 0; i < num_joints; ++i) {
    msg.position[i] = q_goal(i);
  }
  commanded_joint_state_publisher_->publish(msg);
}

void ConfigurableJointImpedanceController::publishAcceptedTarget_(
    const std::array<double, 7>& target) {
  if (!accepted_target_publisher_) {
    return;
  }
  sensor_msgs::msg::JointState msg;
  msg.header.stamp = get_node()->now();
  msg.header.frame_id = arm_id_ + "_accepted_target";
  msg.name.assign(joint_names_.begin(), joint_names_.end());
  msg.position.assign(target.begin(), target.end());
  accepted_target_publisher_->publish(msg);
}

}  // namespace franka_fr3_arm_controllers
