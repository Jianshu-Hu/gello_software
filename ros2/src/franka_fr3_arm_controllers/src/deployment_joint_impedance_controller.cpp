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

#include <franka_fr3_arm_controllers/deployment_joint_impedance_controller.hpp>

#include <Eigen/Eigen>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <exception>
#include <string>

namespace franka_fr3_arm_controllers {

controller_interface::InterfaceConfiguration
DeploymentJointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
DeploymentJointImpedanceController::state_interface_configuration() const {
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

controller_interface::return_type DeploymentJointImpedanceController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/) {
  updateJointStates_();

  if (!deployment_enabled_) {
    publishHoldPosition_();
    return controller_interface::return_type::OK;
  }

  if (!motion_generator_initialized_) {
    motion_generator_initialized_ = initializeMotionGenerator_();
    if (!motion_generator_initialized_) {
      publishHoldPosition_();
      return controller_interface::return_type::OK;
    }
  }

  Vector7d q_goal;
  if (!move_to_start_position_finished_) {
    auto trajectory_time = this->get_node()->now() - start_time_;
    auto motion_generator_output = motion_generator_->getDesiredJointPositions(trajectory_time);
    move_to_start_position_finished_ = motion_generator_output.second;
    q_goal = motion_generator_output.first;
  } else if (!commandValuesAreFresh_()) {
    holdCurrentPosition_();
    publishHoldPosition_();
    return controller_interface::return_type::OK;
  } else {
    for (int i = 0; i < num_joints; ++i) {
      q_goal(i) = command_values_[i];
    }
  }

  Vector7d tau_d_calculated = calculateTauDGains_(q_goal);
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }
  publishCommandedJointState_(q_goal);

  return controller_interface::return_type::OK;
}

void DeploymentJointImpedanceController::jointStateCallback_(
    const sensor_msgs::msg::JointState msg) {
  if (msg.position.size() < command_values_.size()) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Received deployment joint target size is smaller than expected size.");
    return;
  }

  const auto msg_time = rclcpp::Time(msg.header.stamp);
  if (msg_time < command_accept_time_) {
    RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 2000,
        "Ignoring stale deployment joint target published before controller activation/enabling");
    return;
  }

  std::copy(msg.position.begin(), msg.position.begin() + command_values_.size(), command_values_.begin());
  last_command_time_ = msg_time;
  command_values_valid_ = true;
}

CallbackReturn DeploymentJointImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<double>("k_alpha", 0.99);
    auto_declare<double>("command_timeout_sec", 0.5);
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn DeploymentJointImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  namespace_prefix_ = get_node()->get_namespace();
  if (namespace_prefix_ == "/" || namespace_prefix_.empty()) {
    namespace_prefix_.clear();
  } else {
    namespace_prefix_ = namespace_prefix_.substr(1) + "_";
  }

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  auto k_alpha = get_node()->get_parameter("k_alpha").as_double();
  command_timeout_sec_ = get_node()->get_parameter("command_timeout_sec").as_double();

  if (!validateGains_(k_gains, "k_gains") || !validateGains_(d_gains, "d_gains")) {
    return CallbackReturn::FAILURE;
  }

  for (int i = 0; i < num_joints; ++i) {
    d_gains_(i) = d_gains.at(i);
    k_gains_(i) = k_gains.at(i);
    joint_names_[i] = arm_id_ + "_joint" + std::to_string(i + 1);
  }

  if (k_alpha < 0.0 || k_alpha > 1.0) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_alpha should be in the range [0, 1]");
    return CallbackReturn::FAILURE;
  }
  if (command_timeout_sec_ <= 0.0) {
    RCLCPP_FATAL(get_node()->get_logger(), "command_timeout_sec should be > 0");
    return CallbackReturn::FAILURE;
  }

  k_alpha_ = k_alpha;
  dq_filtered_.setZero();

  joint_state_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      "deployment/joint_states", 1,
      [this](const sensor_msgs::msg::JointState& msg) { jointStateCallback_(msg); });
  commanded_joint_state_publisher_ =
      get_node()->create_publisher<sensor_msgs::msg::JointState>("franka/commanded_joint_states",
                                                                 10);
  deployment_enable_service_ = get_node()->create_service<std_srvs::srv::SetBool>(
      "joint_impedance_controller/set_deployment_enabled",
      [this](const std::shared_ptr<std_srvs::srv::SetBool::Request>& request,
             std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
        handleSetDeploymentEnabled_(request, response);
      });

  return CallbackReturn::SUCCESS;
}

CallbackReturn DeploymentJointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  move_to_start_position_finished_ = false;
  motion_generator_initialized_ = false;
  hold_position_initialized_ = false;
  motion_generator_.reset();
  deployment_enabled_ = false;
  const rclcpp::Time activation_time = get_node()->now();
  resetCommandTracking_(activation_time);
  dq_filtered_.setZero();
  start_time_ = activation_time;

  return CallbackReturn::SUCCESS;
}

CallbackReturn DeploymentJointImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  deployment_enabled_ = false;
  hold_position_initialized_ = false;
  motion_generator_initialized_ = false;
  move_to_start_position_finished_ = false;
  motion_generator_.reset();

  return CallbackReturn::SUCCESS;
}

auto DeploymentJointImpedanceController::calculateTauDGains_(const Vector7d& q_goal) -> Vector7d {
  dq_filtered_ = (1 - k_alpha_) * dq_filtered_ + k_alpha_ * dq_;
  return k_gains_.cwiseProduct(q_goal - q_) + d_gains_.cwiseProduct(-dq_filtered_);
}

void DeploymentJointImpedanceController::publishCommandedJointState_(const Vector7d& q_goal) {
  if (!commanded_joint_state_publisher_) {
    return;
  }

  sensor_msgs::msg::JointState msg;
  const auto current_time_ns = get_node()->now().nanoseconds();
  msg.header.stamp.sec = static_cast<int32_t>(current_time_ns / 1000000000LL);
  msg.header.stamp.nanosec = static_cast<uint32_t>(current_time_ns % 1000000000LL);
  msg.header.frame_id = arm_id_ + "_link0";
  msg.name.assign(joint_names_.begin(), joint_names_.end());
  msg.position.resize(num_joints);

  for (int i = 0; i < num_joints; ++i) {
    msg.position[i] = q_goal(i);
  }

  commanded_joint_state_publisher_->publish(msg);
}

bool DeploymentJointImpedanceController::validateGains_(const std::vector<double>& gains,
                                                        const std::string& gains_name) {
  if (gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "%s parameter not set", gains_name.c_str());
    return false;
  }

  if (gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "%s should be of size %d but is of size %ld",
                 gains_name.c_str(), num_joints, gains.size());
    return false;
  }

  return true;
}

void DeploymentJointImpedanceController::resetCommandTracking_(const rclcpp::Time& reference_time) {
  command_values_valid_ = false;
  command_values_.fill(0.0);
  last_command_time_ = reference_time;
  command_accept_time_ = reference_time;
}

void DeploymentJointImpedanceController::updateJointStates_() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

bool DeploymentJointImpedanceController::commandValuesAreFresh_() const {
  if (!command_values_valid_) {
    return false;
  }
  return (get_node()->now() - last_command_time_).seconds() <= command_timeout_sec_;
}

void DeploymentJointImpedanceController::holdCurrentPosition_() {
  hold_position_ = q_;
  hold_position_initialized_ = true;
  move_to_start_position_finished_ = false;
  motion_generator_initialized_ = false;
  motion_generator_.reset();
  start_time_ = get_node()->now();
}

void DeploymentJointImpedanceController::publishHoldPosition_() {
  if (!hold_position_initialized_) {
    holdCurrentPosition_();
  }
  Vector7d tau_d_calculated = calculateTauDGains_(hold_position_);
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }
  publishCommandedJointState_(hold_position_);
}

bool DeploymentJointImpedanceController::initializeMotionGenerator_() {
  if (!commandValuesAreFresh_()) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 10 * 1000,
                         "Waiting for fresh deployment joint targets...");
    return false;
  }

  Vector7d q_goal;
  updateJointStates_();
  for (int i = 0; i < num_joints; ++i) {
    q_goal(i) = command_values_[i];
  }

  const double motion_generator_speed_factor = 0.2;
  motion_generator_ = std::make_unique<MotionGenerator>(motion_generator_speed_factor, q_, q_goal);
  hold_position_initialized_ = false;
  return true;
}

void DeploymentJointImpedanceController::setDeploymentEnabled_(bool enabled) {
  deployment_enabled_ = enabled;
  resetCommandTracking_(get_node()->now());
  if (!enabled) {
    updateJointStates_();
    holdCurrentPosition_();
  } else {
    hold_position_initialized_ = false;
    motion_generator_initialized_ = false;
    move_to_start_position_finished_ = false;
    motion_generator_.reset();
    start_time_ = get_node()->now();
  }
}

void DeploymentJointImpedanceController::handleSetDeploymentEnabled_(
    const std::shared_ptr<std_srvs::srv::SetBool::Request>& request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
  setDeploymentEnabled_(request->data);
  response->success = true;
  response->message = request->data ? "enabled" : "disabled";
  RCLCPP_INFO(get_node()->get_logger(), "Deployment control %s via service request.",
              request->data ? "enabled" : "disabled");
}

}  // namespace franka_fr3_arm_controllers

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(franka_fr3_arm_controllers::DeploymentJointImpedanceController,
                       controller_interface::ControllerInterface)
