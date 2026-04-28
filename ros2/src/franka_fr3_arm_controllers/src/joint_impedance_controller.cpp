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

#include <franka_fr3_arm_controllers/joint_impedance_controller.hpp>

#include <Eigen/Eigen>
#include <cassert>
#include <cmath>
#include <exception>
#include <string>

using std::placeholders::_1;

namespace franka_fr3_arm_controllers {

controller_interface::InterfaceConfiguration
JointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::state_interface_configuration() const {
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

controller_interface::return_type JointImpedanceController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  updateJointStates_();
  Vector7d q_goal;
  Vector7d tau_d_calculated;

  if (deployment_mode_ && !deployment_enabled_) {
    if (!hold_position_initialized_) {
      hold_position_ = q_;
      hold_position_initialized_ = true;
      q_goal_filter_initialized_ = false;
    }
    q_goal = hold_position_;
    q_goal = filterJointGoal_(q_goal, period);
    tau_d_calculated = calculateTauDGains_(q_goal);
    for (int i = 0; i < num_joints; ++i) {
      command_interfaces_[i].set_value(tau_d_calculated(i));
    }
    publishCommandedJointState_(q_goal);

    return controller_interface::return_type::OK;
  }

  if (!motion_generator_initialized_) {
    // After starting the controller we wait for valid joint states from the input topic
    // Until we get valid joint states we hold the current robot position to avoid
    // any drift or motion caused by stale commands or lack of an active teleop source.
    motion_generator_initialized_ = initializeMotionGenerator_();

    if (!motion_generator_initialized_) {
      if (!hold_position_initialized_) {
        hold_position_ = q_;
        hold_position_initialized_ = true;
        q_goal_filter_initialized_ = false;
      }
      q_goal = hold_position_;
      q_goal = filterJointGoal_(q_goal, period);
      tau_d_calculated = calculateTauDGains_(q_goal);
      for (int i = 0; i < num_joints; ++i) {
        command_interfaces_[i].set_value(tau_d_calculated(i));
      }
      publishCommandedJointState_(q_goal);

      return controller_interface::return_type::OK;
    }
  }

  if (!move_to_start_position_finished_) {
    // We have received valid joint states and initialized the motion generator
    // Now we move smoothly to the first joint position received from the input topic
    auto trajectory_time = this->get_node()->now() - start_time_;
    auto motion_generator_output = motion_generator_->getDesiredJointPositions(trajectory_time);
    move_to_start_position_finished_ = motion_generator_output.second;

    q_goal = motion_generator_output.first;
  }

  if (move_to_start_position_finished_) {
    // After reaching the start position we follow the joint position from the input topic
    // This is the normal operation mode of the controller
    if (!gello_position_values_valid_) {
      RCLCPP_FATAL(get_node()->get_logger(), "Timeout: No valid joint states received from Gello");
      rclcpp::shutdown();  // Exit the node permanently
    }
    for (int i = 0; i < num_joints; ++i) {
      q_goal(i) = gello_position_values_[i];
    }
  }

  q_goal = filterJointGoal_(q_goal, period);
  tau_d_calculated = calculateTauDGains_(q_goal);

  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }
  publishCommandedJointState_(q_goal);

  return controller_interface::return_type::OK;
}

void JointImpedanceController::jointStateCallback_(const sensor_msgs::msg::JointState msg) {
  if (last_joint_state_time_.seconds() == 0.0) {
    return;
  }

  if (msg.position.size() < gello_position_values_.size()) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Received joint state size is smaller than expected size.");
    return;
  }

  const auto msg_time = rclcpp::Time(msg.header.stamp);
  if (msg_time < command_accept_time_) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                         "Ignoring stale joint command message published before controller activation/enabling");
    return;
  }

  std::copy(msg.position.begin(), msg.position.begin() + gello_position_values_.size(),
            gello_position_values_.begin());

  validateGelloPositions_(msg);
  last_joint_state_time_ = msg.header.stamp;
}

CallbackReturn JointImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<std::vector<double>>("max_command_velocity", {});
    auto_declare<double>("command_filter_alpha", 0.01);
    auto_declare<bool>("deployment_mode", false);
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  namespace_prefix_ = get_node()->get_namespace();
  if (namespace_prefix_ == "/" || namespace_prefix_.empty()) {
    namespace_prefix_.clear();
  } else {
    // Remove leading slash and add trailing underscore
    namespace_prefix_ = namespace_prefix_.substr(1) + "_";
  }

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  auto k_alpha = get_node()->get_parameter("k_alpha").as_double();
  auto max_command_velocity =
      get_node()->get_parameter("max_command_velocity").as_double_array();
  auto command_filter_alpha = get_node()->get_parameter("command_filter_alpha").as_double();

  if (!validateGains_(k_gains, "k_gains") || !validateGains_(d_gains, "d_gains") ||
      !validateFilterParameters_(max_command_velocity)) {
    return CallbackReturn::FAILURE;
  }

  for (int i = 0; i < num_joints; ++i) {
    d_gains_(i) = d_gains.at(i);
    k_gains_(i) = k_gains.at(i);
    max_command_velocity_(i) = max_command_velocity.at(i);
  }

  if (k_alpha < 0.0 || k_alpha > 1.0) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_alpha should be in the range [0, 1]");
    return CallbackReturn::FAILURE;
  }

  if (command_filter_alpha <= 0.0 || command_filter_alpha > 1.0) {
    RCLCPP_FATAL(get_node()->get_logger(), "command_filter_alpha should be in the range (0, 1]");
    return CallbackReturn::FAILURE;
  }

  k_alpha_ = k_alpha;
  command_filter_alpha_ = command_filter_alpha;
  deployment_mode_ = get_node()->get_parameter("deployment_mode").as_bool();
  deployment_enabled_ = !deployment_mode_;

  dq_filtered_.setZero();
  q_goal_filtered_.setZero();
  q_goal_filter_initialized_ = false;
  for (int i = 0; i < num_joints; ++i) {
    joint_names_[i] = arm_id_ + "_joint" + std::to_string(i + 1);
  }

  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }

  joint_state_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      "gello/joint_states", 1,
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

CallbackReturn JointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  move_to_start_position_finished_ = false;
  motion_generator_initialized_ = false;
  hold_position_initialized_ = false;
  motion_generator_.reset();
  activation_time_ = get_node()->now();
  resetCommandTracking_(activation_time_);
  if (!deployment_mode_) {
    deployment_enabled_ = true;
  }
  dq_filtered_.setZero();
  q_goal_filtered_ = q_;
  q_goal_filter_initialized_ = true;
  start_time_ = activation_time_;

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  deployment_enabled_ = !deployment_mode_;
  hold_position_initialized_ = false;
  q_goal_filter_initialized_ = false;
  motion_generator_initialized_ = false;
  move_to_start_position_finished_ = false;
  motion_generator_.reset();

  return CallbackReturn::SUCCESS;
}

auto JointImpedanceController::calculateTauDGains_(const Vector7d& q_goal) -> Vector7d {
  dq_filtered_ = (1 - k_alpha_) * dq_filtered_ + k_alpha_ * dq_;
  Vector7d tau_d_calculated;
  tau_d_calculated = k_gains_.cwiseProduct(q_goal - q_) + d_gains_.cwiseProduct(-dq_filtered_);

  return tau_d_calculated;
}

void JointImpedanceController::publishCommandedJointState_(const Vector7d& q_goal) {
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

bool JointImpedanceController::validateGains_(const std::vector<double>& gains,
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

bool JointImpedanceController::validateFilterParameters_(
    const std::vector<double>& max_command_velocity) {
  if (max_command_velocity.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "max_command_velocity parameter not set");
    return false;
  }

  if (max_command_velocity.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "max_command_velocity should be of size %d but is of size %ld",
                 num_joints, max_command_velocity.size());
    return false;
  }

  for (const auto& velocity : max_command_velocity) {
    if (velocity <= 0.0) {
      RCLCPP_FATAL(get_node()->get_logger(), "max_command_velocity values must be positive");
      return false;
    }
  }

  return true;
}

JointImpedanceController::Vector7d JointImpedanceController::filterJointGoal_(
    const Vector7d& q_goal,
    const rclcpp::Duration& period) {
  if (!q_goal_filter_initialized_) {
    q_goal_filtered_ = q_;
    q_goal_filter_initialized_ = true;
  }

  const double dt = std::max(period.seconds(), 0.001);
  Vector7d q_filtered = command_filter_alpha_ * q_goal + (1.0 - command_filter_alpha_) * q_goal_filtered_;

  for (int i = 0; i < num_joints; ++i) {
    const double max_step = max_command_velocity_(i) * dt;
    const double step = std::clamp(q_filtered(i) - q_goal_filtered_(i), -max_step, max_step);
    q_filtered(i) = q_goal_filtered_(i) + step;
  }

  q_goal_filtered_ = q_filtered;
  return q_goal_filtered_;
}

void JointImpedanceController::validateGelloPositions_(const sensor_msgs::msg::JointState& msg) {
  const double max_time_diff = 2.0;  // Relaxed from 0.5 to allow for clock drift
  auto current_time = get_node()->now();
  auto time_since_last_joint_state = (current_time - last_joint_state_time_).seconds();
  auto time_since_msg_stamp = (current_time - msg.header.stamp).seconds();
  gello_position_values_valid_ =
      (time_since_last_joint_state < max_time_diff && time_since_msg_stamp < max_time_diff);
  if (!gello_position_values_valid_) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Gello position values are not valid. Time since last joint state: %f // Time "
                "since message stamp: %f",
                time_since_last_joint_state, time_since_msg_stamp);
  }
}

void JointImpedanceController::enterHoldMode_() {
  hold_position_ = q_;
  hold_position_initialized_ = true;
  q_goal_filter_initialized_ = false;
  move_to_start_position_finished_ = false;
  motion_generator_initialized_ = false;
  motion_generator_.reset();
  start_time_ = get_node()->now();
}

void JointImpedanceController::resetCommandTracking_(const rclcpp::Time& reference_time) {
  gello_position_values_valid_ = false;

  // Initialize with current pose instead of zeros
  for (int i = 0; i < 7; ++i) {
    gello_position_values_[i] = q_(i);
  }
  q_goal_filtered_ = q_;
  q_goal_filter_initialized_ = true;

  last_joint_state_time_ = reference_time;
  command_accept_time_ = reference_time;
}

void JointImpedanceController::setDeploymentEnabled_(bool enabled) {
  deployment_enabled_ = enabled;
  resetCommandTracking_(get_node()->now());
  if (!enabled) {
    updateJointStates_();
    enterHoldMode_();
  } else {
    hold_position_initialized_ = false;
  }
}

void JointImpedanceController::handleSetDeploymentEnabled_(
    const std::shared_ptr<std_srvs::srv::SetBool::Request>& request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
  if (!deployment_mode_) {
    response->success = true;
    response->message = "deployment_mode is false; teleoperation mode remains enabled.";
    return;
  }

  setDeploymentEnabled_(request->data);
  response->success = true;
  response->message = request->data ? "enabled" : "disabled";
  RCLCPP_INFO(
      get_node()->get_logger(), "Deployment control %s via service request.",
      request->data ? "enabled" : "disabled");
}

void JointImpedanceController::updateJointStates_() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

bool JointImpedanceController::initializeMotionGenerator_() {
  if (!joint_state_subscriber_ || joint_state_subscriber_->get_publisher_count() == 0) {
    gello_position_values_valid_ = false;
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 10 * 1000,
                         "Waiting for an active joint command publisher on gello/joint_states...");
    return false;
  }

  if (!gello_position_values_valid_) {
    // Only send a warning once every 10 seconds in order not to spam the log
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 10 * 1000,
                         "Waiting for valid joint states...");
    return false;
  }

  Vector7d q_goal;
  updateJointStates_();
  for (int i = 0; i < num_joints; ++i) {
    q_goal(i) = gello_position_values_[i];
  }
  RCLCPP_INFO(get_node()->get_logger(), "q_goal of motion generator: [%f, %f, %f, %f, %f, %f, %f]",
              q_goal(0), q_goal(1), q_goal(2), q_goal(3), q_goal(4), q_goal(5), q_goal(6));

  const double motion_generator_speed_factor = 0.2;
  motion_generator_ = std::make_unique<MotionGenerator>(motion_generator_speed_factor, q_, q_goal);
  return true;
}

}  // namespace franka_fr3_arm_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_fr3_arm_controllers::JointImpedanceController,
                       controller_interface::ControllerInterface)
