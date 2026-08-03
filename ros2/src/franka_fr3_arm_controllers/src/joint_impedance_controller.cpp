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
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <exception>
#include <string>
#include <vector>

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

controller_interface::return_type JointImpedanceController::update(const rclcpp::Time& /*time*/,
                                                                   const rclcpp::Duration& period) {
  updateJointStates_();
  Vector7d q_goal;

  if (!motion_generator_initialized_) {
    motion_generator_initialized_ = initializeMotionGenerator_();
    if (!motion_generator_initialized_) {
      if (!hold_position_initialized_) {
        hold_position_ = q_;
        hold_position_initialized_ = true;
      }
      q_goal = hold_position_;
    }
  }

  if (motion_generator_initialized_ && !move_to_start_position_finished_) {
    auto trajectory_time = this->get_node()->now() - start_time_;
    auto motion_generator_output = motion_generator_->getDesiredJointPositions(trajectory_time);
    const bool was_finished = move_to_start_position_finished_;
    move_to_start_position_finished_ = motion_generator_output.second;
    q_goal = motion_generator_output.first;
    if (!was_finished && move_to_start_position_finished_ && safe_mode_ != SafeMode::kOff) {
      safe_target_limiter_.reset(q_goal);
    }
  }

  if (motion_generator_initialized_ && move_to_start_position_finished_) {
    bool command_valid = gello_position_values_valid_;
    if (safe_mode_ != SafeMode::kOff) {
      constexpr double kSafeCommandTimeoutSec = 0.5;
      command_valid = command_valid && (get_node()->now() - last_joint_state_time_).seconds() <
                                           kSafeCommandTimeoutSec;
    }

    if (!command_valid) {
      if (safe_mode_ == SafeMode::kOff) {
        RCLCPP_FATAL(get_node()->get_logger(),
                     "Timeout: No valid joint states received from Gello");
        rclcpp::shutdown();
        for (int i = 0; i < num_joints; ++i) {
          q_goal(i) = gello_position_values_[i];
        }
      } else {
        RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                              "Safe mode is holding measured position because the GELLO command "
                              "stream is stale or invalid");
        q_goal = q_;
        safe_target_limiter_.reset(q_);
      }
    } else {
      Vector7d raw_target;
      for (int i = 0; i < num_joints; ++i) {
        raw_target(i) = gello_position_values_[i];
      }

      if (safe_mode_ == SafeMode::kOff) {
        q_goal = raw_target;
      } else {
        double dt_seconds = period.seconds();
        if (!std::isfinite(dt_seconds) || dt_seconds <= 0.0) {
          RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                               "Invalid controller period; using 0.001 s for safe mode");
          dt_seconds = 0.001;
        }
        if (!safe_target_limiter_.initialized()) {
          safe_target_limiter_.reset(q_);
        }
        const auto safe_result = safe_target_limiter_.update(raw_target, q_, dt_seconds);
        if (!safe_result.valid) {
          RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                                "Safe mode rejected a non-finite joint target; holding position");
          q_goal = q_;
        } else {
          publishSafeModeDiagnostics_(raw_target, safe_result);
          q_goal = safe_mode_ == SafeMode::kEnforce ? safe_result.target : raw_target;
        }
      }
    }
  }

  Vector7d tau_d_calculated = calculateTauDGains_(q_goal);
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
                         "Ignoring stale teleoperation joint command message published before "
                         "controller activation");
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
    auto_declare<double>("k_alpha", 0.99);
    const auto safe_defaults = SafeJointTargetLimiter::defaultConfig();
    auto vector_from_eigen = [](const Vector7d& values) {
      return std::vector<double>(values.data(), values.data() + values.size());
    };
    auto_declare<std::string>("safe_mode", "off");
    auto_declare<double>("safe_position_margin_rad", 0.02);
    auto_declare<std::vector<double>>("safe_max_velocity_rad_s",
                                      vector_from_eigen(safe_defaults.max_velocity));
    auto_declare<std::vector<double>>("safe_max_acceleration_rad_s2",
                                      vector_from_eigen(safe_defaults.max_acceleration));
    auto_declare<std::vector<double>>("safe_max_tracking_error_rad",
                                      vector_from_eigen(safe_defaults.max_tracking_error));
    auto_declare<double>("safe_diagnostic_publish_rate_hz", 25.0);
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
    namespace_prefix_ = namespace_prefix_.substr(1) + "_";
  }

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  auto k_alpha = get_node()->get_parameter("k_alpha").as_double();

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
  k_alpha_ = k_alpha;
  dq_filtered_.setZero();

  if (!configureSafeMode_()) {
    return CallbackReturn::FAILURE;
  }

  joint_state_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      "gello/joint_states", 1,
      [this](const sensor_msgs::msg::JointState& msg) { jointStateCallback_(msg); });
  commanded_joint_state_publisher_ = get_node()->create_publisher<sensor_msgs::msg::JointState>(
      "franka/commanded_joint_states", 10);
  if (safe_mode_ != SafeMode::kOff) {
    raw_commanded_joint_state_publisher_ =
        get_node()->create_publisher<sensor_msgs::msg::JointState>(
            "franka/raw_commanded_joint_states", 10);
    safe_commanded_joint_state_publisher_ =
        get_node()->create_publisher<sensor_msgs::msg::JointState>(
            "franka/safe_commanded_joint_states", 10);
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  move_to_start_position_finished_ = false;
  motion_generator_initialized_ = false;
  hold_position_initialized_ = false;
  motion_generator_.reset();
  const rclcpp::Time activation_time = get_node()->now();
  resetCommandTracking_(activation_time);
  safe_target_limiter_.clear();
  last_safe_diagnostic_publish_time_ = activation_time;
  dq_filtered_.setZero();
  start_time_ = activation_time;

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  hold_position_initialized_ = false;
  motion_generator_initialized_ = false;
  move_to_start_position_finished_ = false;
  motion_generator_.reset();
  safe_target_limiter_.clear();

  return CallbackReturn::SUCCESS;
}

auto JointImpedanceController::calculateTauDGains_(const Vector7d& q_goal) -> Vector7d {
  dq_filtered_ = (1 - k_alpha_) * dq_filtered_ + k_alpha_ * dq_;
  return k_gains_.cwiseProduct(q_goal - q_) + d_gains_.cwiseProduct(-dq_filtered_);
}

bool JointImpedanceController::configureSafeMode_() {
  const std::string mode = get_node()->get_parameter("safe_mode").as_string();
  if (mode == "off") {
    safe_mode_ = SafeMode::kOff;
  } else if (mode == "monitor") {
    safe_mode_ = SafeMode::kMonitor;
  } else if (mode == "enforce") {
    safe_mode_ = SafeMode::kEnforce;
  } else {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "safe_mode must be one of: off, monitor, enforce (received '%s')", mode.c_str());
    return false;
  }

  const double margin = get_node()->get_parameter("safe_position_margin_rad").as_double();
  const double diagnostic_rate =
      get_node()->get_parameter("safe_diagnostic_publish_rate_hz").as_double();
  const auto max_velocity = get_node()->get_parameter("safe_max_velocity_rad_s").as_double_array();
  const auto max_acceleration =
      get_node()->get_parameter("safe_max_acceleration_rad_s2").as_double_array();
  const auto max_tracking_error =
      get_node()->get_parameter("safe_max_tracking_error_rad").as_double_array();

  if (!std::isfinite(diagnostic_rate) || diagnostic_rate <= 0.0) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "safe_diagnostic_publish_rate_hz must be finite and positive");
    return false;
  }
  if (!validateGains_(max_velocity, "safe_max_velocity_rad_s") ||
      !validateGains_(max_acceleration, "safe_max_acceleration_rad_s2") ||
      !validateGains_(max_tracking_error, "safe_max_tracking_error_rad")) {
    return false;
  }

  try {
    auto config = SafeJointTargetLimiter::defaultConfig(margin);
    for (int i = 0; i < num_joints; ++i) {
      config.max_velocity(i) = max_velocity.at(i);
      config.max_acceleration(i) = max_acceleration.at(i);
      config.max_tracking_error(i) = max_tracking_error.at(i);
    }
    safe_target_limiter_.configure(config);
  } catch (const std::exception& exception) {
    RCLCPP_FATAL(get_node()->get_logger(), "Invalid safe mode configuration: %s", exception.what());
    return false;
  }

  safe_diagnostic_publish_period_sec_ = 1.0 / diagnostic_rate;
  RCLCPP_INFO(get_node()->get_logger(), "Collection safe mode: %s", mode.c_str());
  return true;
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

void JointImpedanceController::publishSafeModeDiagnostics_(
    const Vector7d& raw_target,
    const SafeJointTargetLimiter::Result& result) {
  if (!raw_commanded_joint_state_publisher_ || !safe_commanded_joint_state_publisher_) {
    return;
  }

  const auto now = get_node()->now();
  if (last_safe_diagnostic_publish_time_.nanoseconds() != 0 &&
      (now - last_safe_diagnostic_publish_time_).seconds() < safe_diagnostic_publish_period_sec_) {
    return;
  }
  last_safe_diagnostic_publish_time_ = now;

  sensor_msgs::msg::JointState raw_msg;
  raw_msg.header.stamp = now;
  raw_msg.header.frame_id = arm_id_ + "_link0";
  raw_msg.name.assign(joint_names_.begin(), joint_names_.end());
  raw_msg.position.assign(raw_target.data(), raw_target.data() + raw_target.size());
  raw_commanded_joint_state_publisher_->publish(raw_msg);

  sensor_msgs::msg::JointState safe_msg = raw_msg;
  safe_msg.position.assign(result.target.data(), result.target.data() + result.target.size());
  safe_msg.velocity.assign(result.target_velocity.data(),
                           result.target_velocity.data() + result.target_velocity.size());
  safe_commanded_joint_state_publisher_->publish(safe_msg);

  if (result.anyLimited()) {
    const char* action = safe_mode_ == SafeMode::kEnforce ? "limited" : "would limit";
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                         "Collection safe mode %s GELLO target (max raw-to-safe error %.4f rad)",
                         action, result.max_raw_to_safe_error_rad);
  }
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

void JointImpedanceController::validateGelloPositions_(const sensor_msgs::msg::JointState& msg) {
  const double max_time_diff = 0.5;
  auto current_time = get_node()->now();
  auto time_since_last_joint_state = (current_time - last_joint_state_time_).seconds();
  auto time_since_msg_stamp = (current_time - msg.header.stamp).seconds();
  gello_position_values_valid_ =
      (time_since_last_joint_state < max_time_diff && time_since_msg_stamp < max_time_diff);
  if (safe_mode_ != SafeMode::kOff &&
      !std::all_of(gello_position_values_.begin(), gello_position_values_.end(),
                   [](double value) { return std::isfinite(value); })) {
    gello_position_values_valid_ = false;
    RCLCPP_WARN(get_node()->get_logger(), "Gello joint command contains a non-finite value");
  }
  if (!gello_position_values_valid_) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Gello position values are not valid. Time since last joint state: %f // Time "
                "since message stamp: %f",
                time_since_last_joint_state, time_since_msg_stamp);
  }
}

void JointImpedanceController::resetCommandTracking_(const rclcpp::Time& reference_time) {
  gello_position_values_valid_ = false;
  gello_position_values_.fill(0.0);
  last_joint_state_time_ = reference_time;
  command_accept_time_ = reference_time;
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

  if (safe_mode_ != SafeMode::kOff &&
      (get_node()->now() - last_joint_state_time_).seconds() >= 0.5) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 10 * 1000,
                         "Waiting for a fresh GELLO joint command before safe startup...");
    return false;
  }

  if (!gello_position_values_valid_) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 10 * 1000,
                         "Waiting for valid joint states...");
    return false;
  }

  Vector7d q_goal;
  updateJointStates_();
  for (int i = 0; i < num_joints; ++i) {
    q_goal(i) = gello_position_values_[i];
  }

  if (safe_mode_ == SafeMode::kEnforce) {
    bool position_limited = false;
    q_goal = safe_target_limiter_.clampToPositionRange(q_goal, &position_limited);
    if (position_limited) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Initial GELLO target was clamped to the configured safe position range");
    }
  }

  const double motion_generator_speed_factor = 0.2;
  motion_generator_ = std::make_unique<MotionGenerator>(motion_generator_speed_factor, q_, q_goal);
  return true;
}

}  // namespace franka_fr3_arm_controllers

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(franka_fr3_arm_controllers::JointImpedanceController,
                       controller_interface::ControllerInterface)
