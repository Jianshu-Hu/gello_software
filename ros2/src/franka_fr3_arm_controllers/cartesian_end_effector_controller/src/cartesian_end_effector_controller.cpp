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

#include <franka_fr3_arm_controllers/cartesian_end_effector_controller.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>

#include <Eigen/Geometry>

namespace {

Eigen::Quaterniond normalizedOrIdentity(Eigen::Quaterniond q) {
  const double norm = q.norm();
  if (!std::isfinite(norm) || norm < 1e-9) {
    return Eigen::Quaterniond::Identity();
  }
  q.normalize();
  return q;
}

Eigen::Vector3d translationFromColumnMajorArray(const std::array<double, 16>& pose) {
  return Eigen::Vector3d(pose[12], pose[13], pose[14]);
}

Eigen::Quaterniond orientationFromColumnMajorArray(const std::array<double, 16>& pose) {
  Eigen::Matrix3d rotation;
  for (int col = 0; col < 3; ++col) {
    for (int row = 0; row < 3; ++row) {
      rotation(row, col) = pose[static_cast<size_t>(col * 4 + row)];
    }
  }
  return normalizedOrIdentity(Eigen::Quaterniond(rotation));
}

double limitedAlpha(double dt, double time_constant) {
  if (time_constant <= 0.0) {
    return 1.0;
  }
  return std::clamp(dt / (time_constant + dt), 0.0, 1.0);
}

}  // namespace

namespace franka_fr3_arm_controllers {

controller_interface::InterfaceConfiguration
CartesianEndEffectorController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_pose_->get_command_interface_names();
  return config;
}

controller_interface::InterfaceConfiguration
CartesianEndEffectorController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_pose_->get_state_interface_names();
  return config;
}

controller_interface::return_type CartesianEndEffectorController::update(
    const rclcpp::Time& time,
    const rclcpp::Duration& period) {
  Eigen::Vector3d vr_position;
  Eigen::Quaterniond vr_orientation;
  const bool has_target = readLatestVrPose_(vr_position, vr_orientation);

  const auto current_pose = franka_cartesian_pose_->getCurrentPoseMatrix();
  const Eigen::Vector3d current_position = translationFromColumnMajorArray(current_pose);
  const Eigen::Quaterniond current_orientation = orientationFromColumnMajorArray(current_pose);

  if (!command_initialized_) {
    commanded_position_ = current_position;
    commanded_orientation_ = current_orientation;
    command_initialized_ = true;
  }

  const double target_age =
      time.seconds() - latest_vr_pose_time_sec_.load(std::memory_order_relaxed);
  if (!has_target || target_age > command_timeout_sec_) {
    reference_initialized_ = false;
    filterTargetPose_(current_position, current_orientation, period.seconds());
    franka_cartesian_pose_->setCommand(
        commanded_orientation_, commanded_position_);
    return controller_interface::return_type::OK;
  }

  if (!reference_initialized_) {
    resetReference_(vr_position, vr_orientation, current_position, current_orientation);
    consumed_vr_pose_sequence_ = latest_vr_pose_sequence_.load(std::memory_order_acquire);
  }

  Eigen::Vector3d target_position;
  Eigen::Quaterniond target_orientation;
  computeTargetPose_(vr_position, vr_orientation, target_position, target_orientation);
  filterTargetPose_(target_position, target_orientation, period.seconds());

  franka_cartesian_pose_->setCommand(
      commanded_orientation_, commanded_position_);
  return controller_interface::return_type::OK;
}

CallbackReturn CartesianEndEffectorController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::string>("target_pose_topic", "~/target_pose");
    auto_declare<double>("vr_position_scale", 1.0);
    auto_declare<double>("command_timeout_sec", 0.25);
    auto_declare<double>("filter_time_constant_sec", 0.08);
    auto_declare<double>("max_linear_velocity", 0.12);
    auto_declare<double>("max_angular_velocity", 0.70);
    auto_declare<double>("workspace_radius", 0.35);
    auto_declare<double>("min_z", 0.05);
    franka_cartesian_pose_ =
        std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(
            franka_semantic_components::FrankaCartesianPoseInterface(false));
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianEndEffectorController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (get_node()->get_parameter("arm_id").as_string().empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "arm_id parameter not set");
    return CallbackReturn::FAILURE;
  }

  target_pose_topic_ = get_node()->get_parameter("target_pose_topic").as_string();
  vr_position_scale_ = get_node()->get_parameter("vr_position_scale").as_double();
  command_timeout_sec_ = get_node()->get_parameter("command_timeout_sec").as_double();
  filter_time_constant_sec_ = get_node()->get_parameter("filter_time_constant_sec").as_double();
  max_linear_velocity_ = get_node()->get_parameter("max_linear_velocity").as_double();
  max_angular_velocity_ = get_node()->get_parameter("max_angular_velocity").as_double();
  workspace_radius_ = get_node()->get_parameter("workspace_radius").as_double();
  min_z_ = get_node()->get_parameter("min_z").as_double();

  if (vr_position_scale_ <= 0.0 || command_timeout_sec_ <= 0.0 ||
      filter_time_constant_sec_ < 0.0 || max_linear_velocity_ <= 0.0 ||
      max_angular_velocity_ <= 0.0 || workspace_radius_ <= 0.0 || min_z_ < 0.0) {
    RCLCPP_FATAL(get_node()->get_logger(), "Invalid Cartesian end-effector controller parameter");
    return CallbackReturn::FAILURE;
  }

  target_pose_subscriber_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      target_pose_topic_, rclcpp::SystemDefaultsQoS(),
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        targetPoseCallback_(msg);
      });

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianEndEffectorController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_pose_->assign_loaned_command_interfaces(command_interfaces_);
  franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_);

  const auto current_pose = franka_cartesian_pose_->getCurrentPoseMatrix();
  commanded_position_ = translationFromColumnMajorArray(current_pose);
  commanded_orientation_ = orientationFromColumnMajorArray(current_pose);
  reference_initialized_ = false;
  command_initialized_ = true;
  consumed_vr_pose_sequence_ = latest_vr_pose_sequence_.load(std::memory_order_acquire);
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianEndEffectorController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_pose_->release_interfaces();
  reference_initialized_ = false;
  command_initialized_ = false;
  return CallbackReturn::SUCCESS;
}

void CartesianEndEffectorController::targetPoseCallback_(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  if (rclcpp::Time(msg->header.stamp).nanoseconds() == 0) {
    latest_vr_pose_valid_.store(false, std::memory_order_release);
    latest_vr_pose_sequence_.fetch_add(1, std::memory_order_acq_rel);
    return;
  }

  latest_vr_pose_[0].store(msg->pose.position.x, std::memory_order_relaxed);
  latest_vr_pose_[1].store(msg->pose.position.y, std::memory_order_relaxed);
  latest_vr_pose_[2].store(msg->pose.position.z, std::memory_order_relaxed);
  latest_vr_pose_[3].store(msg->pose.orientation.x, std::memory_order_relaxed);
  latest_vr_pose_[4].store(msg->pose.orientation.y, std::memory_order_relaxed);
  latest_vr_pose_[5].store(msg->pose.orientation.z, std::memory_order_relaxed);
  latest_vr_pose_[6].store(msg->pose.orientation.w, std::memory_order_relaxed);
  latest_vr_pose_time_sec_.store(rclcpp::Time(msg->header.stamp).seconds(),
                                 std::memory_order_release);
  latest_vr_pose_valid_.store(true, std::memory_order_release);
  latest_vr_pose_sequence_.fetch_add(1, std::memory_order_acq_rel);
}

bool CartesianEndEffectorController::readLatestVrPose_(
    Eigen::Vector3d& position,
    Eigen::Quaterniond& orientation) const {
  if (!latest_vr_pose_valid_.load(std::memory_order_acquire)) {
    return false;
  }
  position = Eigen::Vector3d(latest_vr_pose_[0].load(std::memory_order_relaxed),
                             latest_vr_pose_[1].load(std::memory_order_relaxed),
                             latest_vr_pose_[2].load(std::memory_order_relaxed));
  orientation = normalizedOrIdentity(Eigen::Quaterniond(
      latest_vr_pose_[6].load(std::memory_order_relaxed),
      latest_vr_pose_[3].load(std::memory_order_relaxed),
      latest_vr_pose_[4].load(std::memory_order_relaxed),
      latest_vr_pose_[5].load(std::memory_order_relaxed)));
  return true;
}

void CartesianEndEffectorController::resetReference_(
    const Eigen::Vector3d& vr_position,
    const Eigen::Quaterniond& vr_orientation,
    const Eigen::Vector3d& current_position,
    const Eigen::Quaterniond& current_orientation) {
  reference_vr_position_ = vr_position;
  reference_vr_orientation_ = normalizedOrIdentity(vr_orientation);
  reference_ee_position_ = current_position;
  reference_ee_orientation_ = normalizedOrIdentity(current_orientation);
  reference_initialized_ = true;
}

void CartesianEndEffectorController::computeTargetPose_(
    const Eigen::Vector3d& vr_position,
    const Eigen::Quaterniond& vr_orientation,
    Eigen::Vector3d& target_position,
    Eigen::Quaterniond& target_orientation) {
  Eigen::Vector3d delta_position = (vr_position - reference_vr_position_) * vr_position_scale_;
  const double delta_norm = delta_position.norm();
  if (delta_norm > workspace_radius_) {
    delta_position *= workspace_radius_ / delta_norm;
  }

  target_position = reference_ee_position_ + delta_position;
  target_position.z() = std::max(target_position.z(), min_z_);

  Eigen::Quaterniond delta_orientation =
      normalizedOrIdentity(vr_orientation) * reference_vr_orientation_.inverse();
  target_orientation = normalizedOrIdentity(delta_orientation * reference_ee_orientation_);
}

void CartesianEndEffectorController::filterTargetPose_(
    const Eigen::Vector3d& target_position,
    const Eigen::Quaterniond& target_orientation,
    double dt) {
  dt = std::clamp(dt, 0.001, 0.02);
  const double alpha = limitedAlpha(dt, filter_time_constant_sec_);

  Eigen::Vector3d filtered_position =
      alpha * target_position + (1.0 - alpha) * commanded_position_;
  Eigen::Vector3d position_step = filtered_position - commanded_position_;
  const double max_position_step = max_linear_velocity_ * dt;
  if (position_step.norm() > max_position_step) {
    position_step *= max_position_step / position_step.norm();
  }
  commanded_position_ += position_step;

  Eigen::Quaterniond filtered_orientation =
      commanded_orientation_.slerp(alpha, normalizedOrIdentity(target_orientation));
  const double angular_distance = commanded_orientation_.angularDistance(filtered_orientation);
  const double max_angular_step = max_angular_velocity_ * dt;
  if (angular_distance > max_angular_step && angular_distance > 1e-9) {
    filtered_orientation = commanded_orientation_.slerp(max_angular_step / angular_distance,
                                                        filtered_orientation);
  }
  commanded_orientation_ = normalizedOrIdentity(filtered_orientation);
}

}  // namespace franka_fr3_arm_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_fr3_arm_controllers::CartesianEndEffectorController,
                       controller_interface::ControllerInterface)
