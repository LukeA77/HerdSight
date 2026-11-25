// HerdSight onboard mission through PX4 offboard setpoints (Phase 5 plan; D-16, D-41).
//
// Runs the same behaviour tree and the same SurveyLogic as the custom-mode mission
// (mission_node.cpp), so the survey, herd mode and welfare limits are one piece of code. A carrot
// moves along the straight line to SurveyLogic's current target at the plan's speed, slowing to
// stop at the target, and is streamed to PX4 as a position setpoint with a velocity feed-forward.
// PX4 must be set to return if the stream stops (COM_OBL_RC_ACT = 3, sim/bridge/run_flight.sh),
// so a crash of this node hands the drone to PX4's own return.
#include <behaviortree_cpp/bt_factory.h>

#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <cmath>
#include <functional>
#include <herdsight_msgs/msg/herd_found.hpp>
#include <herdsight_msgs/msg/still_captured.hpp>
#include <limits>
#include <memory>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <rclcpp/rclcpp.hpp>
#include <utility>
#include <vector>

#include "herdsight_mission/survey_logic.hpp"

using namespace std::chrono_literals;
using px4_msgs::msg::VehicleCommand;
using px4_msgs::msg::VehicleStatus;

namespace herdsight {

namespace {
constexpr float kLookaheadM = 6.0f;     // the carrot waits if the drone is further behind
constexpr float kBrakeMs2 = 2.0f;       // slowing towards a target
constexpr float kMinSpeed = 0.5f;       // so the carrot always reaches its target
constexpr float kMaxVerticalMs = 2.5f;  // climb and descent speed along a target line
constexpr float kClimbDoneM = 2.0f;
constexpr double kDtS = 0.02;
}  // namespace

// A behaviour-tree action defined by what it starts and how it knows it has finished.
class Step : public BT::StatefulActionNode {
 public:
  using Start = std::function<void()>;
  using Poll = std::function<BT::NodeStatus()>;
  Step(const std::string& name, const BT::NodeConfig& cfg, Start start, Poll poll)
      : StatefulActionNode(name, cfg), start_(std::move(start)), poll_(std::move(poll)) {}
  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus onStart() override {
    start_();
    return BT::NodeStatus::RUNNING;
  }
  BT::NodeStatus onRunning() override { return poll_(); }
  void onHalted() override {}

 private:
  Start start_;
  Poll poll_;
};

class OffboardMission : public rclcpp::Node {
 public:
  OffboardMission() : Node("herdsight_mission") {
    const auto plan_file = declare_parameter<std::string>("plan_file", "");
    const auto sortie = static_cast<size_t>(declare_parameter<int>("sortie", 0));
    declare_parameter<bool>("autostart", true);  // accepted for the same launch line
    Limits limits;
    limits.herd_mode_height_m = static_cast<float>(declare_parameter<double>("herd_mode_height_m", 40.0));
    limits.clearance_m = static_cast<float>(declare_parameter<double>("clearance_m", 30.0));
    limits.frame_along_m = static_cast<float>(declare_parameter<double>("frame_along_m", 0.0));
    limits.pass_through = declare_parameter<bool>("pass_through", true);
    // teach square: after take-off, fly a square of this half-side at teach_height_m around the
    // launch point before climbing, so the camera return has reference frames near the ground
    // around home (0 = straight climb)
    teach_half_m_ = static_cast<float>(declare_parameter<double>("teach_square_m", 0.0));
    teach_height_m_ = static_cast<float>(declare_parameter<double>("teach_height_m", 12.0));
    plan_ = Plan::load(plan_file);
    logic_ = std::make_unique<SurveyLogic>(plan_, limits, sortie);
    RCLCPP_INFO(get_logger(), "survey of %s: %d stills, offboard", plan_.camp.c_str(),
                logic_->photosPlanned());

    const auto qos = rclcpp::QoS(5).best_effort();
    mode_pub_ = create_publisher<px4_msgs::msg::OffboardControlMode>("/fmu/in/offboard_control_mode", qos);
    sp_pub_ = create_publisher<px4_msgs::msg::TrajectorySetpoint>("/fmu/in/trajectory_setpoint", qos);
    cmd_pub_ = create_publisher<VehicleCommand>("/fmu/in/vehicle_command", qos);
    still_pub_ = create_publisher<herdsight_msgs::msg::StillCaptured>("herdsight/still", 10);
    pos_sub_ = create_subscription<px4_msgs::msg::VehicleLocalPosition>(
        "/fmu/out/vehicle_local_position_v1", qos,
        [this](const px4_msgs::msg::VehicleLocalPosition& m) {
          pos_ = Eigen::Vector3f(m.x, m.y, m.z);
          pos_ok_ = m.xy_valid && m.z_valid;
        });
    status_sub_ = create_subscription<VehicleStatus>(
        "/fmu/out/vehicle_status_v1", qos, [this](const VehicleStatus& m) {
          nav_state_ = m.nav_state;
          armed_ = m.arming_state == VehicleStatus::ARMING_STATE_ARMED;
          checks_ok_ = m.pre_flight_checks_pass;
        });
    herd_sub_ = create_subscription<herdsight_msgs::msg::HerdFound>(
        "herdsight/herd_found", 10, [this](const herdsight_msgs::msg::HerdFound& m) {
          // HerdFound uses east-north-up; the logic works north-east-down
          if (phase_ == Phase::Survey) {
            logic_->herdFound(Eigen::Vector2f(m.centre.y, m.centre.x), m.extent_m);
          }
        });
    register_steps();
    tree_ = std::make_unique<BT::Tree>(factory_.createTreeFromFile(
        ament_index_cpp::get_package_share_directory("herdsight_mission") +
        "/behavior_trees/mission.xml"));
    control_timer_ = create_wall_timer(20ms, [this]() { control(); });
    tree_timer_ = create_wall_timer(100ms, [this]() {
      const auto status = tree_->tickOnce();
      if (status != BT::NodeStatus::RUNNING) {
        RCLCPP_INFO(get_logger(), "mission finished: %s", BT::toStr(status).c_str());
        tree_timer_->cancel();
      }
    });
  }

 private:
  enum class Phase { Idle, Climb, Survey, Px4 };  // Px4: PX4 is flying, nothing is streamed

  void register_steps() {
    auto step = [this](const std::string& id, Step::Start start, Step::Poll poll) {
      factory_.registerBuilder<Step>(id, [start, poll](const std::string& n, const BT::NodeConfig& c) {
        return std::make_unique<Step>(n, c, start, poll);
      });
    };
    auto done = [](bool ok) { return ok ? BT::NodeStatus::SUCCESS : BT::NodeStatus::RUNNING; };
    step("WaitReadyToArm", [] {}, [this, done] { return done(pos_ok_ && checks_ok_); });
    step(
        "Arm", [this] { arm_ticks_ = 0; },
        [this, done] {
          if (!armed_ || nav_state_ != VehicleStatus::NAVIGATION_STATE_OFFBOARD) {
            if (streamed_ > 50 && arm_ticks_++ % 10 == 0) {  // after 1 s of setpoints, every 1 s
              command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
              command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
            }
            return BT::NodeStatus::RUNNING;
          }
          return done(true);
        });
    step(
        "TakeOff",
        [this] {
          climb_path_.clear();
          const float x = pos_.x(), y = pos_.y();
          if (teach_half_m_ > 0.0f) {
            const float h = -teach_height_m_, d = teach_half_m_;
            const std::pair<float, float> corners[] = {{0.0f, 0.0f}, {d, d},   {d, -d},     {-d, -d},
                                                       {-d, d},     {d, d},   {0.0f, 0.0f}};
            for (const auto& [dx, dy] : corners) {
              climb_path_.emplace_back(x + dx, y + dy, h);
            }
            RCLCPP_INFO(get_logger(), "teach square: +/-%.0f m at %.0f m", d, teach_height_m_);
          }
          climb_path_.emplace_back(x, y, -plan_.height_agl_m);
          climb_step_ = 0;
          climb_target_ = climb_path_.front();
          phase_ = Phase::Climb;
        },
        [this] {
          if (lost_offboard()) {
            phase_ = Phase::Px4;  // PX4 has taken over: stop streaming
            return BT::NodeStatus::FAILURE;
          }
          if ((pos_ - climb_target_).norm() > kClimbDoneM) return BT::NodeStatus::RUNNING;
          if (++climb_step_ < climb_path_.size()) {
            climb_target_ = climb_path_[climb_step_];
            return BT::NodeStatus::RUNNING;
          }
          return BT::NodeStatus::SUCCESS;
        });
    step(
        "Survey", [this] { phase_ = Phase::Survey; },
        [this] {
          if (lost_offboard()) {
            RCLCPP_WARN(get_logger(), "PX4 left offboard with %d of %d stills taken",
                        logic_->photosTaken(), logic_->photosPlanned());
            phase_ = Phase::Px4;
            return BT::NodeStatus::FAILURE;
          }
          if (!logic_->done()) return BT::NodeStatus::RUNNING;
          RCLCPP_INFO(get_logger(), "survey left with %d of %d stills taken", logic_->photosTaken(),
                      logic_->photosPlanned());
          return BT::NodeStatus::SUCCESS;
        });
    step(
        "ReturnHome",
        [this] {
          phase_ = Phase::Px4;
          command(VehicleCommand::VEHICLE_CMD_NAV_RETURN_TO_LAUNCH);
        },
        [this, done] {
          if (nav_state_ != VehicleStatus::NAVIGATION_STATE_AUTO_RTL && armed_ &&
              rtl_ticks_++ % 10 == 0) {
            command(VehicleCommand::VEHICLE_CMD_NAV_RETURN_TO_LAUNCH);
          }
          return done(nav_state_ == VehicleStatus::NAVIGATION_STATE_AUTO_RTL || !armed_);
        });
    step("WaitLanded", [] {}, [this, done] { return done(!armed_); });
  }

  bool lost_offboard() const {
    return !armed_ || nav_state_ != VehicleStatus::NAVIGATION_STATE_OFFBOARD;
  }

  void command(uint32_t cmd, float p1 = 0.0f, float p2 = 0.0f) {
    VehicleCommand m;
    m.timestamp = static_cast<uint64_t>(get_clock()->now().nanoseconds() / 1000);
    m.command = cmd;
    m.param1 = p1;
    m.param2 = p2;
    m.target_system = 1;
    m.target_component = 1;
    m.source_system = 1;
    m.source_component = 1;
    m.from_external = true;
    cmd_pub_->publish(m);
  }

  void publish_still(const Capture& c) {
    herdsight_msgs::msg::StillCaptured m;
    m.header.stamp = now();
    m.header.frame_id = "map";
    m.index = static_cast<uint32_t>(c.photo_id < 0 ? 0 : c.photo_id);
    m.position_estimate.x = pos_.y();  // the geotag is PX4's estimate, east-north-up
    m.position_estimate.y = pos_.x();
    m.position_estimate.z = -pos_.z();
    m.height_agl_m = -pos_.z();
    m.yaw_rad = plan_.yaw_ned_rad;
    m.gimbal_pitch_rad = -1.5708f;
    still_pub_->publish(m);
  }

  void control() {
    if (phase_ == Phase::Px4 || !pos_ok_) return;
    const auto stamp = static_cast<uint64_t>(get_clock()->now().nanoseconds() / 1000);
    px4_msgs::msg::OffboardControlMode mode;
    mode.timestamp = stamp;
    mode.position = true;
    mode.velocity = true;
    mode_pub_->publish(mode);
    ++streamed_;
    if (!carrot_set_) {
      carrot_ = pos_;
      carrot_set_ = true;
    }
    Eigen::Vector3f target = carrot_;
    if (phase_ == Phase::Climb) {
      target = climb_target_;
    } else if (phase_ == Phase::Survey) {
      if (auto c = logic_->update(pos_)) publish_still(*c);
      const auto t = logic_->flyTo(pos_);
      if (t) target = t->ned;
    }
    const Eigen::Vector3f to = target - carrot_;
    const float dist = to.norm();
    Eigen::Vector3f vel = Eigen::Vector3f::Zero();
    if (dist > 1e-3f) {
      const Eigen::Vector3f u = to / dist;
      float v = std::min(plan_.speed_m_s, std::sqrt(2.0f * kBrakeMs2 * dist) + kMinSpeed);
      if (std::abs(u.z()) > 1e-3f) v = std::min(v, kMaxVerticalMs / std::abs(u.z()));
      vel = u * v;
      if ((pos_ - carrot_).norm() < kLookaheadM) {
        const float step = std::min(dist, v * static_cast<float>(kDtS));
        carrot_ += u * step;
      }
    }
    px4_msgs::msg::TrajectorySetpoint sp;
    sp.timestamp = stamp;
    sp.position = {carrot_.x(), carrot_.y(), carrot_.z()};
    sp.velocity = {vel.x(), vel.y(), vel.z()};
    sp.yaw = plan_.yaw_ned_rad;
    sp_pub_->publish(sp);
  }

  Plan plan_;
  std::unique_ptr<SurveyLogic> logic_;
  Phase phase_ = Phase::Idle;
  Eigen::Vector3f pos_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f carrot_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f climb_target_ = Eigen::Vector3f::Zero();
  std::vector<Eigen::Vector3f> climb_path_;
  size_t climb_step_ = 0;
  float teach_half_m_ = 0.0f;
  float teach_height_m_ = 12.0f;
  bool carrot_set_ = false;
  bool pos_ok_ = false;
  bool armed_ = false;
  bool checks_ok_ = false;
  uint8_t nav_state_ = std::numeric_limits<uint8_t>::max();
  int streamed_ = 0;
  int arm_ticks_ = 0;
  int rtl_ticks_ = 0;
  BT::BehaviorTreeFactory factory_;
  std::unique_ptr<BT::Tree> tree_;
  rclcpp::TimerBase::SharedPtr control_timer_, tree_timer_;
  rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr mode_pub_;
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr sp_pub_;
  rclcpp::Publisher<VehicleCommand>::SharedPtr cmd_pub_;
  rclcpp::Publisher<herdsight_msgs::msg::StillCaptured>::SharedPtr still_pub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr pos_sub_;
  rclcpp::Subscription<VehicleStatus>::SharedPtr status_sub_;
  rclcpp::Subscription<herdsight_msgs::msg::HerdFound>::SharedPtr herd_sub_;
};

}  // namespace herdsight

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<herdsight::OffboardMission>());
  rclcpp::shutdown();
  return 0;
}
