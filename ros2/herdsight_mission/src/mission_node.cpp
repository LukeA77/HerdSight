// HerdSight onboard mission (Phase 5 plan, step 7; D-16, D-41).
//
// The survey, with herd mode inside it, is registered with PX4 as a custom flight mode through
// px4-ros2-interface-lib, so PX4's own failsafes and arming checks stay in charge: if this node
// stops, PX4 leaves the mode and applies its failsafe (hold or return). A mode executor runs
// the mission as a behaviour tree (BehaviorTree.CPP): arm, take off, survey, return, wait until
// landed. Each still is published with PX4's own position estimate at the trigger time.
#include <behaviortree_cpp/bt_factory.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <herdsight_msgs/msg/herd_found.hpp>
#include <herdsight_msgs/msg/still_captured.hpp>
#include <memory>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/components/mode_executor.hpp>
#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/control/setpoint_types/multicopter/goto.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <rclcpp/rclcpp.hpp>

#include "herdsight_mission/survey_logic.hpp"

using namespace std::chrono_literals;

namespace herdsight {

class SurveyMode : public px4_ros2::ModeBase {
 public:
  explicit SurveyMode(rclcpp::Node& node) : ModeBase(node, Settings{"HerdSight Survey"}) {
    goto_ = std::make_shared<px4_ros2::MulticopterGotoSetpointType>(*this);
    local_ = std::make_shared<px4_ros2::OdometryLocalPosition>(*this);
    plan_file_ = node.declare_parameter<std::string>("plan_file", "");
    sortie_ = static_cast<size_t>(node.declare_parameter<int>("sortie", 0));
    // simulation only: start the mission as soon as it is registered, without a pilot selecting it
    autostart_ = node.declare_parameter<bool>("autostart", false);
    limits_.herd_mode_height_m =
        static_cast<float>(node.declare_parameter<double>("herd_mode_height_m", 40.0));
    limits_.clearance_m = static_cast<float>(node.declare_parameter<double>("clearance_m", 30.0));
    limits_.frame_along_m =
        static_cast<float>(node.declare_parameter<double>("frame_along_m", 0.0));
    // take survey stills in passing (false: stop at every photo point, as in the first flights)
    limits_.pass_through = node.declare_parameter<bool>("pass_through", true);
    still_pub_ = node.create_publisher<herdsight_msgs::msg::StillCaptured>("herdsight/still", 10);
    herd_sub_ = node.create_subscription<herdsight_msgs::msg::HerdFound>(
        "herdsight/herd_found", 10, [this](const herdsight_msgs::msg::HerdFound& m) {
          if (logic_) {
            // HerdFound uses east-north-up; the logic works north-east-down
            logic_->herdFound(Eigen::Vector2f(m.centre.y, m.centre.x), m.extent_m);
          }
        });
  }

  bool autostart() const { return autostart_; }

  void onActivate() override {
    plan_ = Plan::load(plan_file_);
    logic_ = std::make_unique<SurveyLogic>(plan_, limits_, sortie_);
    RCLCPP_INFO(node().get_logger(), "survey of %s: %d stills, %s", plan_.camp.c_str(),
                logic_->photosPlanned(), limits_.pass_through ? "in passing" : "stopping at each");
  }

  void onDeactivate() override {
    if (logic_) {
      RCLCPP_INFO(node().get_logger(), "survey left with %d of %d stills taken",
                  logic_->photosTaken(), logic_->photosPlanned());
    }
  }

  void updateSetpoint(float /*dt_s*/) override {
    if (!logic_ || !local_->positionXYValid()) return;
    const Eigen::Vector3f pos = local_->positionNed();
    if (auto c = logic_->update(pos)) publishStill(*c, pos);
    const auto t = logic_->flyTo(pos);
    if (!t) {
      completed(px4_ros2::Result::Success);
      return;
    }
    goto_->update(t->ned, plan_.yaw_ned_rad, plan_.speed_m_s);
  }

  int photosTaken() const { return logic_ ? logic_->photosTaken() : 0; }

 private:
  void publishStill(const Capture& c, const Eigen::Vector3f& pos) {
    herdsight_msgs::msg::StillCaptured m;
    m.header.stamp = node().now();
    m.header.frame_id = "map";
    m.index = static_cast<uint32_t>(c.photo_id < 0 ? 0 : c.photo_id);
    // the geotag is PX4's estimate, converted to east-north-up for home base
    m.position_estimate.x = pos.y();
    m.position_estimate.y = pos.x();
    m.position_estimate.z = -pos.z();
    m.height_agl_m = -pos.z();
    m.yaw_rad = plan_.yaw_ned_rad;
    m.gimbal_pitch_rad = -1.5708f;
    still_pub_->publish(m);
  }

  std::shared_ptr<px4_ros2::MulticopterGotoSetpointType> goto_;
  std::shared_ptr<px4_ros2::OdometryLocalPosition> local_;
  rclcpp::Publisher<herdsight_msgs::msg::StillCaptured>::SharedPtr still_pub_;
  rclcpp::Subscription<herdsight_msgs::msg::HerdFound>::SharedPtr herd_sub_;
  std::string plan_file_;
  size_t sortie_ = 0;
  bool autostart_ = false;
  Limits limits_;
  Plan plan_;
  std::unique_ptr<SurveyLogic> logic_;
};

// A behaviour-tree action that starts one asynchronous executor step and waits for its result.
class AsyncStep : public BT::StatefulActionNode {
 public:
  using Done = std::function<void(bool)>;
  using Starter = std::function<void(Done)>;
  AsyncStep(const std::string& name, const BT::NodeConfig& cfg, Starter start)
      : StatefulActionNode(name, cfg), start_(std::move(start)) {}
  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus onStart() override {
    done_ = ok_ = false;
    start_([this](bool ok) {
      ok_ = ok;
      done_ = true;
    });
    return BT::NodeStatus::RUNNING;
  }
  BT::NodeStatus onRunning() override {
    if (!done_) return BT::NodeStatus::RUNNING;
    return ok_ ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }
  void onHalted() override {}

 private:
  Starter start_;
  bool done_ = false;
  bool ok_ = false;
};

class MissionExecutor : public px4_ros2::ModeExecutorBase {
 public:
  explicit MissionExecutor(SurveyMode& survey)
      : ModeExecutorBase(Settings{}.activate(survey.autostart()
                                                 ? Settings::Activation::ActivateImmediately
                                                 : Settings::Activation::ActivateAlways),
                         survey),
        survey_(survey) {
    auto step = [this](const std::string& id, std::function<void(AsyncStep::Done)> start) {
      factory_.registerBuilder<AsyncStep>(
          id, [start](const std::string& name, const BT::NodeConfig& cfg) {
            return std::make_unique<AsyncStep>(name, cfg, start);
          });
    };
    auto ok = [](AsyncStep::Done done) {
      return [done](px4_ros2::Result r) { done(r == px4_ros2::Result::Success); };
    };
    step("WaitReadyToArm", [this, ok](auto done) { waitReadyToArm(ok(done)); });
    step("Arm", [this, ok](auto done) { arm(ok(done)); });
    step("TakeOff", [this, ok](auto done) { takeoff(ok(done)); });
    step("Survey", [this, ok](auto done) { scheduleMode(survey_.id(), ok(done)); });
    step("ReturnHome", [this, ok](auto done) { rtl(ok(done)); });
    step("WaitLanded", [this, ok](auto done) { waitUntilDisarmed(ok(done)); });
    tree_file_ = ament_index_cpp::get_package_share_directory("herdsight_mission") +
                 "/behavior_trees/mission.xml";
  }

  void onActivate() override {
    tree_ = std::make_unique<BT::Tree>(factory_.createTreeFromFile(tree_file_));
    timer_ = node().create_wall_timer(100ms, [this]() {
      if (!tree_) return;
      const auto status = tree_->tickOnce();
      if (status != BT::NodeStatus::RUNNING) {
        RCLCPP_INFO(node().get_logger(), "mission finished: %s",
                    BT::toStr(status).c_str());
        timer_->cancel();
      }
    });
  }

  void onDeactivate(DeactivateReason reason) override {
    // PX4 took over (failsafe or pilot); its own failsafe decides what happens next
    RCLCPP_WARN(node().get_logger(), "mission deactivated (%s)",
                reason == DeactivateReason::FailsafeActivated ? "failsafe" : "other");
    if (timer_) timer_->cancel();
    if (tree_) tree_->haltTree();
  }

 private:
  SurveyMode& survey_;
  BT::BehaviorTreeFactory factory_;
  std::unique_ptr<BT::Tree> tree_;
  std::string tree_file_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace herdsight

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<px4_ros2::NodeWithModeExecutor<herdsight::MissionExecutor,
                                                                 herdsight::SurveyMode>>(
      "herdsight_mission", false));
  rclcpp::shutdown();
  return 0;
}
