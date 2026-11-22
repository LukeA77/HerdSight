// Survey and herd-mode logic without ROS (Phase 5 plan, step 7; D-6, D-7).
//
// The plan comes from python/herdsight/mission/planner.py (plan_to_dict, written as YAML).
// Positions are in PX4's local frame: north, east, down (metres from the take-off point).
// SurveyLogic walks the plan's photo points in order. When the onboard trigger finds a herd,
// it inserts a herd-mode inspection: climb or stay at survey height to a descent point clear of
// the herd, descend there to the herd-mode height, move over the herd, take one still, climb
// back to survey height, and return to where the survey stopped. The drone never descends within
// the clearance radius of the herd and never flies below the herd-mode height (D-6).
#pragma once

#include <Eigen/Core>
#include <optional>
#include <string>
#include <vector>

namespace herdsight {

struct PhotoPoint {
  int id;
  Eigen::Vector3f ned;  // down is negative height above take-off
};

struct Plan {
  std::string camp;
  float height_agl_m{0};
  float speed_m_s{0};
  float yaw_ned_rad{0};
  std::vector<std::vector<PhotoPoint>> sorties;
  static Plan load(const std::string& yaml_path);
};

struct Limits {
  float herd_mode_height_m{40.0f};   // never lower (configs/mission.yaml)
  float clearance_m{30.0f};          // descend only this far from the herd's edge
  float reached_m{1.5f};             // a target counts as reached within this distance
  float frame_along_m{0.0f};         // survey still's along-track footprint at survey height
};

enum class TargetKind { Photo, Transit, HerdPhoto };

struct Target {
  Eigen::Vector3f ned;
  TargetKind kind;
  int photo_id;  // -1 unless kind == Photo
};

struct Capture {
  int photo_id;  // -1 for a herd-mode still
  TargetKind kind;
  Eigen::Vector3f ned;
};

class SurveyLogic {
 public:
  SurveyLogic(Plan plan, Limits limits, size_t sortie = 0);

  // The target to fly to now; empty when the sortie is finished.
  std::optional<Target> current() const;
  // Report the position estimate; if the current target is reached, advance and return the
  // still to take there (if any).
  std::optional<Capture> update(const Eigen::Vector3f& pos_ned);
  // The onboard trigger found a herd: centre (north, east) and diameter. Returns false if no
  // herd mode is needed (the herd already fits a survey still) or one is already queued.
  bool herdFound(const Eigen::Vector2f& centre_ne, float extent_m);

  bool done() const { return queue_.empty(); }
  int photosTaken() const { return photos_taken_; }
  int photosPlanned() const { return photos_planned_; }
  int herdStills() const { return herd_stills_; }
  // Lowest height above take-off reached so far by any planned target (for welfare checks).
  float lowestHeightM() const { return lowest_height_; }

 private:
  Plan plan_;
  Limits limits_;
  std::vector<Target> queue_;  // front is current
  size_t next_ = 0;
  int photos_taken_ = 0;
  int photos_planned_ = 0;
  int herd_stills_ = 0;
  bool herd_queued_ = false;
  float lowest_height_;
};

}  // namespace herdsight
