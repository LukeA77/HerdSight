#include "herdsight_mission/survey_logic.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace herdsight {

SurveyLogic::SurveyLogic(Plan plan, Limits limits, size_t sortie)
    : plan_(std::move(plan)), limits_(limits), lowest_height_(plan_.height_agl_m) {
  if (sortie >= plan_.sorties.size()) throw std::out_of_range("no such sortie");
  for (const auto& ph : plan_.sorties[sortie]) {
    queue_.push_back({ph.ned, TargetKind::Photo, ph.id});
    lowest_height_ = std::min(lowest_height_, -ph.ned.z());
  }
  photos_planned_ = static_cast<int>(queue_.size());
}

std::optional<Target> SurveyLogic::current() const {
  if (queue_.empty()) return std::nullopt;
  return queue_.front();
}

std::optional<Target> SurveyLogic::flyTo(const Eigen::Vector3f& pos_ned) const {
  if (queue_.empty()) return std::nullopt;
  const Target& first = queue_.front();
  if (!limits_.pass_through || first.kind != TargetKind::Photo || !last_) return first;
  // only once the drone is on the line through the last point and the current one
  const Eigen::Vector3f leg = first.ned - *last_;
  if (leg.norm() < 1e-3f) return first;
  const Eigen::Vector3f dir = leg.normalized();
  const Eigen::Vector3f off = pos_ned - *last_;
  if ((off - off.dot(dir) * dir).norm() > limits_.capture_m) return first;
  size_t end = 0;
  while (end + 1 < queue_.size() && queue_[end + 1].kind == TargetKind::Photo) {
    const Eigen::Vector3f step = queue_[end + 1].ned - queue_[end].ned;
    if (step.norm() < 1e-3f || step.normalized().dot(dir) < limits_.line_cos) break;
    ++end;
  }
  return queue_[end];
}

// In passing, a photo point counts as reached once the drone, close enough to the line, is level
// with it or beyond; otherwise (and for every other target) within reached_m.
bool SurveyLogic::passed(const Target& t, const Eigen::Vector3f& pos_ned) const {
  if ((pos_ned - t.ned).norm() <= limits_.reached_m) return true;
  if (!limits_.pass_through || t.kind != TargetKind::Photo || !last_) return false;
  const Eigen::Vector3f leg = t.ned - *last_;
  if (leg.norm() < 1e-3f) return false;
  const Eigen::Vector3f dir = leg.normalized();
  const Eigen::Vector3f off = pos_ned - t.ned;
  const float along = off.dot(dir);
  return along >= 0.0f && (off - along * dir).norm() <= limits_.capture_m;
}

std::optional<Capture> SurveyLogic::update(const Eigen::Vector3f& pos_ned) {
  if (queue_.empty()) return std::nullopt;
  const Target t = queue_.front();
  if (!passed(t, pos_ned)) return std::nullopt;
  queue_.erase(queue_.begin());
  last_ = t.ned;
  if (t.kind == TargetKind::Photo) {
    ++photos_taken_;
    return Capture{t.photo_id, t.kind, t.ned};
  }
  if (t.kind == TargetKind::HerdPhoto) {
    ++herd_stills_;
    herd_queued_ = false;
    return Capture{-1, t.kind, t.ned};
  }
  return std::nullopt;
}

bool SurveyLogic::herdFound(const Eigen::Vector2f& centre_ne, float extent_m) {
  if (herd_queued_ || queue_.empty()) return false;
  const float survey_h = plan_.height_agl_m;
  // lowest height (not below the herd-mode floor) whose still frames the herd with a margin
  float h = limits_.herd_mode_height_m;
  if (limits_.frame_along_m > 0) {
    h = std::max(h, survey_h * 1.2f * extent_m / limits_.frame_along_m);
  }
  if (h >= survey_h) return false;  // the herd already fits a survey still
  // where the survey stopped: the drone returns there afterwards
  const Target resume = queue_.front();
  Eigen::Vector2f here(resume.ned.x(), resume.ned.y());
  Eigen::Vector2f away = here - centre_ne;
  if (away.norm() < 1e-3f) away = Eigen::Vector2f(1.0f, 0.0f);
  away.normalize();
  const Eigen::Vector2f descent = centre_ne + away * (extent_m / 2 + limits_.clearance_m);
  const float ground = resume.ned.z() + survey_h;  // down coordinate of the ground, approx.
  auto at = [&](const Eigen::Vector2f& ne, float height) {
    return Eigen::Vector3f(ne.x(), ne.y(), ground - height);
  };
  std::vector<Target> ins = {
      {at(descent, survey_h), TargetKind::Transit, -1},   // approach at survey height
      {at(descent, h), TargetKind::Transit, -1},          // descend clear of the herd
      {at(centre_ne, h), TargetKind::HerdPhoto, -1},      // over the herd at herd-mode height
      {at(centre_ne, survey_h), TargetKind::Transit, -1}, // climb straight up
  };
  queue_.insert(queue_.begin(), ins.begin(), ins.end());
  lowest_height_ = std::min(lowest_height_, h);
  herd_queued_ = true;
  return true;
}

}  // namespace herdsight
