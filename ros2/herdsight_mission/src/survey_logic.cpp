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

std::optional<Capture> SurveyLogic::update(const Eigen::Vector3f& pos_ned) {
  if (queue_.empty()) return std::nullopt;
  const Target t = queue_.front();
  if ((pos_ned - t.ned).norm() > limits_.reached_m) return std::nullopt;
  queue_.erase(queue_.begin());
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
