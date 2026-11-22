#include <yaml-cpp/yaml.h>

#include <stdexcept>

#include "herdsight_mission/survey_logic.hpp"

namespace herdsight {

Plan Plan::load(const std::string& yaml_path) {
  const YAML::Node d = YAML::LoadFile(yaml_path);
  Plan p;
  p.camp = d["camp"].as<std::string>();
  p.height_agl_m = d["height_agl_m"].as<float>();
  p.speed_m_s = d["speed_m_s"].as<float>();
  p.yaw_ned_rad = d["yaw_ned_rad"].as<float>();
  for (const auto& s : d["sorties"]) {
    std::vector<PhotoPoint> photos;
    for (const auto& ph : s["photos"]) {
      const auto ned = ph["ned"];
      photos.push_back({ph["id"].as<int>(),
                        Eigen::Vector3f(ned[0].as<float>(), ned[1].as<float>(), ned[2].as<float>())});
    }
    p.sorties.push_back(std::move(photos));
  }
  if (p.sorties.empty()) throw std::runtime_error("plan has no sorties: " + yaml_path);
  return p;
}

}  // namespace herdsight
