cmake_minimum_required(VERSION 3.16)
project(herdsight_mission)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(px4_ros2_cpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(herdsight_msgs REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

# Pure C++ mission logic (no ROS): the plan, the survey and herd-mode state machine.
add_library(mission_core STATIC src/plan.cpp src/survey_logic.cpp)
target_include_directories(mission_core PUBLIC include)
target_link_libraries(mission_core PUBLIC yaml-cpp Eigen3::Eigen)

add_executable(mission_node src/mission_node.cpp)
target_link_libraries(mission_node mission_core)
ament_target_dependencies(mission_node rclcpp ament_index_cpp px4_ros2_cpp px4_msgs herdsight_msgs behaviortree_cpp)

add_executable(offboard_mission_node src/offboard_mission_node.cpp)
target_link_libraries(offboard_mission_node mission_core)
ament_target_dependencies(offboard_mission_node rclcpp ament_index_cpp px4_msgs herdsight_msgs behaviortree_cpp)

install(TARGETS mission_node offboard_mission_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY behavior_trees DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_survey_logic test/test_survey_logic.cpp)
  target_link_libraries(test_survey_logic mission_core)
endif()

ament_package()
