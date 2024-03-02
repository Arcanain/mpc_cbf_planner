cmake_minimum_required(VERSION 3.8)
project(mpc_cbf_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(CasADi REQUIRED)
find_package(Eigen3 REQUIRED)

# mpc_cbf_planner
add_executable(mpc_cbf_planner src/mpc_cbf_planner.cpp)
ament_target_dependencies(mpc_cbf_planner rclcpp tf2_geometry_msgs tf2_ros nav_msgs geometry_msgs visualization_msgs)
target_include_directories(mpc_cbf_planner PRIVATE
  ${CASADI_INCLUDE_DIR}
)
target_link_libraries(mpc_cbf_planner
  ${CASADI_LIBRARIES}
  ${PYTHON_LIBRARIES}
  /usr/local/lib/libcasadi.so
)

# Install Cpp executables
install(TARGETS
  mpc_cbf_planner
  DESTINATION lib/${PROJECT_NAME}
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# Install other files
install(DIRECTORY
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
