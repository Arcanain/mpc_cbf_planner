#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/msg/marker_array.hpp"

#include <casadi/casadi.hpp>
#include <cmath>
#include <iostream>
#include <vector>

using namespace casadi;
using std::placeholders::_1;
using namespace std::chrono_literals;

class MPCNode : public rclcpp::Node
{
public:
  MPCNode()
  : Node("mpc_cbf_planner")
  {
    // システムパラメータの設定
    dt = 0.25;
    nx = 3;
    nu = 2;
    N = 20;
    P = 100 * casadi::DM::eye(nx);
    // Q = casadi::DM::diag(casadi::DM({10.0, 10.0, 1.0}));
    // R = casadi::DM::diag(casadi::DM({1.0, 0.1}));
    // Q = casadi::DM::diag(casadi::DM({1.0, 1.0, 1.0}));
    // R = casadi::DM::diag(casadi::DM({1.0, 0.1}));
    Q = casadi::DM::diag(casadi::DM({10.0, 10.0, 1.0}));
    R = casadi::DM::diag(casadi::DM({1.0, 1.0}));
    p = 200;
    gamma = 0.2;

    // 制約の設定
    // umin = casadi::DM({-1, -1});
    // umax = casadi::DM({1, 1});
    // umin = casadi::DM({-2, -1});
    // umax = casadi::DM({2, 1});
    umin = casadi::DM({-0.5, -0.5});
    umax = casadi::DM({0.5, 0.5});

    // 初期状態の設定
    xTrue = casadi::DM({0.0, 0.0, 0.0});

    // 目標値
    xTarget = casadi::DM({6.0, 0.0, 0.0});

    // システム行列の設定
    A = casadi::DM::eye(nx);
    B = casadi::DM(
      {{dt * cos(xTrue(2).scalar()), 0.0},
       {dt * sin(xTrue(2).scalar()), 0.0},
       {0.0, dt}});

    // 障害物の設定
    //std::vector<casadi::DM> obs_pos_list = {{3.0, 0.0}, {4.0, 0.0}, {4.0, 1.0}};
    //std::vector<casadi::DM> obs_pos_list = {{3.0, 0.0}, {4.0, 1.0}, {4.0, 2.0}};
    //std::vector<casadi::DM> obs_pos_list = {{3.0, 0.0}, {4.0, 0.0}, {4.0, 1.0}};
    vehicle_diameter = 0.25;
    obs_diameter = 0.25;
    obs_r = vehicle_diameter + obs_diameter;

    // MPC予測軌道のPublish
    mpc_predicted_path_pub =
      this->create_publisher<nav_msgs::msg::Path>("mpc_predicted_path", 50);
    odom_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

    // cmd_velのPublish
    cmd_vel_pub = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);

    // odometoryのSubscribe
    odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
      "odom", 10, std::bind(&MPCNode::odometry_callback, this, _1));

    // obstacleのSubscrie
    local_obstacle_subscriber = this->create_subscription<visualization_msgs::msg::MarkerArray>(
      "local_obstacle_markers", 10,
      std::bind(&MPCNode::local_obstacle_callback, this, std::placeholders::_1));

    // MPCの実行タイマー
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(dt * 1000)), std::bind(&MPCNode::mpc_step, this));

    // 静的な変換を送信するタイマー
    // static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    // send_static_transform();

    // 時刻の初期化
    current_time = this->get_clock()->now();
    last_time = this->get_clock()->now();

    // 座標変換とパス出力
    //publish_transform_and_path(xTrue);
  }

private:
  void mpc_step()
  {
    // MPC制御の解
    casadi::DM uopt = solve_mpc(xTrue);

    // 状態の更新
    //xTrue = update_state(xTrue, uopt);

    // 座標変換とパス出力
    //publish_transform_and_path(xTrue);

    // MPC予測軌道を出力
    publish_predicted_path();

    // 結果の出力
    RCLCPP_INFO_STREAM(this->get_logger(), "Optimal control: " << uopt);
    RCLCPP_INFO_STREAM(this->get_logger(), "Updated state: " << xTrue);

    // uoptをgeometry_msgs::Twistに変換してパブリッシュ
    geometry_msgs::msg::Twist cmd_vel_msg;
    cmd_vel_msg.linear.x = uopt(0).scalar();  // 速度[m/s]
    cmd_vel_msg.angular.z = uopt(1).scalar(); // 角速度[rad/s]
    cmd_vel_pub->publish(cmd_vel_msg);
  }

  DM solve_mpc(DM xTrue)
  {
    // 最適化問題と最適化変数を定義
    casadi::Opti opti;
    casadi::MX x = opti.variable(nx, N + 1);
    casadi::MX u = opti.variable(nu, N);
    // スラック変数の定義（障害物の数 × 時間ステップ数）
    int num_obstacles = obs_pos_list.size();
    casadi::MX delta = opti.variable(num_obstacles, N);

    // 目的関数を定義
    casadi::MX cost = 0;
    for (int i = 0; i < N; ++i) {
      cost += mtimes(mtimes((x(Slice(), i) - xTarget).T(), Q), (x(Slice(), i) - xTarget)) +
        mtimes(mtimes(u(Slice(), i).T(), R), u(Slice(), i));
    }
    cost += mtimes(mtimes((x(Slice(), N) - xTarget).T(), P), (x(Slice(), N) - xTarget));

    // ダイナミクス制約条件を定義
    opti.subject_to(x(Slice(), 0) == xTrue);
    for (int i = 0; i < N; ++i) {
      casadi::MX Ai = casadi::MX::eye(nx);
      casadi::MX Bi = casadi::MX::vertcat(
      {
        casadi::MX::horzcat({dt * cos(x(2, i)), 0.0}),
        casadi::MX::horzcat({dt * sin(x(2, i)), 0.0}),
        casadi::MX::horzcat({0.0, dt})
      });

      opti.subject_to(x(Slice(), i + 1) == mtimes(Ai, x(Slice(), i)) + mtimes(Bi, u(Slice(), i)));
      opti.subject_to(umin <= u(i));
      opti.subject_to(u(i) <= umax);
    }

    // 障害物制約条件を定義
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < num_obstacles; ++j) {
        const auto & obs_pos = obs_pos_list[j];
        casadi::MX b = mtimes((x(Slice(0, 2), i) - obs_pos).T(), (x(Slice(0, 2), i) - obs_pos)) - obs_r * obs_r;
        casadi::MX b_next = mtimes((x(Slice(0, 2), i + 1) - obs_pos).T(), (x(Slice(0, 2), i + 1) - obs_pos)) - obs_r * obs_r;
        opti.subject_to(b_next - b + gamma * b >= 0);
      }
    }

    // 最適化問題を解く
    opti.minimize(cost);
    opti.solver("ipopt");
    casadi::OptiSol sol = opti.solve();

    // MPC予測軌道の値を保存
    mpc_predicted_path_val = sol.value(x);

    // 結果を出力
    casadi::DM xopt = sol.value(x);
    casadi::DM uopt = sol.value(u(Slice(), 0));
    casadi::DM Jopt = sol.value(cost);

    RCLCPP_INFO_STREAM(this->get_logger(), "xopt: " << sol.value(x));
    RCLCPP_INFO_STREAM(this->get_logger(), "uopt: " << sol.value(u));
    RCLCPP_INFO_STREAM(this->get_logger(), "Jopt: " << sol.value(cost));

    return uopt;
  }

  casadi::DM update_state(const casadi::DM & xTrue, const casadi::DM & uopt)
  {
    casadi::DM A = casadi::DM::eye(nx);
    casadi::DM B = casadi::DM(
      {{dt * cos(xTrue(2).scalar()), 0.0},
       {dt * sin(xTrue(2).scalar()), 0.0},
       {0.0, dt}});
    casadi::DM xUpdated = mtimes(A, xTrue) + mtimes(B, uopt);
    return xUpdated;
  }

  void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // オドメトリからx, y, thetaを取得
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;

    tf2::Quaternion quat;
    tf2::fromMsg(msg->pose.pose.orientation, quat);
    tf2::Matrix3x3 mat(quat);
    double roll, pitch, yaw;
    mat.getRPY(roll, pitch, yaw);

    // xTrueに値を代入
    xTrue = casadi::DM({x, y, yaw});
  }

  void local_obstacle_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
  {
    obs_pos_list.clear();  // Clear the list before adding new positions

    for (const auto &marker : msg->markers) {
      if (marker.type == visualization_msgs::msg::Marker::CYLINDER) {
        casadi::DM pos = casadi::DM({marker.pose.position.x, marker.pose.position.y});
        obs_pos_list.push_back(pos);
      }
    }

    // Debug output
    for (const auto &pos : obs_pos_list) {
      RCLCPP_INFO(this->get_logger(), "Obstacle position: (%.2f, %.2f)", pos(0).scalar(), pos(1).scalar());
    }
  }

  void publish_predicted_path()
  {
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->get_clock()->now();
    path_msg.header.frame_id = "map";  // 適切なframe_idに設定

    // 予測軌道をクリア
    path_msg.poses.clear();

    for (int i = 0; i < N; ++i) {
      geometry_msgs::msg::PoseStamped pose_stamped;
      pose_stamped.header = path_msg.header;
      pose_stamped.pose.position.x = mpc_predicted_path_val(0, i).scalar();
      pose_stamped.pose.position.y = mpc_predicted_path_val(1, i).scalar();
      pose_stamped.pose.orientation.w = 1.0;  // 回転なし

      path_msg.poses.push_back(pose_stamped);
    }

    mpc_predicted_path_pub->publish(path_msg);
  }

  // void publish_transform_and_path(const casadi::DM & xTrue)
  // {
  //   // 結果を格納
  //   x = xTrue(0).scalar();
  //   y = xTrue(1).scalar();
  //   th = xTrue(2).scalar();

  //   // 座標変換（odom to base_link）
  //   tf2::Quaternion odom_quat;
  //   odom_quat.setRPY(0, 0, th);  // ロール、ピッチ、ヨーをセット
  //   geometry_msgs::msg::Quaternion odom_quat_msg =
  //     tf2::toMsg(odom_quat);    // tf2::Quaternionからgeometry_msgs::msg::Quaternionに変換

  //   geometry_msgs::msg::TransformStamped odom_trans;
  //   odom_trans.header.stamp = current_time;
  //   odom_trans.header.frame_id = "odom";
  //   odom_trans.child_frame_id = "base_link";

  //   odom_trans.transform.translation.x = x;
  //   odom_trans.transform.translation.y = y;
  //   odom_trans.transform.translation.z = 0.0;
  //   odom_trans.transform.rotation = odom_quat_msg;

  //   odom_broadcaster->sendTransform(odom_trans);

  //   // パスを格納
  //   geometry_msgs::msg::PoseStamped this_pose_stamped;
  //   this_pose_stamped.pose.position.x = x;
  //   this_pose_stamped.pose.position.y = y;
  //   this_pose_stamped.pose.orientation = odom_quat_msg;
  //   this_pose_stamped.header.stamp = current_time;
  //   this_pose_stamped.header.frame_id = "odom";
  //   path.header.frame_id = "odom";
  //   path.poses.push_back(this_pose_stamped);

  //   // パスを公開
  //   path_pub->publish(path);

  //   // 時刻を更新
  //   last_time = current_time;
  // }

  // void send_static_transform()
  // {
  //   geometry_msgs::msg::TransformStamped static_transform_stamped;
  //   static_transform_stamped.header.stamp = this->get_clock()->now();
  //   static_transform_stamped.header.frame_id = "map";
  //   static_transform_stamped.child_frame_id = "odom";
  //   static_transform_stamped.transform.translation.x = 0.0;
  //   static_transform_stamped.transform.translation.y = 0.0;
  //   static_transform_stamped.transform.translation.z = 0.0;
  //   static_transform_stamped.transform.rotation.x = 0.0;
  //   static_transform_stamped.transform.rotation.y = 0.0;
  //   static_transform_stamped.transform.rotation.z = 0.0;
  //   static_transform_stamped.transform.rotation.w = 1.0;
  //   static_broadcaster_->sendTransform(static_transform_stamped);
  // }

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mpc_predicted_path_pub;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr local_obstacle_subscriber;
  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> odom_broadcaster;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
  nav_msgs::msg::Path mpc_predicted_path;
  rclcpp::Time current_time, last_time;
  double dt;
  int nx, nu, N;
  casadi::DM A, B, Q, R, P, xTrue;
  casadi::DM umin, umax;
  casadi::DM xTarget;
  casadi::DM vehicle_diameter, obs_diameter, obs_r, p;
  casadi::DM mpc_predicted_path_val;
  std::vector<casadi::DM> obs_pos_list;
  double x, y, th;
  double gamma;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPCNode>());
  rclcpp::shutdown();
  return 0;
}
