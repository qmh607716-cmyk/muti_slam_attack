#!/usr/bin/env python3
"""
BimodalSpoofingSimulation.py
=============================

LVI-SAM 双模态（LiDAR + Visual）脆弱性分析 ROS 节点。
ApproximateTimeSynchronizer 同步 LiDAR 帧与相机帧。

数据流：
  /points_raw (LiDAR ~10Hz)  ──┐
                                 ├── ApproximateTimeSync ──► 同步回调 ──► CSV
  /camera/image_raw/compressed ──┘
                                   ├── ApproximateTimeSync ──► 同步回调 ──► CSV
  /camera/image_raw/compressed ──┘
"""

import datetime
import math
import os
import sys
import threading

import rospy
import numpy as np
import cv2
import message_filters
from sensor_msgs.msg import PointCloud2, CompressedImage, Image as ImageMsg
from nav_msgs.msg import Odometry

# Add the scripts/ directory to sys.path so that:
#   import functions.calc_bimodal_smvs   → works via <pkg>/lib/<pkg>/functions/
#   import registration_separated         → works directly in scripts/
# Both the source tree (src/slamspoof/scripts/) and the installed tree
# (devel_catkin_tools/lib/slamspoof_icra/) have these files in the same dir.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import functions.calc_bimodal_smvs as cb
import registration_separated


# ---------------------------------------------------------------------------
# 从 LiDAR 点云估算 depth_ratio
# ---------------------------------------------------------------------------

def _compute_depth_ratio_from_points(xyz: np.ndarray, robot_yaw: float) -> float:
    """
    估算"落在当前帧相机 FOV 内的 LiDAR 点比例"作为深度辅助置信度。

    原理：LVI-SAM 中，若 LiDAR 点能被投影到相机图像上，则该区域有 LiDAR depth 辅助。
    我们近似用：相机 FOV 内的 LiDAR 点越多 → 视觉越能依赖深度 → depth_ratio 越高。

    Args:
        xyz:         shape (N, 3), LiDAR 点（机器人坐标系）
        robot_yaw:   retained for API compatibility. The point cloud is already
                     in the LiDAR local frame, so yaw must not rotate the FOV.

    Returns:
        float: depth_ratio ∈ [0, 1]
    """
    if xyz is None or len(xyz) == 0:
        return 0.0

    # 相机水平 FOV（LVI-SAM params_camera.yaml）
    CAM_FX = 669.894
    CAM_U0 = 377.946
    CAM_W  = 720.0
    fov_h = 2.0 * math.atan(CAM_W / 2.0 / CAM_FX)  # ≈ 1.07 rad

    # 过滤掉机器人正下方（地面）和过远的点
    r = np.linalg.norm(xyz[:, :2], axis=1)
    valid = (r > 0.3) & (r < 50.0)  # 0.3m ~ 50m
    if not valid.any():
        return 0.0

    pts = xyz[valid]
    r   = r[valid]

    # LiDAR 局部角（度数 → 弧度）
    theta = np.arctan2(pts[:, 1], pts[:, 0])  # shape (N,)

    # 点云已经在 LiDAR 局部坐标系内；相机 FOV 也必须在同一局部坐标系
    # 中判断。这里不能加 robot_yaw，否则会把局部点云按世界航向误旋转。
    lidar_to_cam_yaw = -0.04  # rad
    cam_lidar = lidar_to_cam_yaw

    # 相机系角（以光轴为中心）
    theta_cam = theta - cam_lidar
    theta_cam = np.arctan2(np.sin(theta_cam), np.cos(theta_cam))

    # 在 FOV 内（±FOV/2）
    half_fov = fov_h / 2.0
    in_fov = np.abs(theta_cam) <= half_fov

    if not in_fov.any():
        return 0.0

    # 加权：近处点权重更高（更可靠的深度）
    r_in_fov = r[in_fov]
    weights  = 1.0 / (r_in_fov + 0.1)
    weighted_in_fov = np.sum(weights)

    r_all     = r
    weights_all = 1.0 / (r_all + 0.1)
    weighted_total = np.sum(weights_all)

    depth_ratio = float(weighted_in_fov / weighted_total) if weighted_total > 0 else 0.0
    return float(np.clip(depth_ratio, 0.0, 1.0))


# ---------------------------------------------------------------------------
# OpenCV 图像质量分析
# ---------------------------------------------------------------------------

def compute_image_quality(cv_img: np.ndarray):
    """
    从单帧相机图像计算视觉质量指标（无需 VINS 跟踪）。

    指标：
      - kp_list:      ORB 关键点列表（用于改进版 V-Vul 分桶）
      - feature_count: ORB 特征点数（用于旧版 V-Vul）
      - sharpness:     拉普拉斯方差（越高 → 越清晰）
      - contrast:      对比度（越高 → 视觉越可靠）

    返回 dict：
        kp_list:       cv2.KeyPoint 列表
        feature_count: int
        sharpness:     float
        contrast:      float
        gray:          灰度图（用于光流计算）
    """
    if cv_img is None or cv_img.size == 0:
        return {"kp_list": [], "feature_count": 0, "sharpness": 0.0, "contrast": 0.0, "gray": None}

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img

    # ORB 特征点
    orb = cv2.ORB_create(nfeatures=200)
    kp  = orb.detect(gray, None)
    feature_count = len(kp)

    # 清晰度：拉普拉斯方差
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 对比度
    contrast = float(gray.std())

    return {"kp_list": kp, "feature_count": feature_count, "sharpness": sharpness, "contrast": contrast, "gray": gray}


# ---------------------------------------------------------------------------
# ROS Image → numpy (supports both CompressedImage and raw Image)
# ---------------------------------------------------------------------------

def img_to_cv(msg):
    """
    Convert either sensor_msgs/CompressedImage or sensor_msgs/Image to cv2 BGR.
    """
    if isinstance(msg, CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    elif isinstance(msg, ImageMsg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        if msg.encoding in ("rgb8",):
            img = np_arr.reshape(msg.height, msg.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif msg.encoding in ("bgr8",):
            img = np_arr.reshape(msg.height, msg.width, 3)
        elif msg.encoding in ("mono8",):
            gray = np_arr.reshape(msg.height, msg.width)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif msg.encoding in ("mono16",):
            gray = np_arr.reshape(msg.height, msg.width)
            gray = cv2.convertScaleAbs(gray)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif msg.encoding in ("compressed",):
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        else:
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
        return img
    return None


# ---------------------------------------------------------------------------
# BimodalNode
# ---------------------------------------------------------------------------

class BimodalNode:
    def __init__(self):
        # ---- 参数 ----
        self.smvs_dir = rospy.get_param(
            "smvs_save_dir",
            "/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/smvs/"
        )
        self.vul_dir = rospy.get_param(
            "vulnerablity_save_dir",
            "/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/vul/"
        )

        lidar_topic = rospy.get_param("subscribe_topic_name", "/points_raw")
        odom_topic  = rospy.get_param(
            "subscribe_odometry_name", "/lvi_sam/lidar/mapping/odometry"
        )
        self.cam_topic = rospy.get_param(
            "camera_topic", "/camera/image_raw/compressed"
        )

        self.topic_len     = int(rospy.get_param("lidar_topic_length", 22))
        self.num_iteration = int(rospy.get_param("number_of_iterations", 1))
        self.sample_rate   = float(rospy.get_param("sample_rate", 0.2))
        self.scale_trans   = float(rospy.get_param("noise_variance", 0.1))

        import os
        os.makedirs(self.smvs_dir, exist_ok=True)
        os.makedirs(self.vul_dir, exist_ok=True)

        self.lidar_start_time = None

        # Image transport: supports "compressed" (CompressedImage) and "raw" (Image)
        # Default to "raw" for KITTI which uses raw sensor_msgs/Image
        self._img_transport = rospy.get_param("camera_transport", "raw")
        rospy.loginfo(f"[BimodalNode] Camera transport: {self._img_transport}")

        # ---- odometry ----
        self.X   = self.Y   = self.Z   = None
        self.yaw = 0.0

        # ---- 日志计数器 ----
        self._log_lock    = threading.Lock()
        self._log_counter = 0
        self._frame_count = 0

        # ---- ApproximateTimeSynchronizer：LiDAR ~10Hz，Camera ~30Hz ----
        # slop=0.2s：容忍两帧最大时间差 200ms（相机最大帧间 33ms，完全安全）
        sub_lidar = message_filters.Subscriber(lidar_topic, PointCloud2)

        if self._img_transport == "compressed":
            sub_cam = message_filters.Subscriber(self.cam_topic, CompressedImage)
        else:
            from sensor_msgs.msg import Image as ImageMsg
            sub_cam = message_filters.Subscriber(self.cam_topic, ImageMsg)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_lidar, sub_cam], queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self._sync_callback)

        # Odometry 单独订阅（不需要同步）
        self._odom_sub = rospy.Subscriber(
            odom_topic, Odometry, self._odom_callback
        )

        # ---- VINS 特征 topic fallback ----
        # 如果 bag 里跑过 LVI-SAM，会有 /lvi_sam/feature/points（VINS 跟踪的特征点）
        # 否则退回到 ORB + 图像质量近似
        self._vins_feature_topic = rospy.get_param(
            "vins_feature_topic", "/lvi_sam/feature/points"
        )
        self._use_vins = rospy.get_param("use_vins_feature", False)
        if self._use_vins:
            try:
                self._vins_sub = rospy.Subscriber(
                    self._vins_feature_topic, PointCloud2, self._vins_callback
                )
                rospy.loginfo(f"[BimodalNode] VINS feature topic: {self._vins_feature_topic} (enabled)")
            except Exception:
                rospy.logwarn("[BimodalNode] VINS feature topic not available, falling back to ORB")
                self._use_vins = False
        else:
            rospy.loginfo("[BimodalNode] Using ORB approximation (use_vins_feature=false)")

        # ---- V-Vul EMA 滤波器 ----
        self._vvul_ema = cb.VVulEMAFilter(alpha=0.60)

        # 前帧数据（用于光流计算和特征匹配）
        self._prev_kp_list = []
        self._prev_gray    = None

        # VINS 特征缓存
        self._vins_cache = {
            "last_track_num": 0,
            "avg_parallax": 0.0,
            "depth_ratio": 0.0,
        }

        self._init_csv()

        rospy.loginfo(f"[BimodalNode] LiDAR topic : {lidar_topic}")
        rospy.loginfo(f"[BimodalNode] Camera topic: {self.cam_topic}")
        rospy.loginfo(f"[BimodalNode] Odom  topic : {odom_topic}")
        rospy.loginfo(f"[BimodalNode] SMVS  dir   : {self.smvs_dir}")
        rospy.loginfo(f"[BimodalNode] Vul   dir   : {self.vul_dir}")
        rospy.loginfo(f"[BimodalNode] Sync  slop  : 0.2s")

    # -------------------------------------------------------------------------
    def _init_csv(self):
        now    = datetime.datetime.now()
        ts_str = now.strftime("%m_%d_%H_%M_%S")

        self._smvs_fname = f"{self.smvs_dir}/{ts_str}.csv"
        self._vul_fname  = f"{self.vul_dir}/vul_{ts_str}.csv"

        import csv

        # SMVS 主文件
        smvs_header = [
            "timestamp", "x", "y", "z", "yaw",
            "frame_bi_smvs", "frame_l_smvs", "frame_v_smvs",
            "feature_count", "sharpness", "contrast", "depth_ratio",
            "v_vul_max", "v_vul_mean",
            "vul_angle_deg", "vec_x", "vec_y",
        ]
        with open(self._smvs_fname, "w", newline="") as f:
            csv.writer(f).writerow(smvs_header)

        # 脆弱性详细文件（72 桶）
        vul_header = [
            "timestamp", "x", "y", "z", "yaw",
            "vul_angle_deg", "vec_x", "vec_y", "frame_bi_smvs",
        ]
        for i in range(72):
            vul_header.append(f"l_vul_{i:02d}")
        for i in range(72):
            vul_header.append(f"v_vul_{i:02d}")
        for i in range(72):
            vul_header.append(f"bi_vul_{i:02d}")

        with open(self._vul_fname, "w", newline="") as f:
            csv.writer(f).writerow(vul_header)

        rospy.loginfo(f"[BimodalNode] SMVS CSV: {self._smvs_fname}")
        rospy.loginfo(f"[BimodalNode] Vul   CSV: {self._vul_fname}")

    # -------------------------------------------------------------------------
    def _odom_callback(self, msg: Odometry):
        self.X = msg.pose.pose.position.x
        self.Y = msg.pose.pose.position.y
        self.Z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    # -------------------------------------------------------------------------
    def _vins_callback(self, feature_msg):
        """订阅 VINS 特征点 topic，解析跟踪质量（optical flow 视差、深度比例）。"""
        import functions.calc_bimodal_smvs as cb
        info = cb.compute_visual_vul_from_feature_msg(feature_msg)
        self._vins_cache["last_track_num"] = info.get("last_track_num", 0)
        self._vins_cache["avg_parallax"]    = info.get("avg_parallax", 0.0)
        self._vins_cache["depth_ratio"]     = info.get("depth_ratio", 0.0)

    # -------------------------------------------------------------------------
    # 同步回调：LiDAR 帧和相机帧时间差 ≤ 0.2s 时触发
    # -------------------------------------------------------------------------
    def _sync_callback(self, lidar_msg: PointCloud2, cam_msg):
        import csv

        # ---- 时间戳 ----
        bag_time   = float(lidar_msg.header.stamp.to_sec())
        if self.lidar_start_time is None:
            self.lidar_start_time = bag_time
        time_stamp = bag_time - self.lidar_start_time

        # ---- 解析点云 ----
        iteration = int(len(lidar_msg.data) / self.topic_len)
        bin_data  = np.frombuffer(lidar_msg.data, dtype=np.uint8)
        bin_pts   = bin_data.reshape(iteration, self.topic_len)

        x   = bin_pts[:, 0:4].view(np.float32)[:, 0]
        y   = bin_pts[:, 4:8].view(np.float32)[:, 0]
        z   = bin_pts[:, 8:12].view(np.float32)[:, 0]
        xyz = np.hstack([x[:, None], y[:, None], z[:, None]])

        # ---- 相机质量（已与 LiDAR 帧同步）----
        cv_img = img_to_cv(cam_msg)
        q_info = compute_image_quality(cv_img)
        kp_list      = q_info["kp_list"]
        feature_count = q_info["feature_count"]
        sharpness     = q_info["sharpness"]
        contrast      = q_info["contrast"]
        gray          = q_info["gray"]

        # ---- 从 LiDAR 点云计算 depth_ratio（相机 FOV 内加权点数比例）----
        depth_ratio = _compute_depth_ratio_from_points(xyz, self.yaw)
        if depth_ratio is None:
            depth_ratio = 0.0

        # ---- VINS fallback：优先用 VINS 特征跟踪数据 ----
        if self._use_vins:
            feature_count_vul = self._vins_cache.get("last_track_num", 0)
            parallax_for_vul  = self._vins_cache.get("avg_parallax", 0.0)
            depth_ratio_vul   = self._vins_cache.get("depth_ratio", 0.0)
        else:
            feature_count_vul = feature_count
            parallax_for_vul  = sharpness
            depth_ratio_vul   = depth_ratio

        # ---- G-ICP Hessian 分析（LiDAR 模态）----
        coord, dot_eig = registration_separated.execute_gicp(
            xyz, self.num_iteration, self.sample_rate, self.scale_trans
        )

        # ---- LiDAR 脆弱性 L-Vul ----
        l_vul = cb.compute_lidar_vul_from_hessian(coord, dot_eig, step_deg=5.0)
        l_vul_max = float(l_vul.max()) if len(l_vul) > 0 else 1.0

        # ---- 视觉补偿能力 V-Vul（分桶 Q + 光流一致性 + 时序 EMA）----
        lidar_in_fov_count = int(depth_ratio_vul * 10000)
        v_vul, v_info = cb.compute_visual_vul(
            robot_yaw=self.yaw,
            kp_list=kp_list,
            prev_kp_list=self._prev_kp_list,
            prev_gray=self._prev_gray,
            curr_gray=gray,
            lidar_xyz_in_cam_fov=lidar_in_fov_count,
        )
        # 时序 EMA 平滑
        v_vul = self._vvul_ema.update(v_vul)
        # 保存前帧数据
        self._prev_kp_list = kp_list
        self._prev_gray    = gray

        # ---- 双模态融合 Bi-Vul ----
        bi_vul = cb.fuse_bimodal(l_vul, v_vul, l_vul_max=l_vul_max)

        # ---- 帧级 SMVS ----
        frame_bi_smvs = cb.frame_smvs(bi_vul, step_deg=5.0)
        frame_l_smvs  = cb.frame_lidar_smvs(l_vul, step_deg=5.0)
        frame_v_smvs  = cb.frame_visual_smvs(v_vul, step_deg=5.0)

        # ---- 最脆弱方向 ----
        vul_angle = cb.vulnerable_direction(bi_vul, step_deg=5.0)
        vec_x, vec_y = cb.vec_from_angle(vul_angle)

        # ---- 写入 SMVS CSV ----
        smvs_row = [
            f"{time_stamp:.3f}",
            f"{self.X or 0.0:.6f}",
            f"{self.Y or 0.0:.6f}",
            f"{self.Z or 0.0:.6f}",
            f"{self.yaw:.6f}",
            f"{frame_bi_smvs:.6f}",
            f"{frame_l_smvs:.6f}",
            f"{frame_v_smvs:.6f}",
            str(feature_count),
            f"{sharpness:.4f}",
            f"{contrast:.4f}",
            f"{depth_ratio:.4f}",
            f"{v_vul.max():.6f}",
            f"{v_vul.mean():.6f}",
            f"{vul_angle:.2f}",
            f"{vec_x:.6f}",
            f"{vec_y:.6f}",
        ]
        with open(self._smvs_fname, "a", newline="") as f:
            csv.writer(f).writerow(smvs_row)

        # ---- 写入 Vul CSV ----
        vul_row = [
            f"{time_stamp:.3f}",
            f"{self.X or 0.0:.6f}",
            f"{self.Y or 0.0:.6f}",
            f"{self.Z or 0.0:.6f}",
            f"{self.yaw:.6f}",
            f"{vul_angle:.2f}",
            f"{vec_x:.6f}",
            f"{vec_y:.6f}",
            f"{frame_bi_smvs:.6f}",
        ]
        vul_row.extend([f"{v:.8f}" for v in l_vul])
        vul_row.extend([f"{v:.8f}" for v in v_vul])
        vul_row.extend([f"{v:.8f}" for v in bi_vul])

        with open(self._vul_fname, "a", newline="") as f:
            csv.writer(f).writerow(vul_row)

        # ---- 日志（每 10 帧）----
        with self._log_lock:
            self._log_counter += 1
            self._frame_count += 1
            if self._log_counter % 10 == 0:
                rospy.loginfo(
                f"[{time_stamp:.1f}s] L-SMVS={frame_l_smvs:.1f}  "
                f"V-SMVS={frame_v_smvs:.3f}  Bi-SMVS={frame_bi_smvs:.1f}  "
                f"feat={feature_count}  sharp={sharpness:.1f}  "
                f"v_vul_max={v_vul.max():.4f}"
            )


def main():
    rospy.set_param("use_sim_time", True)
    rospy.init_node("bimodal_smvs_node", anonymous=True)
    node = BimodalNode()
    rospy.loginfo("[BimodalNode] Node started. Waiting for messages...")
    rospy.spin()


if __name__ == "__main__":
    main()
