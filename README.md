# SLAMSpoof for LVI-SAM: 双模态 LiDAR 欺骗攻击框架

## 目录

- [概述](#概述)
- [原版 SLAMSpoof (ICRA 2025) 与本工作的区别](#原版-slamspoof-icra-2025-与本工作的区别)
- [攻击手段总览](#攻击手段总览)
- [核心概念：原版假墙 vs 集中几何](#核心概念原版假墙-vs-集中几何)
- [项目结构](#项目结构)
- [LVI-SAM 数据集与 Topic 规格](#lvi-sam-数据集与-topic-规格)
- [完整实验流程](#完整实验流程)
- [配置参数详解](#配置参数详解)
- [关键实现细节](#关键实现细节)
- [实验结果](#实验结果)
- [引用](#引用)

---

## 概述

本项目将 **SLAMSpoof** (ICRA 2025) LiDAR 欺骗攻击框架移植到 **LVI-SAM** (TixiaoShan et al., ICRA 2021)，并进行了多维度扩展。LVI-SAM 是一个紧耦合的 LiDAR-视觉-惯性 SLAM 系统，比原版 SLAMSpoof 评估的 A-LOAM / KISS-ICP 等纯 LiDAR SLAM 具有更复杂的多传感器融合结构。

本工作的核心贡献：

1. **LVI-SAM 全传感器兼容**：攻击编辑器保留全部原始 Topic（LiDAR、Camera、IMU、GPS），仅修改 `/points_raw` 的点云数据。
2. **点云格式完整保留**：使用 22 字节完整 Velodyne 布局 (`x, y, z, intensity, ring, time`)，而非原版 xyz-only 格式，确保 LVI-SAM 的 scan registration 不崩溃。
3. **双模态脆弱性分析**：将视觉模态的补偿能力纳入 SMVS，提出 Bi-Vul 融合公式。
4. **yaw 感知攻击方向**：在 LiDAR 局部坐标系下计算攻击窗口，与机器人世界朝向解耦。
5. **集中几何攻击 (D-SLAMSpoof)**：提出极坐标菱形/墙角约束几何，使伪造点在 scan matching 中产生方向性更强的几何约束。
6. **振荡注入 (D-SLAMSpoof)**：动态墙的振荡周期由 SLAM 对应距离阈值 $M_{corr}$ 自动推导（公式 4），保证最快振荡频率不被 outlier filtering 拒绝。

---

## 原版 SLAMSpoof (ICRA 2025) 与本工作的区别

| 维度 | 原版 SLAMSpoof | 本工作 |
|---|---|---|
| **目标 SLAM** | A-LOAM / KISS-ICP（纯 LiDAR） | LVI-SAM（LiDAR-视觉-惯性紧耦合） |
| **保留 Topic** | 仅 LiDAR + IMU | 全部原始 Topic（LiDAR、Camera、IMU、GPS） |
| **点云格式** | 重建为 xyz-only（`point_step=12`），破坏 ring/time 字段 | 保留完整 22 字节 Velodyne 布局 |
| **攻击方向坐标系** | 世界坐标系，无 yaw 校正 | LiDAR 局部坐标系（减去机器人 yaw） |
| **SMVS 模态** | 仅 LiDAR（GICP Hessian 特征值） | LiDAR + 视觉双模态融合（Bi-Vul） |
| **攻击几何** | 随机均匀分布平面墙 | 4 种几何模型（随机平面、束投影、菱形集中、墙角集中） |
| **动态墙振荡周期** | 用户手动指定 | 由 $M_{corr}$ 约束自动推导最优周期 |
| **Spoofer 选择** | 论文忠实实现（射线交点 + 轨迹拟合） | 支持论文忠实版 + 双模态版 |
| **输出点云字段** | `x, y, z`（3字段） | `x, y, z, intensity, ring, time`（6字段） |

---

## 攻击手段总览

本工作共实现 **6 种攻击模式**，分为 3 大类：

### 一、HFR 攻击（removal）

删除攻击窗口内的真实点，替换为随机噪声点。

```
原始帧点云 → 删除[center±range/2]内所有点 → 注入随机噪声点
```

- **噪声空间分布**：在攻击窗口内均匀随机采样角度和距离（1~50m）
- **垂直通道**：从 LiDAR 固定垂直角集合（VLP-16: 16个通道，VLP-32C: 32个通道）中随机选择
- **time 字段**：基于扫描相位 `azimuth / 360 × scan_period`
- **ring 字段**：从 z/r 反推最近通道

### 二、假墙注入（static）

删除攻击窗口内的真实点，注入伪造的平面墙。

**4 种几何模型**（由 `static_geometry_model` 参数选择）：

#### 1. `original_random`（原版随机平面墙）

```python
theta = uniform(center - range/2, center + range/2)  # 均匀随机角度
r = wall_distance  # 固定距离
z = r * sin(vertical_angle)  # 垂直角从固定集合随机
```

特点：伪造点在角度方向上均匀散布，形成**均匀分布的平面**。几何约束分散在攻击窗口的各个方向上。

#### 2. `beam_project`（束投影墙）

```python
# 继承被删除点的 ring 和 time 字段
x_fake = (x0 / |p0|) * wall_distance
y_fake = (y0 / |p0|) * wall_distance
z_fake = (z0 / |p0|) * wall_distance
ring_fake = ring0
time_fake = time0
```

特点：保持 LiDAR 扫描线的拓扑结构不变（每条 scan line 的 ring 和时间戳信息得到保留），是**结构最真实的**注入方式。

#### 3. `square`（菱形集中几何）

```python
# D-SLAMSpoof 极坐标方程
theta_prime = theta - rotate_rad
d_fake = S / (|sin(theta_prime)| + |cos(theta_prime)|)
```

通过旋转角 `rotate_rad` 控制形状：
- `rotate=0` → L 形墙角（相邻两条边朝向 LiDAR）
- `rotate=π/4` → 平面墙（相对两条边垂直于径向）

特点：**几何约束集中在边缘方向**，而非均匀分布。当边缘垂直于 LiDAR 径向方向时，scan matching 受到最强约束，产生持续定向漂移。

#### 4. `corner`（L 形墙角）

```python
# square + rotate=0 的别名
rotate_rad = 0.0
```

特点：两条相邻边缘面向 LiDAR，形成 **L 形墙角约束**。

### 三、动墙注入（dynamic）

墙距离随时间周期性变化，在 `wall_distance_min` 和 `wall_distance_max` 之间线性扫描。

```python
frac = (timestamp % t_cycle) / t_cycle
wall_dist = (wall_dist_max - wall_dist_min) * frac + wall_dist_min
```

**最优振荡周期自动推导**（D-SLAMSpoof 公式 4）：

```python
t_cycle = (d_max - d_min) / M_corr * Δt
```

其中：
- $M_{corr}$：SLAM 的最大对应距离阈值（LVI-SAM 保守值约 1.0m）
- $\Delta t$：LiDAR 扫描周期（VLP-16 = 0.1s）

这是**最快且不被 outlier filtering 拒绝**的振荡频率。快于此频率 → 伪造点被滤除；慢于此频率 → 单位时间累积漂移减少。

以上 4 种几何模型均可用于 dynamic 模式，通过 `static_geometry_model` 参数选择。

---

## 核心概念：原版假墙 vs 集中几何

### 几何约束的本质差异

SLAM 的 scan matching 通过最小化当前帧点与参考帧/地图点的对应距离误差来估计机器人位姿。当注入伪造点时，伪造点与真实点之间的几何关系决定了 scan matching 的收敛方向。

**原版 `original_random`（均匀平面墙）**：

```
攻击窗口内所有角度均匀注入
→ 伪造点均匀散布在 [d=wall_dist] 的弧线上
→ 几何约束来自弧线上均匀分布的点
→ scan matching 在各方向上受到的"拉力"大致相同
→ 漂移方向取决于 SLAM 实现细节（优化顺序、数值稳定性等）
```

**D-SLAMSpoof `square`（菱形集中几何）**：

```
极坐标方程 d_fake = S/(|sin θ'|+|cos θ'|)
→ 当 θ' = 0° 或 90° 时（边缘方向），d_fake = S
→ 当 θ' = 45° 时（对角线方向），d_fake = S/√2 ≈ 0.707S
→ 边缘方向的距离约束最强，对角线方向最弱
→ 所有最强约束都集中在少数几个方向上
→ scan matching 持续被拉向边缘法线方向
→ 产生可预测的、方向性的持续漂移
```

**关键洞察**：均匀墙的攻击效果依赖 SLAM 的数值误差特性（不可控），而集中几何通过主动设计约束分布，将 scan matching 引导向期望的漂移方向（可控）。

### 四种几何模型的对比

| 模型 | 角度分布 | 距离分布 | ring/time 继承 | 几何约束特点 |
|---|---|---|---|---|
| `original_random` | 均匀随机 | 固定 | 否 | 均匀分散 |
| `beam_project` | 继承原帧 | 固定 | 是（ring+time） | 继承原拓扑 |
| `square` | 菱形方程 | 变化（S/(\|sin\|+\|cos\|)） | 否 | 集中在边缘方向 |
| `corner` | L形方程 | 变化（同上） | 否 | 集中在墙角两侧 |

---

## 项目结构

```
slamspoof/
├── scripts/
│   ├── SpoofingSimulation.py         # 原版 SLAMSpoof ROS 节点（LiDAR-only）
│   ├── BimodalSpoofingSimulation.py  # 双模态 SMVS 分析节点（LVI-SAM）
│   ├── spoofing_editer_lvisam.py    # LVI-SAM Rosbag 攻击编辑器（保留全 Topic）
│   ├── spoofing_editer_imu.py        # 原版 SLAMSpoof 编辑器（仅 LiDAR + IMU）
│   ├── select_spoofer_from_smvs_paper.py  # 论文忠实 spoofer 位置选择
│   ├── select_spoofer_from_bimodal.py      # 双模态 spoofer 位置选择
│   ├── compare_lvisam_trajectory.py  # 轨迹对比与偏差可视化
│   ├── extract_lvisam_odom_csv.py   # 从 /lvi_sam/lidar/mapping/odometry 提取轨迹
│   ├── registration_separated.py      # GICP Hessian 分析（small_gicp 后端）
│   ├── visualize.py                   # SMVS 空间热力图可视化
│   └── functions/
│       ├── spoofing_sim_lvisam.py    # LVI-SAM 攻击函数（6种模式）
│       ├── spoofing_sim.py            # 原版 SLAMSpoof 攻击函数（3种模式）
│       ├── calc_bimodal_smvs.py      # 双模态 SMVS 计算（L-Vul + V-Vul → Bi-Vul）
│       ├── calc_smvs.py               # 原版 SMVS 计算
│       └── bag2array.py               # 二进制点云解析工具
├── launch/
│   ├── run_smvs_lvisam.launch        # 原版 SMVS 分析（LVI-SAM 数据）
│   ├── run_bimodal_smvs_lvisam.launch # 双模态 SMVS 分析
│   ├── rosbag_editer_lvisam.launch   # LVI-SAM 攻击编辑器
│   ├── extract_trajectory_lvisam.launch  # 轨迹提取
│   ├── run_slam_loam.launch          # 原版 SMVS（A-LOAM）
│   └── visualizer.launch             # SMVS 热力图
├── config.json                       # 原版 SLAMSpoof 配置
├── config_lvisam.json                # LVI-SAM 配置
└── config_lvisam_kitti_dynamic_fixed.json  # KITTI 动态墙配置
```

---

## LVI-SAM 数据集与 Topic 规格

### 使用的数据集

本工作测试使用 LVI-SAM 官方 `handheld.bag`（室内手持场景）。

### Topic 规格

| Topic | 类型 | 说明 |
|---|---|---|
| `/points_raw` | `sensor_msgs/PointCloud2` | Velodyne VLP-16，`point_step=22` |
| `/imu_correct` | `sensor_msgs/Imu` | 校正后 IMU |
| `/imu_raw` | `sensor_msgs/Imu` | 原始 IMU |
| `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | 相机图像 |
| `/gps/fix` | `sensor_msgs/NavSatFix` | GPS |
| `/lvi_sam/lidar/mapping/odometry` | `nav_msgs/Odometry` | LVI-SAM 激光里程计输出 |

### Velodyne VLP-16 点云字段布局（22 字节）

```
offset  field       datatype
  0     x           float32
  4     y           float32
  8     z           float32
 12     intensity   float32
 16     ring        uint16
 18     time        float32
```

> **关键**：攻击编辑器**永不重建** PointCloud2。它读取原始字节，按 `point_step=22` 重塑为矩阵，删除/替换 22 字节记录，再写回原始字段布局。注入的伪造点合成完整的 22 字节记录（x, y, z, intensity, ring, time），不破坏字段结构。

---

## 完整实验流程

### Step 1 — 记录原始（无攻击）轨迹

```bash
# Terminal 1
roslaunch lvi_sam run.launch

# Terminal 2
roslaunch slamspoof_icra extract_trajectory_lvisam.launch

# Terminal 3
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/handheld.bag
```

输出 CSV：`~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/{timestamp}_lvisam_traj.csv`

### Step 2 — 计算 SMVS

**方案 A：原版 LiDAR-only SMVS**

```bash
roslaunch slamspoof_icra run_smvs_lvisam.launch
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/handheld.bag
```

**方案 B：双模态 SMVS（LiDAR + Visual）**

```bash
roslaunch slamspoof_icra run_bimodal_smvs_lvisam.launch
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/handheld.bag
```

输出：
```
smvs/{timestamp}.csv     # 帧级 SMVS 分数
vul/vul_{timestamp}.csv  # 72 桶分方向脆弱性详情
```

### Step 3 — 自动选择 Spoofer 位置

**方案 A：原版论文忠实方法**

```bash
python scripts/select_spoofer_from_smvs_paper.py \
    --smvs  smvs/{timestamp}.csv \
    --vul   vul/vul_{timestamp}.csv \
    --top-k 5 \
    --spoof-distance 15.0 \
    --verbose
```

**方案 B：双模态方法（使用 Bi-Vul）**

```bash
python scripts/select_spoofer_from_bimodal.py \
    --smvs  smvs/{timestamp}.csv \
    --vul   vul/vul_{timestamp}.csv \
    --score-column frame_bi_smvs \
    --top-k 5 \
    --verbose
```

两种方案都基于 **射线交点 + 轨迹直线拟合**（SLAMSpoof Section III-C），区别在于方案 B 使用双模态 Bi-Vul 替代原版 L-SMVS 作为帧排序依据。

### Step 4 — 生成攻击 Rosbag

```bash
# 配置 config_lvisam.json 中的 spoofing_mode 和几何参数
roslaunch slamspoof_icra rosbag_editer_lvisam.launch
```

编辑器根据 `spoofing_mode` 选择攻击模式：

```json
{
  "spoofing_mode": "removal",       // HFR 噪声攻击
  "spoofing_mode": "static",        // 假墙注入
  "spoofing_mode": "dynamic"        // 动墙注入
}
```

### Step 5 — 在攻击 Rosbag 上运行 LVI-SAM

```bash
# Terminal 1
roslaunch lvi_sam run.launch

# Terminal 2
roslaunch slamspoof_icra extract_trajectory_lvisam.launch

# Terminal 3
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/handheld_attacked.bag
```

### Step 6 — 轨迹对比与偏差分析

```bash
python scripts/compare_lvisam_trajectory.py \
    --original ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/original_traj.csv \
    --attacked ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/attack_traj.csv \
    --spoofer-x 44.517 \
    --spoofer-y 129.788 \
    --output-prefix attack_vs_original \
    --output-dir ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/
```

输出：XY 轨迹叠加图、偏差随时间变化图、偏差统计摘要（均值、最大值、终值）。

---

## 配置参数详解

### 攻击模式配置

```json
{
  "main": {
    "input_file":    "/path/to/input.bag",
    "output_file":   "/path/to/output.bag",
    "reference_file": "/path/to/original_traj.csv",

    "spoofing_mode": "removal",     // "removal" | "static" | "dynamic"

    "spoofer_x":         44.517,     // 世界坐标系 X
    "spoofer_y":        129.788,     // 世界坐标系 Y
    "distance_threshold":  5.0,       // 触发半径（米）

    "lidar_topic":       "/points_raw",
    "lidar_topic_length": 22,         // Velodyne VLP-16 = 22 字节
    "imu_topic":         "/imu_correct",

    "spoofing_range":    160.0,       // 攻击窗口总角度（度）

    "wall_dist":          15.0,       // static 模式：固定墙距离
    "wall_distance_min":   5.0,       // dynamic 模式：最小距离
    "wall_distance_max":  25.0,       // dynamic 模式：最大距离
    "spoofing_cycle":      2.0,       // dynamic 模式：振荡周期（秒）

    "point_count_model":    "original",   // "original" | "equal_replace" | "pure_removal"
    "static_geometry_model": "square",     // "original_random" | "beam_project" | "square" | "corner" | "planar"
    "wall_intensity":      120.0,         // 注入点强度值

    "square_scale_S":      null,          // 菱形方程常数 S（null = wall_dist × √2）
    "square_rotate_rad":   null,          // 菱形旋转角（null/0 = corner，π/4 = planar）

    "M_corr":              1.0,          // SLAM 最大对应距离（米）
    "lidar_scan_period":   0.1,          // 扫描周期（VLP-16 = 0.1s）
    "auto_cycle":          true,          // 自动推导最优振荡周期

    "rng_seed":            42
  },
  "simulator": {
    "horizontal_resolution": 0.1,    // LiDAR 水平角分辨率（度）
    "vertical_lines":       16.0,    // 垂直扫描线数（VLP-16 = 16）
    "spoofing_rate":        0.3     // HFR 噪声注入率（0.0 ~ 1.0）
  },
  "filtering": {
    "minimum_measuring_distance": 0.0,
    "maximum_measuring_distance": 30.0,
    "minimum_height_threshold": -2.0
  }
}
```

### 参数说明速查表

| 参数 | 说明 | 常用值 |
|---|---|---|
| `spoofing_mode` | 攻击模式 | `removal`/`static`/`dynamic` |
| `spoofer_x/y` | Spoofer 世界坐标 | 从 Step 3 获取 |
| `distance_threshold` | 触发半径 | 5.0 ~ 15.0 m |
| `spoofing_range` | 攻击窗口总角度 | 80° ~ 160° |
| `wall_dist` | 假墙固定距离 | 10.0 ~ 20.0 m |
| `wall_distance_min/max` | 动墙距离范围 | 5.0 ~ 25.0 m |
| `point_count_model` | 注入点数模型 | `original`（按分辨率计算）|
| `static_geometry_model` | 假墙几何模型 | `square`（集中几何）|
| `square_scale_S` | 菱形方程尺度 | null=wall_dist×√2 |
| `square_rotate_rad` | 菱形旋转角 | 0（corner）、π/4（planar）|
| `M_corr` | 对应距离阈值 | 1.0 m（LVI-SAM）|
| `auto_cycle` | 自动最优周期 | true |

---

## 关键实现细节

### 1. yaw 感知攻击方向

攻击方向在 LiDAR 局部坐标系下计算：

```python
world_angle = atan2(spoofer_y - robot_y, spoofer_x - robot_x)
local_angle = world_angle - robot_yaw    # 减去机器人 yaw
local_angle = atan2(sin(local_angle), cos(local_angle))  # 归一化到 [-π, π]
```

无论机器人在世界哪个朝向，移除窗口始终正确对应 spoofer 方向。

### 2. ApproximateTimeSynchronizer 同步

双模态 SMVS 节点使用 `message_filters.ApproximateTimeSynchronizer` 同步 LiDAR (~10Hz) 和 Camera (~30Hz)：

```python
self.ts = message_filters.ApproximateTimeSynchronizer(
    [sub_lidar, sub_cam], queue_size=10, slop=0.2  # 容忍 200ms 时间差
)
```

这确保了每帧 LiDAR 点云与同时间戳的相机图像配对分析。

### 3. G-ICP Hessian 分析

使用 `small_gicp` 库对点云执行 G-ICP：

```python
# 1. 随机采样两个子集
pc1, pc2 = random_sampling(xyz, sample_rate=0.2)

# 2. 加噪声后运行 GICP 对齐
result = small_gicp.align(target, source, tree)

# 3. 提取 Hessian 矩阵 H
hessian = np.asarray(result.H)

# 4. 分解 Hessian → 点级特征值
for i in range(source.size()):
    succ, H, b, e = factor.linearize(...)
    # H 的平移子块特征值 → 局部可定位性指标
```

特征值越小 → 该方向的局部可定位性越差 → 越容易被欺骗。

### 4. 双模态融合（Bi-Vul）

```
L-Vul[k]  = GICP Hessian 特征值分桶（LiDAR 脆弱性）
V-Vul[k]  = γ × cam_coverage[k] × Q[k]    （视觉补偿能力）
Bi-Vul[k] = L-Vul[k] × (1 - V-Vul[k] × L-Vul_norm[k])

Q[k] = w_track × feature_density[k]
     + w_optical × flow_consistency[k]
     + w_depth × depth_quality
     + w_spatial × spatial_dist
     + w_parallax × parallax
```

视觉质量好 → V-Vul 高 → Bi-Vul 低 → 攻击被削弱。

### 5. 帧级 SMVS（HFR / AHFR）

```python
def frame_smvs(vul, d_th=None):
    center_idx = argmax(vul)
    score = 0.0
    for k in range(n_buckets):
        dist = circular_distance(k, center_idx, n)
        if dist <= d_th:
            weight = -dist + d_th
            score += vul[k] * weight
    return score
```

- `d_th=8`（HFR 模式）：攻击范围约 ±40°（80° 总角度）
- `d_th=2`（AHFR 模式）：攻击范围约 ±10°（20° 总角度）

---

## 实验结果

### handheld.bag 全程测试结果

| 攻击方法 | 触发帧数 | 触发率 | 平均偏差 | 最大偏差 |
|---|---|---|---|---|
| 轨迹引导（原版） | 74 / 16,289 | 0.45% | 0.85 m | 3.10 m |
| SMVS 引导（单 LiDAR） | 76 / 16,289 | 0.47% | 1.26 m | 4.29 m |
| 双模态 SMVS 引导（Bi-Vul） | — | — | — | — |

**结论**：在相近的触发率下（约 0.47%），SMVS 引导攻击比纯轨迹引导的最大偏差高出约 1.5 倍，证明基于 scan matching 脆弱性选择攻击位置比仅依赖轨迹几何更为有效。

---

## 引用

```bibtex
@inproceedings{slamspoof2025,
  title={SLAMSpoof: Practical LiDAR Spoofing Attacks on Localization Systems
         Guided by Scan Matching Vulnerability Analysis},
  author={Nagata, Rokuto and Koide, Kenji and Hayakawa, Yuki and
          Suzuki, Ryo and Ikeda, Kazuma and Sako, Ozora and Chen, Qi Alfred and
          Sato, Takami and Yoshioka, Kentaro},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025}
}

@inproceedings{lvisam2021shan,
  title={LVI-SAM: Tightly-coupled Lidar-Visual-Inertial Odometry via
         Smoothing and Mapping},
  author={Shan, Tixiao and Englot, Brendan and Ratti, Carlo and Rus, Daniela},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={5692--5698},
  year={2021}
}
```
