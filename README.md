# SLAMSpoof Attack Framework

## 概述

基于 SLAMSpoof (ICRA 2025) 的 LiDAR 欺骗攻击框架，移植到 **LVI-SAM**（视觉-激光-惯性紧耦合类 SLAM 系统）。

核心流程：**因子图分析 → 脆弱性评估 → Spoofer 位置优化 → 攻击注入 → 轨迹对比**

### 攻击模型

| 模式 | 描述 |
|------|------|
| `removal` | 删除攻击窗口内真实点，注入随机噪声（模拟硬件干扰） |
| `static` | 注入固定距离假墙，强迫 SLAM 估计向欺骗方向偏移 |
| `dynamic` | 假墙距离周期性振荡，利用因子图约束传导放大偏移 |

`static` 模式支持多种几何注入模型：

| 模型 | 来源 | 描述 |
|------|------|------|
| `original_random` | 原版 | 均匀随机角度圆柱墙，几何约束分散 |
| `beam_project` | 本工作 | 沿 scan line 方向投影到固定距离，继承 ring/time |
| `square` | D-SLAMSpoof | 菱形几何，约束集中在边缘方向 |
| `corner` | D-SLAMSpoof | L 形墙角（square + rotate=0），两侧边缘面向 LiDAR |

---

## 快速开始

### 需求

- ROS Noetic + Catkin Tools
- LVI-SAM（`~/catkin_ws/devel_catkin_tools`）
- LIO-SAM（`~/catkin_ws/devel_catkin_tools`）
- `small_gicp`（G-ICP 后端，用于 LVI-SAM）
- 数据集：`~/catkin_ws/src/LVI-SAM/datasets/*.bag`

### 环境准备

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash
```

---

## 完整实验流程

> **代理模型说明**：LIO-SAM 作为 LVI-SAM 的简化代理模型。通过代理模型快速获取因子图结构，用于分析定位约束的连通性和隔离性，指导 Spoofer 位置优化。

### 阶段 0：LIO-SAM 代理模型采集因子图（可选，推荐先行）

> 通过 LIO-SAM 快速获取因子图结构。
> 当前代理配置启用 LiDAR-IMU 约束和 ICP/距离触发的 loop closure，但不包含视觉因子。该代理图用于近似受害系统的图拓扑、回环连接和局部约束密度。


```bash
# ========== 终端 1 ==========
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash
rosparam set use_sim_time true

# 设置 dump 输出目录（与 LVI-SAM 相同路径，共享同一 pipeline）
export LIO_GRAPH_DUMP_DIR=/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/graph_dumps
rm -f $LIO_GRAPH_DUMP_DIR/dump_*.json

roslaunch slamspoof run_lio_sam.launch

# ========== 终端 2（等终端 1 全部节点启动后再播放）==========
source /opt/ros/noetic/setup.bash
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/handheld.bag --clock
```

---

### 阶段 1：录制原始轨迹（基线）

```bash
# ========== 终端 1 ==========
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash
rosparam set use_sim_time true
roslaunch lvi_sam run.launch

# ========== 终端 2 ==========
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash
mkdir -p ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/original
rosbag record -O ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/original/handheld_original_traj.bag \
    /lvi_sam/lidar/mapping/odometry

# ========== 终端 3 ==========
source /opt/ros/noetic/setup.bash
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/handheld.bag --clock --pause
```

bag 播放完毕后，提取轨迹：

```bash
python3 ~/catkin_ws/src/slamspoof/scripts/extract_lvisam_odom_csv.py \
    --bag ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/original/handheld_original_traj.bag \
    --out ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/original/handheld_original_traj.csv
```

---

### 阶段 2：采集双模态 SMVS

```bash
# ========== 终端 1 ==========
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash
rosparam set use_sim_time true
roslaunch lvi_sam run.launch

# ========== 终端 2 ==========
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash
roslaunch slamspoof run_bimodal_smvs_lvisam.launch

# ========== 终端 3 ==========
source /opt/ros/noetic/setup.bash
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/handheld.bag --clock
```

输出文件（由 launch 文件 `smvs_save_dir` / `vulnerablity_save_dir` 参数指定）：
```
slamspoof_handheld/smvs/{timestamp}.csv      # 帧级 SMVS 分数
slamspoof_handheld/vul/vul_{timestamp}.csv    # 分方向脆弱性
```

---

### 阶段 3：选择 Spoofer 位置（SMVS + 因子图引导）

> `--graph-dump-dir` 参数使用阶段 0（LIO-SAM）运行的 dump 目录。

```bash
# 【重要】将路径替换为阶段 2 新生成的 CSV 文件
python3 ~/catkin_ws/src/slamspoof/scripts/select_spoofer_bi_bo.py \
    --smvs ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/smvs/{timestamp}.csv \
    --vul  ~/cat_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/vul/vul_{timestamp}.csv \
    --traj ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/original/handheld_original_traj.csv \
    --graph-dump-dir ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/graph_dumps \
    --top-k 20 \
    --distance-threshold 15.0 \
    --spoofing-range 80.0 \
    --cma-calls 200 \
    --verbose
```

**评分公式**：

```
score(S) = opportunity(S) + α × structural(S)

opportunity(S) = reach(S) × bivul_gate(S)
  reach(S)      : 15m范围内帧的Gaussian加权和 [0,1]
  bivul_gate(S) : 72维脆弱向量的方向匹配程度 [0,1]

structural(S) = structural_blended(S) × coverage(S)
  structural_blended(S) = √(structural_bc(S) × lidar_dominance(S))
    structural_bc(S)    : betweenness介数中心性（越高越在图咽喉上）
    lidar_dominance(S) : LiDAR约束强度（边长大+yaw变化大→越弱）
  coverage(S)          : sigmoid(受影响帧数/5)

α = 0.3
```

| 因子 | 含义 |
|------|------|
| `reach` | 攻击范围内可到达的轨迹点数比例 |
| `bivul_gate` | 攻击窗口内的双模态脆弱性（方向匹配） |
| `structural_bc` | 因子图介数中心性（咽喉程度） |
| `lidar_dominance` | LiDAR约束弱（边长大、急弯） |
| `coverage` | 攻击覆盖范围（受影响帧数） |

输出中的 `bo_x` 和 `bo_y` 即最优 Spoofer 世界坐标，填入阶段 4 配置。

---

### 阶段 4：生成攻击 Rosbag

修改配置（`config_lvisam.json`）：

```json
{
  "main": {
    "input_file": "/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/handheld.bag",
    "output_file": "/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/attack_static/handheld_attack_static.bag",
    "reference_file": "/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/original/handheld_original_traj.csv",
    "spoofing_mode": "static",
    "spoofer_x": <阶段3输出>,
    "spoofer_y": <阶段3输出>,
    "static_geometry_model": "square",
    "wall_dist": 15.0
  }
}
```

生成攻击 bag：

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash
roslaunch slamspoof rosbag_editer_lvisam.launch \
    config_file_path:=/home/qu_menghao/catkin_ws/src/slamspoof/config_lvisam.json
```

---

### 阶段 5：录制攻击轨迹

```bash
# ========== 终端 1 ==========
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash
rosparam set use_sim_time true
roslaunch lvi_sam run.launch

# ========== 终端 2 ==========
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash
mkdir -p ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/attack_static
rosbag record -O ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/attack_static/handheld_attack_static_traj.bag \
    /lvi_sam/lidar/mapping/odometry

# ========== 终端 3 ==========
source /opt/ros/noetic/setup.bash
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/attack_static/handheld_attack_static.bag --clock
```

bag 播放完毕后，提取轨迹：

```bash
python3 ~/catkin_ws/src/slamspoof/scripts/extract_lvisam_odom_csv.py \
    --bag ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/attack_static/handheld_attack_static_traj.bag \
    --out ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/attack_static/handheld_attack_static_traj.csv
```

---

### 阶段 6：轨迹对比

```bash
python3 ~/catkin_ws/src/slamspoof/scripts/evaluate_attack.py \
    --orig ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/original/handheld_original_traj.csv \
    --att  ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/attack_static/handheld_attack_static_traj.csv \
    --out-dir ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/eval \
    --title "handheld: static attack (square)" \
    --spoofer-x <阶段3输出> --spoofer-y <阶段3输出> \
    --distance-threshold 15.0
```

---

## 关键参数说明

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `spoofing_mode` | 攻击模式 | `removal` / `static` / `dynamic` |
| `spoofer_x/y` | Spoofer 世界坐标 | 从阶段 3 获取 |
| `distance_threshold` | 触发半径 | 15 ~ 30 m |
| `spoofing_range` | 攻击窗口总角度 | 80° |
| `wall_dist` | 假墙固定距离（static） | 15 m |
| `wall_distance_min/max` | 动墙距离范围（dynamic） | 5 ~ 25 m |
| `static_geometry_model` | 假墙几何模型 | `original_random` / `beam_project` / `square` / `corner` |
| `square_rotate_rad` | 菱形旋转角 | 0（corner）/ π/4（平面） |
| `M_corr` | SLAM 最大对应距离 | 1.0 m（LVI-SAM） |
| `auto_cycle` | 自动推导最优振荡周期 | `true` |
| `LIO_GRAPH_DUMP_DIR` | 因子图 dump 输出目录 | `~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/graph_dumps` |
| `--graph-dump-dir` | Spoofer 选择脚本的因子图路径 | 指向 `LIO_GRAPH_DUMP_DIR` |

---

## 因子图 Dump 格式说明

LIO-SAM 和 LVI-SAM 的因子图均以 JSON 格式 dump，字段完全兼容，可共用同一 pipeline。

**节点格式**（每个 `dump_XXXXX.json` 的 `nodes` 数组）：

```json
{
  "id": 0,
  "x": 0.0, "y": 0.0, "z": 0.0,
  "qx": 0.1136, "qy": 0.0114, "qz": -0.0013, "qw": 0.9935
}
```

**因子格式**（`factors` 数组中的每个因子）：

```json
{
  "fidx": 0,
  "type": "BetweenFactor",
  "keys": ["X0", "X1"],
  "tx": 0.0, "ty": 0.0, "tz": 0.0,
  "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
  "noise": [0.000001, 0.000001, 0.000001, 0.0001, 0.0001, 0.0001],
  "source": "odometry"
}
```

| source 值 | 含义 |
|-----------|------|
| `prior` | 先验因子（首节点） |
| `odometry` | LiDAR 或 IMU 里程计因子 |
| `loop_closure` | 回环检测因子（LIO-SAM/LVI-SAM dump 中均可能出现） |

**多文件拼接**：因子图分布在多个增量 dump 文件中（每个关键帧一次），`select_spoofer_bi_bo.py` 的 `_precompute_graph_data()` 自动加载所有 dump 并拼接为完整图。

---

## 引用

```bibtex
@inproceedings{slamspoof2025,
  title={SLAMSpoof: Practical LiDAR Spoofing Attacks on Localization
         Systems Guided by Scan Matching Vulnerability Analysis},
  author={Nagata, R. and Koide, K. and Hayakawa, Y. and Suzuki, R.
           and Ikeda, K. and Sako, O. and Chen, Q.A. and Sato, T.
           and Yoshioka, K.},
  booktitle={ICRA},
  year={2025}
}

@inproceedings{lvisam2021shan,
  title={LVI-SAM: Tightly-coupled Lidar-Visual-Inertial Odometry
         via Smoothing and Mapping},
  author={Shan, T. and Englot, B. and Ratti, C. and Rus, D.},
  booktitle={ICRA},
  pages={5692--5698},
  year={2021}
}

@inproceedings{liosam2021shan,
  title={LIO-SAM: Tightly-coupled Lidar Inertial Odometry via
         Smoothing and Mapping},
  author={Shan, T. and Englot, B. and Meyers, D. and Wang, W.
          and Ratti, C. and Rus, D.},
  booktitle={ICRA},
  pages={5692--5698},
  year={2021}
}
```
