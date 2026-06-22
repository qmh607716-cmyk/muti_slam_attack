# LV-SLAM Attack

## 概述
SLAMSpoof (ICRA 2025) LiDAR 欺骗攻击框架移植到 **LVI-SAM**类LiDAR-视觉-惯性耦合SLAM系统


### 1. `removal` — HFR 噪声攻击
删除攻击窗口内真实点，注入随机噪声点。模拟硬件干扰或信号阻塞。

### 2. `static` — 假墙注入

**原版**：圆柱形均匀假墙（`original_random`），在 `wall_dist` 距离上均匀注入伪造点，几何约束分散。

**扩展**（由 D-SLAMSpoof 论文提出 `square`/`corner`，其余为本工作）：

| 模型 | 来源 | 描述 |
|---|---|---|
| `original_random` | 原版 | 均匀随机角度分布的圆柱墙，几何约束分散 |
| `beam_project` | 本工作 | 沿原 scan line 方向投影到固定距离，继承 ring/time |
| `square` | D-SLAMSpoof | 菱形集中几何（极坐标方程），约束集中在边缘方向 |
| `corner` | D-SLAMSpoof | L 形墙角（square + rotate=0），两侧边缘面向 LiDAR |


### 3. `dynamic` — 动墙注入
墙距离在 `[wall_distance_min, wall_distance_max]` 之间周期性振荡，周期由 $M_{corr}$ 自动推导：
```
t_cycle = (d_max - d_min) / M_corr × Δt
```
该周期是**最快且不被 outlier filtering 拒绝**的振荡频率。

---

## 快速开始

### 需求
- ROS Noetic + Catkin Tools
- LVI-SAM（`~/catkin_ws/devel_catkin_tools`）
- `small_gicp`（G-ICP 后端）
- 数据集：`~/catkin_ws/src/LVI-SAM/datasets/xxx.bag`

### 环境准备
```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash
```

---

## 完整实验流程

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
rosbag record -O ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/original/handheld_original_traj.bag /lvi_sam/lidar/mapping/odometry

# ========== 终端 3 ==========
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/handheld.bag --clock --pause
```

bag 播放完毕后（终端 3 自动结束），提取轨迹：

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
roslaunch slamspoof_icra run_bimodal_smvs_lvisam.launch

# ========== 终端 3 ==========
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/handheld.bag --clock
```

输出文件（由 launch 文件 `smvs_save_dir` / `vulnerablity_save_dir` 参数指定）：
```
slamspoof_handheld/smvs/{timestamp}.csv      # 帧级 SMVS 分数
slamspoof_handheld/vul/vul_{timestamp}.csv  # 分方向脆弱性
```

---

### 阶段 3：选择 Spoofer 位置

```bash
# 【重要】将路径替换为阶段 2 新生成的 CSV 文件
python3 ~/catkin_ws/src/slamspoof/scripts/select_spoofer_from_bimodal.py \
    --smvs ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/smvs/{timestamp}.csv \
    --vul  ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/vul/vul_{timestamp}.csv \
    --score-column frame_bi_smvs \
    --score-threshold 0.0 \
    --top-k 10 \
    --verbose \
    --match-mode nearest_xy
```

输出中的 `spoofer_x` 和 `spoofer_y` 填入 `config_lvisam.json`（见下阶段）

---

### 阶段 4：生成攻击 Rosbag

修改配置（config_lvisam：修改路径/攻击模式/攻击参数）：

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
	roslaunch slamspoof_icra rosbag_editer_lvisam.launch config_file_path:=/home/qu_menghao/catkin_ws/src/slamspoof/config_lvisam.json
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
source ~/catkin_ws/devel_catkin_tools/setup.bash
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
	    --att ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/attack_static/smvs/handheld_attack_static_traj.csv \
	    --out-dir ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/eval \
	    --title "handheld: static attack" \
	    --spoofer-x xxx --spoofer-y xxx --distance-threshold xx
```

---

## 关键参数说明

| 参数 | 说明 | 常用值 |
|---|---|---|
| `spoofing_mode` | 攻击模式 | `removal` / `static` / `dynamic` |
| `spoofer_x/y` | Spoofer 世界坐标 | 从阶段 3 获取 |
| `distance_threshold` | 触发半径 | 15 ~30m |
| `spoofing_range` | 攻击窗口总角度 | 80° |
| `wall_dist` | 假墙固定距离（static） | 15 m |
| `wall_distance_min/max` | 动墙距离范围（dynamic） | 5 ~ 25 m |
| `static_geometry_model` | 假墙几何 | `original_random` / `beam_project` / `square` / `corner` |
| `square_rotate_rad` | 菱形旋转角 | 0（corner）/ π/4（平面） |
| `M_corr` | SLAM 最大对应距离 | 1.0 m（LVI-SAM） |
| `auto_cycle` | 自动推导最优振荡周期 | `true` |



## 实验结果（统一 evo APE 评估）

### 评估口径

- **A**：APE translation RMSE（m），evo Umeyama SE(3)+scale 对齐
- **B**：APE rotation RMSE（deg）
- **C**：max RPE translation（m），1m/10m delta 中的最大值
- **D**：raw 2D 偏差最大值（m，未对齐，反映真实攻击距离）
- 成功阈值：**APE ≥ 4.2 m**（SLAMSpoof 论文 §I 标准，"超出一条车道"）

所有 cell 的 CSV 路径：`~/catkin_ws/src/LVI-SAM/datasets/slamspoof_<platform>/attack_<mode>/<smvs>_<cfg>/`。

### Spoofer 坐标（选点结果）

| Platform × SMVS | spoofer_x | spoofer_y | 选点算法 |
|---|---|---|---|
| handheld + smvs | 193.91 | −12.71 | `select_spoofer_from_smvs_paper.py`（论文 §III-C 半线交点） |
| handheld + bismvs | 47.85 | −77.70 | `select_spoofer_bi_bo.py`（CMA-ES + 累积漂移） |
| jackal + smvs | 32.28 | 10.81 | `select_spoofer_from_smvs_paper.py` |
| jackal + bismvs | 152.55 | 226.58 | `select_spoofer_bi_bo.py` |

注入几何：所有 cell 均为 `static_geometry_model=square`（D-SLAMSpoof 菱形），`wall_dist=15 m`。
Attack direction：固定指向 spoofer（`atan2(spoofer − robot)`，LiDAR local frame）。

### distance_threshold=30, spoofing_range=180

| Group | A: APE-trans-RMSE (m) | B: APE-rot-RMSE (deg) | C: max RPE-trans (m) | D: raw 2D max (m) | 成功? |
|---|---:|---:|---:|---:|---|
| handheld_smvs_static    |  **1.10** |   0.62 |   5.25 |   4.06 | ❌ |
| handheld_smvs_removal   |  **1.51** |   0.82 |   3.92 |   3.46 | ❌ |
| handheld_bismvs_static  | **95.18** |  36.81 |  28.61 | **318.39** | ✅✅ |
| handheld_bismvs_removal |  **3.97** |  12.96 |  19.46 |  13.77 | ❌ |
| jackal_smvs_static      |  **6.13** |   8.41 |  17.00 | **114.35** | ✅ |
| jackal_smvs_removal     |  **0.35** |   0.26 |   7.71 |   7.70 | ❌ |
| jackal_bismvs_static    | **55.26** |  39.00 | **115.64** | **231.32** | ✅✅ |
| jackal_bismvs_removal   |  **0.52** |   0.27 |   7.71 |   7.75 | ❌ |

### distance_threshold=30, spoofing_range=80

| Group | A: APE-trans-RMSE (m) | B: APE-rot-RMSE (deg) | C: max RPE-trans (m) | D: raw 2D max (m) | 成功? |
|---|---:|---:|---:|---:|---|
| handheld_smvs_static    |  **1.48** |   0.84 |   3.88 |   3.46 | ❌ |
| handheld_smvs_removal   |  **1.46** |   0.73 |   5.21 |   3.80 | ❌ |
| handheld_bismvs_static  | **77.20** |  30.95 |  19.52 | **263.25** | ✅✅ |
| handheld_bismvs_removal |  **0.78** |  11.89 |   5.19 |   5.04 | ❌ |
| jackal_smvs_static      |  **0.30** |   0.18 |   7.72 |   7.72 | ❌ |
| jackal_smvs_removal     |  **0.33** |  10.88 |   7.73 |   7.89 | ❌ |
| jackal_bismvs_static    |  **0.44** |   0.29 |   7.69 |   7.76 | ❌ |
| jackal_bismvs_removal   |  **0.35** |  10.88 |  20.40 |   7.91 | ❌ |

### distance_threshold=15, spoofing_range=80

| Group | A: APE-trans-RMSE (m) | B: APE-rot-RMSE (deg) | C: max RPE-trans (m) | D: raw 2D max (m) | 成功? |
|---|---:|---:|---:|---:|---|
| handheld_smvs_static    |  **2.11** |   1.05 |   5.25 |   7.31 | ❌ |
| handheld_smvs_removal   |  **1.90** |   1.00 |   5.26 |   4.75 | ❌ |
| handheld_bismvs_static  |  **2.00** |   1.06 |   3.89 |   5.13 | ❌ |
| handheld_bismvs_removal |  **2.32** |   1.20 |   3.75 |   6.26 | ❌ |
| jackal_smvs_static      |  **0.35** |  10.92 |   7.63 |   7.79 | ❌ |
| jackal_smvs_removal     |  **0.31** |   0.15 |   7.68 |   7.79 | ❌ |
| jackal_bismvs_static    |  **0.39** |  10.85 |  19.19 |   7.81 | ❌ |
| jackal_bismvs_removal   |  **0.35** |  10.83 |  20.41 |   7.84 | ❌ |

### 关键观察

1. **`distance_threshold=30, spoofing_range=180` 是唯一强有效的配置**。Bi-SMVS static 在 handheld (95 m) / jackal (55 m) 都远超 4.2 m 阈值。
2. **Bi-SMVS static >> SMVS static**：在 dt=30/sr=180 下，Bi-SMVS 把 handheld 攻击从 1.10 m 提升到 95.18 m（**86×**），jackal 从 6.13 m 提升到 55.26 m（**9×**）。
3. **`removal` 模式对 LVI-SAM 几乎无效**：所有 removal cell 的 APE < 4.2 m，**最大也才 3.97 m**（handheld_bismvs_removal）。原因：LVI-SAM 的 VINS 视觉跟踪能恢复 HFR 删除的点。
4. **`dt=15` 太短**：spoofer 几乎接触不到 trajectory，攻击窗口太短，没有一次达到阈值。
5. **`dt=30, sr=80` 比 `dt=30, sr=180` 弱很多**：80° 攻击窗不够覆盖完整的 Bi-Vul 高发方向，效果退化（handheld_bismvs_static 77.20 vs 95.18，jackal_bismvs_static 0.44 vs 55.26）。
6. **raw 2D max（攻击瞬间最大偏离）能到 318 m**，但 Umeyama 对齐后 APE 只有 95 m —— 说明攻击会让车短暂偏离 ~300 m 再被 SLAM 拉回。

### 8 组完整结果网格（dt=30, sr=180 选 8 个）

8 组 XY 轨迹（Umeyama 对齐后）        |  8 组逐帧 2D 偏差（对齐后）
:-------------------------:|:-------------------------:
![xy](docs/results/summary_xy_grid.png) | ![dev](docs/results/summary_deviation_grid.png)


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
```
