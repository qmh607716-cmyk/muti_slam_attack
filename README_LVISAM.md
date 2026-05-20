# SLAMSpoof for LVI-SAM — Project README

## Overview

This project ports the **SLAMSpoof** (ICRA 2025) LiDAR spoofing attack framework onto **LVI-SAM**, a tightly-coupled LiDAR-Visual-Inertial SLAM system. The goal is to evaluate LVI-SAM's robustness against scan-matching-based LiDAR spoofing attacks.

The core idea: compute **SMVS** (Scan Matching Vulnerability Score) per frame using GICP-based localizability analysis, then use SMVS to select high-impact spoofing positions. The rosbag attack editor removes points in a configurable angular window while preserving LVI-SAM's full sensor input (camera, IMU, GPS, LiDAR with all fields).

---

## Project Structure

```
slamspoof/                          # ROS package name: slamspoof_icra
├── scripts/
│   ├── SpoofingSimulation.py        # SMVS analysis node (subscribes to live ROS topics)
│   ├── registration_separated.py     # GICP / Hessian computation (small_gicp backend)
│   ├── spoofing_editer_lvisam.py    # Rosbag attack editor for LVI-SAM
│   ├── spoofing_editer_imu.py       # Original SLAMSpoof editor (LiDAR-only, xyz-only output)
│   ├── select_spoofer_from_smvs.py  # Auto-select spoofer position from SMVS CSV
│   ├── compare_lvisam_trajectory.py  # Compare two trajectories and plot deviation
│   ├── extract_lvisam_trajectory.py # Extract /lvi_sam/lidar/mapping/odometry to CSV
│   ├── visualize.py                  # Visualise SMVS spatial distribution
│   ├── odom_node.py                 # Original SLAMSpoof odometry extractor
│   └── functions/
│       ├── calc_smvs.py             # SMVS score aggregation (HFR / AHFR modes)
│       ├── spoofing_sim.py           # HFR noise / static wall injection functions
│       └── bag2array.py             # Binary point cloud utilities
├── launch/
│   ├── run_smvs_lvisam.launch        # Run SMVS analysis + trajectory extraction
│   ├── rosbag_editer_lvisam.launch   # Run bag attack editor
│   ├── extract_trajectory_lvisam.launch  # Run trajectory extractor alone
│   ├── run_slam_loam.launch          # Run SMVS with A-LOAM (original SLAMSpoof)
│   ├── rosbag_editer.launch          # Original bag editor (LiDAR-only)
│   └── visualizer.launch             # SMVS heatmap visualiser
├── config.json                      # Original SLAMSpoof config (trajectory-guided)
├── config_lvisam.json               # LVI-SAM config (SMVS-guided)
└── ...
```

---

## LVI-SAM Dataset

**Bag file:** `~/catkin_ws/src/LVI-SAM/datasets/handheld.bag`

Main topics:

| Topic | Type | Notes |
|---|---|---|
| `/points_raw` | `sensor_msgs/PointCloud2` | Velodyne VLP-16, `point_step=22` |
| `/imu_correct` | `sensor_msgs/Imu` | Corrected IMU |
| `/imu_raw` | `sensor_msgs/Imu` | Raw IMU |
| `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | Camera |
| `/gps/fix` | `sensor_msgs/NavSatFix` | GPS |
| `/lvi_sam/lidar/mapping/odometry` | `nav_msgs/Odometry` | Output odometry |

**Point cloud field layout** (`point_step=22`):

```
field name    offset  datatype
x             0      float32
y             4      float32
z             8      float32
intensity    12      float32
ring         16      uint16
time         18      float32
```

> **Critical:** The attack editor must preserve all 22 bytes per point. Rebuilding a xyz-only `PointCloud2` (`point_step=12`) will break LVI-SAM because `ring` and `time` fields are required.

---

## Complete Experimental Workflow

### Prerequisites

```bash
# Install dependencies (standard SLAMSpoof requirements)
pip install numpy pandas matplotlib rosbags

# Install small_gicp (required by registration_separated.py)
# See: https://github.com/koide3/small_gicp

# Install GTSAM and Ceres (required by LVI-SAM)
sudo add-apt-repository ppa:borglab/gtsam-release-4.0
sudo apt install libgtsam-dev libgtsam-unstable-dev
sudo apt-get install libgoogle-glog-dev libatlas-base-dev

# Source workspace
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

---

### Step 1 — Record Reference (Original) Trajectory

Run LVI-SAM with the original bag and extract the odometry:

```bash
# Terminal 1: Launch LVI-SAM
roslaunch lvi_sam run.launch

# Terminal 2: Extract trajectory
roslaunch slamspoof_icra extract_trajectory_lvisam.launch

# Terminal 3: Play bag
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/handheld.bag
```

The output CSV will be at:
```
~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/{timestamp}_lvisam_traj.csv
```

Rename it to `original_full_traj.csv` and update `config_lvisam.json`:

```json
"reference_file": "/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/original_full_traj.csv"
```

---

### Step 2 — Compute SMVS

Run the SMVS analysis on the live bag:

```bash
# Terminal 1
roslaunch slamspoof_icra run_smvs_lvisam.launch

# Terminal 2
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/handheld.bag
```

Outputs:
```
~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/smvs/{timestamp}.csv
~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/vul/vul_{timestamp}.csv
```

---

### Step 3 — Auto-Select Spoofer Position from SMVS

```bash
python ~/catkin_ws/src/slamspoof/scripts/select_spoofer_from_smvs.py \
    --smvs  ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/smvs/{timestamp}.csv \
    --vul   ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/vul/vul_{timestamp}.csv \
    --config ~/catkin_ws/src/slamspoof/config_lvisam.json \
    --top-k 5 \
    --verbose
```

This prints the top-5 candidate positions and updates `config_lvisam.json` with the best one:

```
# selected spoofer_x, spoofer_y, distance_threshold, spoofing_range
```

Alternatively, manually set spoofer positions in `config_lvisam.json`.

---

### Step 4 — Generate Attacked Rosbag

Three attack modes are available:

#### Mode 1 — `removal` (HFR attack, default)

```bash
# spoofing_mode is already "removal" in config_lvisam.json
roslaunch slamspoof_icra rosbag_editer_lvisam.launch
```

#### Mode 2 — `static` (fixed false wall)

```bash
# Use config_lvisam_static.json (or edit config_lvisam.json to set spoofing_mode: "static")
roslaunch slamspoof_icra rosbag_editer_lvisam.launch \
    _config_file:=$(find slamspoof_icra)/config_lvisam_static.json
```

#### Mode 3 — `dynamic` (moving false wall)

Set `spoofing_mode: "dynamic"` in the config, plus:

```json
"wall_distance_min": 5.0,
"wall_distance_max": 25.0,
"spoofing_cycle": 2.0
```

All three modes preserve the original `point_step=22` layout. The injected / replaced points are synthesised with valid `x, y, z, intensity, ring, time` fields.

The editor:
- Reads `handheld.bag`
- For each `/points_raw` frame: looks up the robot's pose from `original_full_traj.csv`
- If the robot is within `distance_threshold` of the spoofer, removes points in the angular window `[spoofing_range]` centred on the spoofer direction
- Writes all other topics unchanged (camera, IMU, GPS, etc.)
- Outputs `handheld_full_attack_smvs_guided.bag`

---

### Step 5 — Run LVI-SAM on Attacked Bag

```bash
# Terminal 1
roslaunch lvi_sam run.launch

# Terminal 2
roslaunch slamspoof_icra extract_trajectory_lvisam.launch

# Terminal 3
rosbag play ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/handheld_full_attack_smvs_guided.bag
```

Rename the output to `attack_full_smvs_guided_traj.csv`.

---

### Step 6 — Compare Trajectories

```bash
python ~/catkin_ws/src/slamspoof/scripts/compare_lvisam_trajectory.py \
    --original ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/original_full_traj.csv \
    --attacked ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/attack_full_smvs_guided_traj.csv \
    --output-prefix full_smvs_guided \
    --output-dir ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_full/ \
    --spoofer-x 44.517 \
    --spoofer-y 129.788
```

Outputs:
- `{prefix}_trajectory_xy_compare.png` — XY overlay of both trajectories
- `{prefix}_deviation.png` — Deviation over time + XY with max-deviation highlighted
- Console summary with mean / max / final deviation

---

## Configuration Parameters

### `config_lvisam.json`

```json
{
  "main": {
    "input_file":        "/path/to/input.bag",
    "output_file":       "/path/to/output.bag",
    "reference_file":     "/path/to/original_full_traj.csv",

    "# === attack mode ===": "",
    "spoofing_mode":     "removal",   // "removal" | "static" | "dynamic"

    "# === spoofer placement (world coordinates) ===": "",
    "spoofer_x":         44.517,      // world X
    "spoofer_y":        129.788,      // world Y
    "distance_threshold": 5.0,         // metres — trigger radius

    "# === topic settings ===": "",
    "lidar_topic":       "/points_raw",
    "lidar_topic_length": 22,          // Velodyne VLP-16 = 22 bytes
    "imu_topic":         "/imu_correct",

    "# === attack geometry ===": "",
    "spoofing_range":    160.0,       // degrees — angular removal/injection window

    "# === static wall parameter (used when spoofing_mode=static) ===": "",
    "wall_dist":         15.0,        // metres

    "# === dynamic wall parameters (used when spoofing_mode=dynamic) ===": "",
    "wall_distance_min":   5.0,
    "wall_distance_max":  25.0,
    "spoofing_cycle":     2.0,         // seconds per cycle

    "# === reproducibility ===": "",
    "rng_seed":          42
  },
  "filtering": {
    "minimum_measuring_distance": 0.0,
    "maximum_measuring_distance": 30.0,
    "minimum_height_threshold": -2.0
  }
}
```

### Attack modes explained

| Mode | Behaviour | Key parameter |
|---|---|---|
| `removal` | Delete points in the attack window, replace with random noise (HFR) | `spoofing_rate` controls noise density |
| `static` | Delete points in the attack window, inject a flat wall at fixed distance | `wall_dist` (metres) |
| `dynamic` | Same as static, but wall distance oscillates between `wall_distance_min` and `wall_distance_max` over `spoofing_cycle` seconds | `wall_distance_min/max`, `spoofing_cycle` |

The injected points are **always valid 22-byte records** with correct `x, y, z, intensity, ring, time` fields, so LVI-SAM's scan registration will see a structurally valid point cloud.

---

## Key Implementation Details

### Point Cloud Preservation

The editor **never rebuilds** the `PointCloud2`. It reads the raw bytes, reshapes by `point_step=22`, removes entire 22-byte records, and writes back with the original field layout intact:

```python
point_step = points_msg.point_step
n_points = int(len(points_msg.data) / point_step)
bin_points = np.frombuffer(points_msg.data, dtype=np.uint8).reshape(n_points, point_step)
# ... masking ...
kept_points = bin_points[~remove_mask]
points_msg.width = kept_points.shape[0]
points_msg.row_step = kept_points.shape[0] * point_step
points_msg.data = kept_points.reshape(-1)
```

### Yaw-Aware Attack Direction

The attack direction is computed in the **LiDAR local frame** by subtracting the robot's yaw:

```python
world_angle = atan2(spoofer_y - robot_y, spoofer_x - robot_x)
local_angle = world_angle - robot_yaw   # robot_yaw from quaternion
# normalise to [-π, π]
local_angle = atan2(sin(local_angle), cos(local_angle))
```

This ensures the removal window is accurate regardless of the robot's world orientation.

### SMVS Computation

SMVS is computed per frame using GICP registration (via `small_gicp`):

1. Randomly sample two subsets of the point cloud.
2. Add Gaussian noise and run GICP alignment.
3. Extract the Hessian and per-point eigen values.
4. Bin eigen values by angular sectors (5° default).
5. Apply the HFR distance-weighted sum to produce a scalar SMVS per frame.

---

## Experimental Results (handheld.bag, full duration)

| Attack Method | Triggered Frames | Trigger Ratio | Mean Deviation | Max Deviation |
|---|---|---|---|---|
| Trajectory-guided | 74 / 16,289 | 0.45% | 0.85 m | 3.10 m |
| SMVS-guided | 76 / 16,289 | 0.47% | 1.26 m | 4.29 m |

**Conclusion:** With a comparable trigger rate (~0.47%), SMVS-guided attack produces ~1.5× larger maximum deviation, demonstrating that SMVS is an effective prior for attack position selection.

---

## Citation

If you use this code, please cite both the original SLAMSpoof paper and LVI-SAM:

```bibtex
@inproceedings{slamspoof2025,
  title={SLAMSpoof: Practical LiDAR Spoofing Attacks on Localization Systems ...},
  author={Nagata et al.},
  booktitle={ICRA},
  year={2025},
}
@inproceedings{lvisam2021shan,
  title={LVI-SAM: Tightly-coupled Lidar-Visual-Inertial Odometry via Smoothing and Mapping},
  author={Shan et al.},
  booktitle={ICRA},
  year={2021},
}
```
