# R3LIVE Secondary Experiment

This note records the prepared R3LIVE transfer-experiment path. It mirrors the
FAST-LIVO2 secondary branch and keeps the same attack parameters unless
explicitly overridden.

## Workspace

R3LIVE is managed in the main workspace:

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
catkin build r3live lio_sam slamspoof_icra
source ~/catkin_ws/devel_catkin_tools/setup.bash
```

## Dataset

Default expected official sequence:

```text
~/catkin_ws/datasets/official/r3live/raw_rosbags/R3LIVE-Dataset/hku_campus_seq_00.bag
```

If you download a different R3LIVE official sequence, either place it under the
same `R3LIVE-Dataset/` directory and set `SEQUENCE=<bag_name_without_.bag>`, or
set `BAG=/absolute/path/to/file.bag` when running the scripts.

Relevant official topics:

- `/livox/lidar`: `livox_ros_driver/CustomMsg`
- `/livox/imu`: `sensor_msgs/Imu`
- `/camera/image_color/compressed`: `sensor_msgs/CompressedImage`

R3LIVE publishes odometry on:

```text
/aft_mapped_to_init
```

## Step 1: clean run

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash

MODE=clean RUN_NAME=clean_01 PLAY_RATE=1.0 \
  bash ~/catkin_ws/src/slamspoof/scripts/run_r3live_official_once.sh
```

Expected clean trajectory:

```text
~/catkin_ws/datasets/official/r3live/experiments/runs/clean_01/clean_01_traj.csv
```

## Step 2: LIO-SAM proxy graph

This graph is an attacker-side proxy, not R3LIVE's internal optimizer graph.
It is used only for route-level structural cues in Bi-SMVS placement.

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash

bash ~/catkin_ws/src/slamspoof/scripts/prepare_r3live_lio_sam_proxy_graph.sh
```

Expected proxy graph output:

```text
~/catkin_ws/datasets/official/r3live/experiments/lio_proxy_graph_dumps/
```

## Step 3: compute SMVS/Bi-SMVS and select positions

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash

GRAPH_SOURCE=lio_sam \
  bash ~/catkin_ws/src/slamspoof/scripts/prepare_r3live_method_positions.sh
```

For a quick computation smoke test only:

```bash
MAX_FRAMES=5 GRAPH_SOURCE=lio_sam \
  bash ~/catkin_ws/src/slamspoof/scripts/prepare_r3live_method_positions.sh
```

Real runs should not set `MAX_FRAMES`.

Outputs:

```text
~/catkin_ws/datasets/official/r3live/experiments/vulnerability/smvs/hku_campus_seq_00_SMVS.csv
~/catkin_ws/datasets/official/r3live/experiments/vulnerability/smvs/hku_campus_seq_00_BiSMVS.csv
~/catkin_ws/datasets/official/r3live/experiments/selections/smvs_1580.json
~/catkin_ws/datasets/official/r3live/experiments/selections/bismvs_1580.json
~/catkin_ws/datasets/official/r3live/experiments/method_spoofer_positions_1580.csv
```

## Step 4: x3 transfer suite

```bash
bash ~/catkin_ws/src/slamspoof/scripts/run_r3live_methods_1580_x3.sh
```

Defaults:

- methods: `smvs`, `bismvs`
- modes: `static`, `removal`
- runs per condition: `3`
- trigger distance: `15 m`
- spoofing angular range: `80 deg`
- candidate distance to trajectory: `5-14 m`
- playback rate: `1.0`
- run tag: `proxy`

## Step 5: formal x15 suite

```bash
bash ~/catkin_ws/src/slamspoof/scripts/run_r3live_methods_1580_x15.sh
```

This wrapper sets `N_RUNS=15`, `PLAY_RATE=1.0`, and `RUN_TAG=lio_proxy` by
default.

## Implementation Notes

- The victim run is R3LIVE launched through
  `slamspoof_icra/launch/r3live_official.launch`.
- The R3LIVE official launch path keeps `/livox/lidar`, `/livox/imu`, and
  `/camera/image_color/compressed` compatible with the official dataset.
- The attack editor is the same Livox `CustomMsg` editor used by the
  FAST-LIVO2 transfer branch:
  `scripts/spoofing_editer_livox_fastlivo2.py`.
- Static attack uses `beam_project`; removal uses the same HFR-style random
  replacement convention as the FAST-LIVO2 branch.
- The default R3LIVE placement distance is `5-14 m`, because
  `hku_campus_seq_00` is shorter and narrower than the FAST-LIVO2 smoke
  sequence. With the original `8-14 m` band, Bi-SMVS candidates cannot both
  align with the top vulnerable directions and satisfy feasible trigger
  coverage. The trigger distance remains `15 m`.
- R3LIVE vulnerability computation reads
  `~/catkin_ws/src/R3LIVE/config/r3live_config.yaml` for the camera model and
  LiDAR-to-camera projection.
