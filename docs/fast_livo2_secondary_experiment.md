# FAST-LIVO2 Secondary Experiment

This note records the prepared FAST-LIVO2 transfer experiment path.

## Workspace

FAST-LIVO2 and its Vikit dependency are managed in the main workspace:

- `~/catkin_ws/src/FAST-LIVO2`
- `~/catkin_ws/src/rpg_vikit`
- `~/catkin_ws/src/R3LIVE/livox_ros_driver/livox_ros_driver`

Build and source:

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
catkin build livox_ros_driver vikit_common vikit_ros fast_livo slamspoof_icra
source ~/catkin_ws/devel_catkin_tools/setup.bash
```

## Dataset

Prepared official FAST-LIVO2 sequence:

```text
~/catkin_ws/datasets/official/fast_livo2/raw_rosbags/FAST-LIVO2-Dataset/Bright_Screen_Wall.bag
```

Topics:

- `/livox/lidar`: `livox_ros_driver/CustomMsg`
- `/livox/imu`: `sensor_msgs/Imu`
- `/left_camera/image/compressed`: `sensor_msgs/CompressedImage`

FAST-LIVO2 publishes odometry on:

```text
/aft_mapped_to_init
```

## Step 1: clean run

Run FAST-LIVO2 on the official bag and record the clean trajectory:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash

MODE=clean RUN_NAME=clean_01 PLAY_RATE=1.0 \
  bash ~/catkin_ws/src/slamspoof/scripts/run_fast_livo2_official_once.sh
```

Expected clean trajectory:

```text
~/catkin_ws/datasets/official/fast_livo2/experiments/runs/clean_01/clean_01_traj.csv
```

## Calibration checked for the official sequence

For `Bright_Screen_Wall`, the FAST-LIVO2 official dataset calibration and the
local FAST-LIVO2 launch/config agree on the following interfaces:

- `mapping_avia.launch` loads `config/avia.yaml` and `config/camera_pinhole.yaml`.
- LiDAR topic: `/livox/lidar` (`livox_ros_driver/CustomMsg`).
- IMU topic: `/livox/imu`.
- Camera input in the bag: `/left_camera/image/compressed`.
- FAST-LIVO2 republishes it to `/left_camera/image`.
- FAST-LIVO2 applies `time_offset/img_time_offset = 0.1` to image stamps.
- FAST-LIVO2 resizes images by `scale = 0.5`, so the VIO camera model is
  `640 x 512`, `fx=646.78472`, `fy=646.65775`, `cx=313.456795`,
  `cy=261.399612`.
- LiDAR-camera extrinsic in `avia.yaml`:
  `Rcl=[0.00610193,-0.999863,-0.0154172,-0.00615449,0.0153796,-0.999863,0.999962,0.00619598,-0.0060598]`,
  `Pcl=[0.0194384,0.104689,-0.0251952]`.

The dataset `calibration.yaml` contains several repeated sequence-specific
blocks; for `Retail_Street`, `CBD_Building_01`, and `Bright_Screen_Wall`, the
first block matches `camera_pinhole.yaml` and `avia.yaml`.

## Step 2A: compute SMVS/Bi-SMVS and select method positions

This is the formal transfer-experiment selection path. It computes
FAST-LIVO2-specific SMVS/Bi-SMVS CSVs from the official Livox bag and clean
trajectory, then runs the paper-SMVS selector and a FAST-LIVO2 proxy-assisted
Bi-SMVS selector.

FAST-LIVO2 does not expose the same graph-dump interface used by the LVI-SAM
main experiment. The preferred transfer path now runs LIO-SAM on a converted
copy of the same official FAST-LIVO2 bag and uses the exported LIO-SAM graph as
an attacker-side proxy. This proxy graph is used only for route-level structural
cues: structural importance, graph coverage, and LiDAR-side route exposure. It
is not claimed to be FAST-LIVO2's internal optimizer graph.

The FAST-LIVO2 official bag stores Livox packets as `CustomMsg`, while LIO-SAM
expects `PointCloud2` with `ring/time` fields. The proxy-preparation script
therefore converts `/livox/lidar` to `/points_raw`, forwards `/livox/imu` to
`/imu_raw`, and fills invalid IMU orientation quaternions with identity
orientation so LIO-SAM can run as a proxy estimator.

Generate the LIO-SAM proxy graph first:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash

bash ~/catkin_ws/src/slamspoof/scripts/prepare_fast_livo2_lio_sam_proxy_graph.sh
```

Expected proxy graph output:

```text
~/catkin_ws/datasets/official/fast_livo2/experiments/lio_proxy_graph_dumps/
```

Then compute/select method positions using that proxy:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel_catkin_tools/setup.bash

RECOMPUTE_VULNERABILITY=0 GRAPH_SOURCE=lio_sam \
  bash ~/catkin_ws/src/slamspoof/scripts/prepare_fast_livo2_method_positions.sh
```

Outputs:

```text
~/catkin_ws/datasets/official/fast_livo2/experiments/vulnerability/smvs/Bright_Screen_Wall_SMVS.csv
~/catkin_ws/datasets/official/fast_livo2/experiments/vulnerability/vul/vul_Bright_Screen_Wall_SMVS.csv
~/catkin_ws/datasets/official/fast_livo2/experiments/vulnerability/smvs/Bright_Screen_Wall_BiSMVS.csv
~/catkin_ws/datasets/official/fast_livo2/experiments/vulnerability/vul/vul_Bright_Screen_Wall_BiSMVS.csv
~/catkin_ws/datasets/official/fast_livo2/experiments/selections/smvs_1580.json
~/catkin_ws/datasets/official/fast_livo2/experiments/selections/bismvs_1580.json
~/catkin_ws/datasets/official/fast_livo2/experiments/method_spoofer_positions_1580.csv
```

For a quick smoke test of the computation stage only:

```bash
MAX_FRAMES=5 bash ~/catkin_ws/src/slamspoof/scripts/prepare_fast_livo2_method_positions.sh
```

For the real run, do not set `MAX_FRAMES`.

## Step 2B: sample feasible spoofer positions

After the clean trajectory exists:

```bash
CLEAN_CSV=~/catkin_ws/datasets/official/fast_livo2/experiments/runs/clean_01/clean_01_traj.csv \
  bash ~/catkin_ws/src/slamspoof/scripts/prepare_fast_livo2_official_positions.sh
```

This writes:

```text
~/catkin_ws/datasets/official/fast_livo2/experiments/spoofer_positions_1580.csv
```

The default constraints are:

- trigger distance `D=15 m`
- spoofing angular range `80 deg`
- candidate distance to trajectory: `8-14 m`
- score-free constrained sampling, used only as a feasible transfer-test/random baseline location

## Step 3: static attack run

The run script reads the first row of `spoofer_positions_1580.csv` unless
`SPOOFER_X` and `SPOOFER_Y` are explicitly set.

```bash
REF_CSV=~/catkin_ws/datasets/official/fast_livo2/experiments/runs/clean_01/clean_01_traj.csv \
MODE=static RUN_NAME=static_1580_01 PLAY_RATE=1.0 \
  bash ~/catkin_ws/src/slamspoof/scripts/run_fast_livo2_official_once.sh
```

## Step 4: removal/HFR-style attack run

```bash
REF_CSV=~/catkin_ws/datasets/official/fast_livo2/experiments/runs/clean_01/clean_01_traj.csv \
MODE=removal RUN_NAME=removal_1580_01 PLAY_RATE=1.0 \
  bash ~/catkin_ws/src/slamspoof/scripts/run_fast_livo2_official_once.sh
```

## Step 5: method-position transfer suite

After Step 2A has produced `method_spoofer_positions_1580.csv`, run the formal
SMVS/Bi-SMVS transfer comparison:

```bash
bash ~/catkin_ws/src/slamspoof/scripts/run_fast_livo2_methods_1580_x3.sh
```

Defaults:

- methods: `smvs`, `bismvs`
- modes: `static`, `removal`
- runs per condition: `3`
- trigger distance: `15 m`
- spoofing angular range: `80 deg`
- playback rate: `1.0`
- Bi-SMVS trigger coverage constraint: `30%-70%`, target `50%`
- run tag: `proxy`, so new runs do not overwrite earlier smoke-test runs

To run the formal x15 suite with LIO-SAM proxy positions:

```bash
bash ~/catkin_ws/src/slamspoof/scripts/run_fast_livo2_methods_1580_x15.sh
```

This wrapper sets `N_RUNS=15`, `PLAY_RATE=1.0`, and `RUN_TAG=lio_proxy` by
default. To overwrite old run names intentionally, set `RUN_TAG=""`.

## Outputs

Summary CSV:

```text
~/catkin_ws/datasets/official/fast_livo2/experiments/summary.csv
```

Per-run logs and trajectories:

```text
~/catkin_ws/datasets/official/fast_livo2/experiments/runs/<RUN_NAME>/
```

## Implementation notes

- Official FAST-LIVO2 bags use Livox `CustomMsg`, so the LVI-SAM
  PointCloud2 editor is not reused directly.
- The prepared Livox editor is:
  `~/catkin_ws/src/slamspoof/scripts/spoofing_editer_livox_fastlivo2.py`.
- The FAST-LIVO2 vulnerability generator is:
  `~/catkin_ws/src/slamspoof/scripts/compute_fast_livo2_vulnerability.py`.
- The FAST-LIVO2 proxy-assisted Bi-SMVS selector is:
  `~/catkin_ws/src/slamspoof/scripts/select_fast_livo2_bismvs_position.py`.
- The preferred proxy graph is exported by LIO-SAM and documented in
  `selections/bismvs_1580.json` as `source: lio_sam_graph_dump`. The selector
  aligns it to the FAST-LIVO2 clean trajectory before scoring candidate
  spoofer locations. It remains a surrogate attack-planning graph, not
  FAST-LIVO2's internal factor graph.
- Static attack defaults to `beam_project` geometry for Livox data, preserving
  affected points' `offset_time`, `reflectivity`, `tag`, and `line` fields while
  projecting them to the spoofed wall distance.
- Removal mode follows the existing HFR-style convention: remove points in the
  angular window and inject random spoofed returns, not pure point deletion.
