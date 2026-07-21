#!/usr/bin/env bash
set -euo pipefail

# Generate the attacker-side LIO-SAM proxy graph for CBD_Building_01.
# This is only used for placement features, not as the victim estimator.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/catkin_ws/datasets/official/fast_livo2}"
SEQUENCE="${SEQUENCE:-CBD_Building_01}"
BAG="${BAG:-$DATA_ROOT/raw_rosbags/FAST-LIVO2-Dataset/FAST-LIVO2官方数据集/${SEQUENCE}.bag}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/experiments_${SEQUENCE}}"
PROXY_ROOT="${PROXY_ROOT:-$OUT_ROOT/lio_sam_proxy}"

BAG="$BAG" \
OUT_ROOT="$OUT_ROOT" \
PROXY_ROOT="$PROXY_ROOT" \
LIO_BAG="${LIO_BAG:-$PROXY_ROOT/${SEQUENCE}_lio_proxy.bag}" \
GRAPH_DUMP_DIR="${GRAPH_DUMP_DIR:-$OUT_ROOT/lio_proxy_graph_dumps}" \
PLAY_RATE="${PLAY_RATE:-1.0}" \
bash "$SCRIPT_DIR/prepare_fast_livo2_lio_sam_proxy_graph.sh"
