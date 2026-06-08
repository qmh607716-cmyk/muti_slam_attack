#!/usr/bin/env python3
"""
select_spoofer_from_bimodal.py
================================

基于双模态（LiDAR + Visual）脆弱性的 spoofer 位置选择。

融合公式：
  Bi-Vul[k] = L-Vul[k] × (1 - V-Vul[k] × L-Vul_norm[k])
  L-Vul_norm[k] = L-Vul[k] / l_vul_max  (per-frame normaliser)
  V-Vul[k] = γ × cam_coverage[k] × Q[k]
  Q[k] = 0.20 × feature_density[k]
       + 0.30 × flow_consistency[k]
       + 0.20 × depth_quality           (global scalar, same for all buckets)
       + 0.15 × spatial_dist            (global scalar, same for all buckets)
       + 0.15 × parallax                (global scalar, same for all buckets)

含义：
  Bi-Vul = 攻击效果（越小越脆弱 = 攻击越容易成功）
  V-Vul  = 视觉补偿能力（越大 → Bi-Vul 越小 → 攻击被削弱）

用法：
    python select_spoofer_from_bimodal.py \
        --smvs ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/smvs/xxx.csv \
        --vul  ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/vul/xxx.csv \
        --top-k 10 \
        --score-column frame_bi_smvs \
        --verbose

输出 JSON:
    {
        "spoofer_x": float,
        "spoofer_y": float,
        "distance_threshold": float,
        "spoofing_range": float,
        "frame_bi_smvs_threshold": float,
        "selection_info": { ... }
    }
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# 复用原版 spoofer 选择的几何算法（射线交点、sigma 过滤、轨迹拟合等）
import select_spoofer_from_smvs_paper as orig



def _ransac_intersections(rays: List[orig.Ray],
                          iterations: int = 200,
                          dist_thresh: float = 5.0) -> Tuple[float, float, List[int]]:
    """
    RANSAC-based ray intersection: finds the most robust intersection point
    among a set of rays that may contain near-parallel outliers.

    Returns:
        cx, cy: intersection centre (critical-object position)
        inlier_ray_indices: indices into ``rays`` that support this centre
    """
    n = len(rays)
    if n < 2:
        raise SystemExit("RANSAC needs at least 2 rays.")

    best_inliers: List[int] = []
    best_cx, best_cy = 0.0, 0.0
    best_score = -1.0

    xs  = np.array([r.x  for r in rays], dtype=float)
    ys  = np.array([r.y  for r in rays], dtype=float)
    dxs = np.array([r.dx for r in rays], dtype=float)
    dys = np.array([r.dy for r in rays], dtype=float)

    rng = np.random.default_rng(42)

    for _ in range(iterations):
        indices = rng.choice(n, size=2, replace=False)
        i, j = int(indices[0]), int(indices[1])
        out = orig.intersect_rays(rays[i], rays[j])
        if out is None:
            continue
        ix, iy, t1, t2 = out

        # Inlier test: perpendicular distance <= dist_thresh
        perp = np.abs(dys * (xs - ix) - dxs * (ys - iy)) \
               / np.sqrt(dxs**2 + dys**2 + orig.EPS)
        inlier_mask = perp <= dist_thresh
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count > best_score:
            best_score = inlier_count
            best_cx = ix
            best_cy = iy
            best_inliers = [k for k in range(n) if inlier_mask[k]]

    if not best_inliers:
        print("[WARN] RANSAC found no inliers; falling back to sigma-mean.",
              file=sys.stderr)
        all_pts = []
        for i in range(n):
            for j in range(i + 1, n):
                out = orig.intersect_rays(rays[i], rays[j])
                if out is not None:
                    all_pts.append((out[0], out[1]))
        if not all_pts:
            raise SystemExit("No valid intersections found.")
        pts = np.array(all_pts)
        cx_s = float(np.mean(pts[:, 0]))
        cy_s = float(np.mean(pts[:, 1]))
        return cx_s, cy_s, list(range(n))

    # Refine: recompute centre only from inlier-pair intersections
    refined = []
    for a_idx, a in enumerate(best_inliers):
        for b in best_inliers[a_idx + 1:]:
            out = orig.intersect_rays(rays[a], rays[b])
            if out is not None:
                refined.append((out[0], out[1]))
    if refined:
        pts = np.array(refined)
        best_cx = float(np.mean(pts[:, 0]))
        best_cy = float(np.mean(pts[:, 1]))

    return best_cx, best_cy, best_inliers


# ---------------------------------------------------------------------------
# 轨迹约束扫描：沿垂线找"脆弱性好 + 轨迹可达"的最优点
# ---------------------------------------------------------------------------

def trajectory_constrained_scan(
    critical_x: float,
    critical_y: float,
    traj_m: float,
    traj_n: float,
    inlier_rays: list,
    vul_df: pd.DataFrame,
    smvs_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    scan_min: float = 5.0,
    scan_max: float = 80.0,
    scan_step: float = 1.0,
    dist_limit: float = 30.0,
    alpha: float = 0.5,
) -> dict:
    """
    沿过关键物体中心 C 的垂线扫描所有候选 spoofer 位置，
    选取得分最高的位置：
        score = alpha * vuln_score + (1-alpha) * reachability_score

    其中：
        vuln_score = 该方向上所有 inlier 帧的平均 Bi-Vul（越大越脆弱）
        reachability_score = 轨迹点到该 spoofer 的最近距离在 dist_limit 内为 1，否则为 0

    Args:
        critical_x, critical_y: RANSAC 找到的关键物体中心 C
        traj_m, traj_n:         轨迹直线 y = m*x + n
        inlier_rays:             RANSAC inlier 射线列表
        vul_df:                  vulnerability DataFrame（含 bi_vul_00..71）
        smvs_df:                 SMVS DataFrame（含 frame_bi_smvs、timestamp）
        traj_df:                 完整轨迹 DataFrame（含 x, y）
        scan_min/max/step:       扫描范围和步长（m）
        dist_limit:              可达距离上限（m）
        alpha:                   脆弱性权重（1=只看脆弱性，0=只看可达性）

    Returns:
        dict，含 best_x, best_y, best_score, 以及 scan 详细结果
    """
    # 垂线方向向量（轨迹法向量）
    if abs(traj_m) < 1e-6:
        # 轨迹水平 → 垂线垂直
        dir_x, dir_y = 0.0, 1.0
    else:
        # 垂线斜率 = -1/m
        dir_x = 1.0 / math.sqrt(1 + traj_m**2)
        dir_y = traj_m * dir_x

    scan_results = []
    t = scan_min
    while t <= scan_max + 1e-9:
        for sign in [-1.0, 1.0]:
            sx = critical_x + sign * t * dir_x
            sy = critical_y + sign * t * dir_y

            # 1. 计算到轨迹的最近距离
            dx = traj_df["x"].values - sx
            dy = traj_df["y"].values - sy
            dists = np.sqrt(dx**2 + dy**2)
            min_traj_dist = float(np.min(dists))

            # 2. 计算该方向上的脆弱性
            #    每个 inlier 帧都有一个最脆弱方向（bi_vul bucket），取该 bucket 均值
            vuln_scores = []
            for ray in inlier_rays:
                ts = ray.timestamp
                if ts is not None:
                    matched = vul_df.iloc[(vul_df["timestamp"] - ts).abs().argsort()[0]]
                    bi_cols = [c for c in vul_df.columns if c.startswith("bi_vul_")]
                    if bi_cols:
                        bi_vals = matched[bi_cols].values.astype(float)
                        if len(bi_vals) > 0 and not np.all(np.isnan(bi_vals)):
                            vuln_scores.append(float(np.nanmean(bi_vals)))
            avg_vuln = float(np.mean(vuln_scores)) if vuln_scores else 0.0

            # 3. 综合得分（归一化）
            #    脆弱性：Bi-Vul 越大越脆弱，归一化到 [0,1]
            #    可达性：dist < dist_limit → 1.0，否则 → 0.0
            reach = 1.0 if min_traj_dist <= dist_limit else 0.0
            score = alpha * (avg_vuln / 1000.0) + (1 - alpha) * reach

            scan_results.append({
                "sx": float(sx),
                "sy": float(sy),
                "t": float(t),
                "side": "positive" if sign > 0 else "negative",
                "min_traj_dist": float(min_traj_dist),
                "avg_bi_vuln": float(avg_vuln),
                "reachable": bool(reach),
                "score": float(score),
            })
        t += scan_step

    # 按得分排序
    scan_results.sort(key=lambda x: x["score"], reverse=True)
    best = scan_results[0]

    return {
        "best_sx": best["sx"],
        "best_sy": best["sy"],
        "best_score": best["score"],
        "best_min_traj_dist": best["min_traj_dist"],
        "best_avg_bi_vuln": best["avg_bi_vuln"],
        "best_reachable": best["reachable"],
        "best_side": best["side"],
        "alpha": float(alpha),
        "dist_limit": float(dist_limit),
        "scan_range": {"min": scan_min, "max": scan_max, "step": scan_step},
        "top5": scan_results[:5],
    }


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bimodal (LiDAR + Visual) spoofer placement selector. "
                    "Extends SLAMSpoof III-C with dual-modality vulnerability analysis."
    )

    parser.add_argument("--smvs", required=True,
                        help="Bimodal SMVS CSV path. "
                             "Required columns: timestamp, x, y, z, yaw, frame_bi_smvs, "
                             "frame_l_smvs, frame_v_smvs, feature_count, sharpness, contrast, "
                             "depth_ratio, v_vul_max, v_vul_mean, vul_angle_deg, vec_x, vec_y")
    parser.add_argument("--vul", required=True,
                        help="Bimodal vulnerability-direction CSV path. "
                             "Required columns: timestamp, x, y, z, vul_angle_deg, vec_x, vec_y, "
                             "frame_bi_smvs, l_vul_00..l_vul_71, v_vul_00..v_vul_71, bi_vul_00..bi_vul_71")

    parser.add_argument("--output", default=None,
                        help="Output JSON path. If omitted, only prints the result.")
    parser.add_argument("--score-column", default="frame_bi_smvs",
                        choices=["frame_bi_smvs", "frame_l_smvs", "frame_v_smvs"],
                        help="Which SMVS column to use for ranking frames. "
                             "Default: frame_bi_smvs (bimodal fusion)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Use top-k high-SMVS frames. Default: 10")
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Only consider frames with score above this threshold. "
                             "For positive SMVS (bimodal frame_bi_smvs): use 0.0 (default). "
                             "For negative SMVS (original paper): use a negative value like -1000.0.")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Outlier removal in standard deviations. Default: 2.0  "
                             "(used only in sigma fallback; RANSAC is used by default)")
    parser.add_argument("--ransac-iterations", type=int, default=200,
                        help="RANSAC iterations for ray intersection. Default: 200")
    parser.add_argument("--ransac-dist-thresh", type=float, default=5.0,
                        help="RANSAC inlier distance threshold (m). Default: 5.0")
    parser.add_argument("--ransac-min-inlier-ratio", type=float, default=0.5,
                        help="Minimum inlier ratio (0-1) to accept RANSAC result. Default: 0.5  "
                             "If fewer than this fraction of rays support the centre, the script exits.")
    parser.add_argument("--angular-range-deg", type=float, default=60.0,
                        help="Angular sector width (degrees) for local trajectory fitting. "
                             "The sector centred at the best theta is selected to contain the most "
                             "trajectory points. Default: 60.0  "
                             "Set to None (0) to disable angular-sector filtering (use all top-k frames).")
    parser.add_argument("--angular-step-deg", type=float, default=5.0,
                        help="Step size (degrees) for scanning angular sector centres. Default: 5.0")
    parser.add_argument("--scf-strength", type=float, default=1.0,
                        help="SCF penalisation strength (0=disable, 1=full, 0.5=partial). "
                             "Default: 1.0  Penalises 'void' frames (low mean/peak LiDAR structure).")
    parser.add_argument("--time-window", type=float, default=None,
                        help="Time window in seconds for filtering frames. Default: None")
    parser.add_argument("--spoof-distance", type=float, default=15.0,
                        help="Distance from trajectory intersection to spoofer (m). Default: 15.0")
    parser.add_argument("--distance-threshold", type=float, default=15.0,
                        help="distance_threshold written to config JSON. Default: 15.0")
    parser.add_argument("--spoofing-range", type=float, default=80.0,
                        help="spoofing_range written to config JSON. Default: 80.0")
    parser.add_argument("--match-mode", choices=["auto", "timestamp", "nearest_xy"],
                        default="timestamp",
                        help="How to match SMVS frames to vulnerability direction rows. Default: timestamp")
    parser.add_argument("--max-match-distance", type=float, default=None)
    parser.add_argument("--line-formula", choices=["paper", "geometric", "both"],
                        default="paper")
    parser.add_argument("--candidate-side", choices=["same_as_center", "positive_normal",
                        "negative_normal", "both"], default="same_as_center")
    parser.add_argument("--scan-mode", choices=["legacy", "traj_constrained"],
                        default="traj_constrained",
                        help="legacy: fixed distance along perpendicular (original). "
                             "traj_constrained: scan along perpendicular, pick best by vuln+reachability. "
                             "Default: traj_constrained")
    parser.add_argument("--scan-min", type=float, default=5.0,
                        help="Min scan distance along perpendicular line (m). Default: 5.0")
    parser.add_argument("--scan-max", type=float, default=80.0,
                        help="Max scan distance along perpendicular line (m). Default: 80.0")
    parser.add_argument("--scan-step", type=float, default=1.0,
                        help="Scan step along perpendicular line (m). Default: 1.0")
    parser.add_argument("--dist-limit", type=float, default=30.0,
                        help="Max trajectory distance for reachability=1 (m). Default: 30.0")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Vulnerability weight in combined score (0=reachability only, 1=vulnerability only). "
                             "Default: 0.5")
    parser.add_argument("--ref-traj", type=str, default=None,
                        help="Reference trajectory CSV (x,y columns) for reachability evaluation. "
                             "Required when --scan-mode traj_constrained.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print output JSON but do not write the file.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_csvs(smvs_path: str, vul_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    smvs_df = pd.read_csv(smvs_path)
    vul_df  = pd.read_csv(vul_path)

    required_smvs = {"timestamp", "x", "y", "z", "frame_bi_smvs",
                     "frame_l_smvs", "frame_v_smvs", "vec_x", "vec_y",
                     "v_vul_max", "v_vul_mean", "feature_count", "sharpness", "contrast"}
    required_vul  = {"x", "y", "z", "vec_x", "vec_y", "frame_bi_smvs"}

    missing_smvs = required_smvs - set(smvs_df.columns)
    missing_vul  = required_vul  - set(vul_df.columns)

    if missing_smvs:
        raise SystemExit(f"SMVS CSV missing columns: {sorted(missing_smvs)}")
    if missing_vul:
        raise SystemExit(f"Vul CSV missing columns: {sorted(missing_vul)}")

    smvs_df["timestamp"] = pd.to_numeric(smvs_df["timestamp"], errors="coerce")
    smvs_df = smvs_df.dropna(subset=["timestamp"])

    for col in ["x", "y", "z", "frame_bi_smvs", "frame_l_smvs",
                "frame_v_smvs", "vec_x", "vec_y", "v_vul_max", "v_vul_mean",
                "feature_count", "sharpness", "contrast"]:
        smvs_df[col] = pd.to_numeric(smvs_df[col], errors="coerce")
    smvs_df = smvs_df.dropna(subset=["x", "y", "z", "frame_bi_smvs", "vec_x", "vec_y"])

    for col in ["x", "y", "z", "vec_x", "vec_y", "frame_bi_smvs"]:
        vul_df[col] = pd.to_numeric(vul_df[col], errors="coerce")
    vul_df = vul_df.dropna(subset=["x", "y", "z", "vec_x", "vec_y", "frame_bi_smvs"])

    # ---- SCF（空间相干因子）计算 ----
    # SCF = mean(l_vul) / peak(l_vul)，惩罚"所有方向都弱"的虚空帧
    l_cols = [c for c in vul_df.columns if c.startswith("l_vul_")]
    if l_cols:
        l_vals = vul_df[l_cols].values.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            scf = np.where(
                l_vals.max(axis=1) > 1e-9,
                l_vals.mean(axis=1) / l_vals.max(axis=1),
                0.0,
            )
        scf = np.nan_to_num(scf, nan=0.0, posinf=0.0, neginf=0.0)
        vul_df = vul_df.copy()
        vul_df["scf"] = scf
        # 把 SCF 加到 smvs_df（按 timestamp 匹配）
        scf_map = dict(zip(vul_df["timestamp"].values, vul_df["scf"].values))
        if "scf" not in smvs_df.columns:
            smvs_df["scf"] = smvs_df["timestamp"].map(scf_map)

    if len(smvs_df) == 0:
        raise SystemExit("SMVS CSV has no valid rows.")
    if len(vul_df) == 0:
        raise SystemExit("Vul CSV has no valid rows.")

    return smvs_df, vul_df


# ---------------------------------------------------------------------------
# 视觉补偿能力统计（替代旧的自适应权重）
# ---------------------------------------------------------------------------

def report_visual_stats(top_k_frames: pd.DataFrame) -> dict:
    """
    从 top-k 高脆弱性帧中统计视觉质量。

    输出：
        - avg_v_vul_max:     平均 V-Vul 峰值（越大 → 视觉越能削弱攻击）
        - avg_v_vul_mean:    平均 V-Vul 均值
        - avg_feature_count: 平均 ORB 特征点数
        - avg_sharpness:     平均清晰度（拉普拉斯方差）
        - avg_contrast:      平均对比度
    """
    for col in ["v_vul_max", "v_vul_mean", "feature_count", "sharpness", "contrast"]:
        if col in top_k_frames.columns:
            top_k_frames[col] = pd.to_numeric(top_k_frames[col], errors="coerce")

    valid = top_k_frames.dropna(subset=["v_vul_max"], how="all")

    def _mean(col):
        if col in valid.columns and len(valid) > 0:
            return float(valid[col].mean())
        return 0.0

    return {
        "avg_v_vul_max":     _mean("v_vul_max"),
        "avg_v_vul_mean":    _mean("v_vul_mean"),
        "avg_feature_count": _mean("feature_count"),
        "avg_sharpness":     _mean("sharpness"),
        "avg_contrast":      _mean("contrast"),
    }


# ---------------------------------------------------------------------------
# 主选择逻辑
# ---------------------------------------------------------------------------

def select_spoofer(
    smvs_df: pd.DataFrame,
    vul_df: pd.DataFrame,
    score_column: str,
    top_k: int,
    score_threshold: float,
    sigma: float,
    spoof_distance: float,
    line_formula: str,
    candidate_side: str,
    match_mode: str,
    max_match_distance: Optional[float],
    verbose: bool,
    time_window: Optional[float] = None,
    ransac_iterations: int = 200,
    ransac_dist_thresh: float = 5.0,
    ransac_min_inlier_ratio: float = 0.5,
    scf_strength: float = 1.0,
    angular_range_deg: float = 60.0,
    angular_step_deg: float = 5.0,
    scan_mode: str = "traj_constrained",
    scan_min: float = 5.0,
    scan_max: float = 80.0,
    scan_step: float = 1.0,
    dist_limit: float = 30.0,
    alpha: float = 0.5,
    ref_traj_df: Optional[pd.DataFrame] = None,
) -> Dict:

    # ---- 0. Angular sector filtering params ----
    angular_range_rad = math.radians(angular_range_deg) if angular_range_deg > 0 else None
    angular_step_rad = math.radians(angular_step_deg)

    # 检测 SMVS 符号约定
    score_series = smvs_df[score_column]
    score_min = float(score_series.min())
    score_max = float(score_series.max())

    # 判断符号约定：正值 = 越高越脆弱；负值 = 越低(绝对值越大)越脆弱
    if score_max > 0 and score_min >= 0:
        sign_mode = "positive"   # 双模态帧级 SMVS（正值，越大越脆弱）
    elif score_max <= 0 and score_min < 0:
        sign_mode = "negative"  # 原版 SLAMSpoof（负值，越小越脆弱 → 用 nsmallest）
    else:
        sign_mode = "mixed"
        print(f"[WARN] SMVS has mixed signs ({score_min:.2f} ~ {score_max:.2f}); "
              f"using nlargest ordering.", file=sys.stderr)

    if sign_mode == "positive" and score_threshold < 0:
        print(f"[WARN] Positive SMVS detected (range {score_min:.2f} ~ {score_max:.2f}) "
              f"but threshold={score_threshold} is negative. "
              f"Consider using --score-threshold 0.0 to filter out non-vulnerable frames.",
              file=sys.stderr)
    elif sign_mode == "negative" and score_threshold >= 0:
        print(f"[WARN] Negative SMVS detected (range {score_min:.2f} ~ {score_max:.2f}) "
              f"but threshold={score_threshold} is >= 0. "
              f"Consider using a negative threshold (e.g. --score-threshold -1000.0).",
              file=sys.stderr)

    # ---- SCF 调整分数排序（方向②：spoofer 可达性惩罚）----
    # 原始 Bi-Vul 只考虑 LiDAR 和 Visual 脆弱性，不考虑 spoofer 是否可达
    # SCF = mean(l_vul) / peak(l_vul) 惩罚"所有方向都弱"的虚空帧
    # 调整后分数 = frame_bi_smvs × (scf_strength × (1 - scf) + scf)
    #   scf_strength=1: 完全惩罚，scf 低的帧分数大幅下降
    #   scf_strength=0: 禁用，scf 不影响排序
    has_scf = "scf" in smvs_df.columns
    if has_scf and scf_strength > 0:
        smvs_df = smvs_df.copy()
        raw_scf = smvs_df["scf"].values.astype(float)
        # SCF = mean(l_vul) / peak(l_vul), penalises frames with weak structure everywhere.
        # adjusted = raw_scf ** scf_strength:
        #   strength=1.0 → adjusted = scf      (full penalisation of void frames)
        #   strength=0.5 → adjusted = sqrt(scf) (partial)
        #   strength=0.0 → adjusted = 1.0       (disabled, no penalisation)
        with np.errstate(divide='ignore', invalid='ignore'):
            adjusted_scf = np.power(raw_scf, scf_strength)
        adjusted_scf = np.clip(adjusted_scf, 0.0, 1.0)
        smvs_df["_adjusted_score"] = smvs_df[score_column].astype(float) * adjusted_scf
        score_col_for_sort = "_adjusted_score"
        if verbose:
            print(f"[SCF] strength={scf_strength}  "
                  f"scf range=[{raw_scf.min():.3f}, {raw_scf.max():.3f}]  "
                  f"adj range=[{adjusted_scf.min():.3f}, {adjusted_scf.max():.3f}]  "
                  f"corr(raw,adj)={smvs_df[[score_column, '_adjusted_score']].corr().iloc[0,1]:.3f}",
                  file=sys.stderr)
    else:
        score_col_for_sort = score_column

    # 阈值过滤：正 SMVS → 取 > threshold；负 SMVS → 取 < threshold（越负越脆弱）
    if sign_mode == "negative":
        candidates = smvs_df[smvs_df[score_column] < score_threshold].copy()
    else:
        candidates = smvs_df[smvs_df[score_column] > score_threshold].copy()

    if len(candidates) == 0:
        print(f"[WARN] No frames with {score_column} {'< ' if sign_mode == 'negative' else '> '}{score_threshold}; "
              f"using all frames.", file=sys.stderr)
        candidates = smvs_df.copy()

    # 排序：正 SMVS → nlargest（越大越脆弱）；负 SMVS → nsmallest（越小越脆弱）
    if sign_mode == "negative":
        candidates = candidates.nsmallest(min(top_k, len(candidates)), score_col_for_sort).reset_index(drop=True)
    else:
        candidates = candidates.nlargest(min(top_k, len(candidates)), score_col_for_sort).reset_index(drop=True)

    # 时间窗口过滤
    if time_window is not None and time_window > 0 and len(candidates) > 0:
        median_t = float(candidates["timestamp"].median())
        candidates = candidates[
            (candidates["timestamp"] >= median_t - time_window) &
            (candidates["timestamp"] <= median_t + time_window)
        ].reset_index(drop=True)
        if len(candidates) < 2:
            if sign_mode == "negative":
                candidates = smvs_df.nsmallest(min(top_k, len(smvs_df)), score_col_for_sort).reset_index(drop=True)
            else:
                candidates = smvs_df.nlargest(min(top_k, len(smvs_df)), score_col_for_sort).reset_index(drop=True)

    # ---- 2. 统计视觉补偿能力 ----
    v_stats = report_visual_stats(candidates)

    # ---- 3. 构建射线 ----
    #    复用原版 select_spoofer_from_smvs_paper.py 的逻辑
    rays: List[orig.Ray] = []
    for _, row in candidates.iterrows():
        # 匹配 vulnerability row（timestamp 或 nearest_xy）
        if match_mode == "timestamp":
            idx = (vul_df["timestamp"] - row["timestamp"]).abs().idxmin()
            matched = vul_df.loc[idx]
            dist = math.hypot(float(matched["x"]) - float(row["x"]),
                              float(matched["y"]) - float(row["y"]))
        else:  # nearest_xy
            dx = vul_df["x"].to_numpy() - float(row["x"])
            dy = vul_df["y"].to_numpy() - float(row["y"])
            dists = np.hypot(dx, dy)
            pos = int(np.argmin(dists))
            matched = vul_df.iloc[pos]
            dist = float(dists[pos])

        vx = float(matched["vec_x"])
        vy = float(matched["vec_y"])
        norm = math.hypot(vx, vy)
        if norm < orig.EPS:
            continue
        vx /= norm
        vy /= norm

        ray = orig.Ray(
            frame_index=int(row.name),
            timestamp=float(row["timestamp"]) if not pd.isna(row["timestamp"]) else None,
            x=float(row["x"]), y=float(row["y"]),
            dx=float(vx), dy=float(vy),
            smvs=float(row[score_column]),
            matched_vul_index=int(matched.name) if hasattr(matched, 'name') else 0,
            matched_distance=dist,
        )
        rays.append(ray)
        if verbose:
            print(f"[RAY] t={ray.timestamp:.1f}s  P=({ray.x:.2f},{ray.y:.2f})  "
                  f"d=({ray.dx:.4f},{ray.dy:.4f})  smvs={ray.smvs:.1f}",
                  file=sys.stderr)

    if len(rays) < 2:
        raise SystemExit("Fewer than two valid rays. Cannot compute intersections.")

    # ---- 4. RANSAC 射线交点（替代 sigma 过滤）----
    # RANSAC 自动剔除 near-parallel outlier 射线对，
    # 找到最大 inlier 集支持的交点，避免方向平行导致的交点飘移
    cx, cy, inlier_indices = _ransac_intersections(
        rays, iterations=ransac_iterations, dist_thresh=ransac_dist_thresh
    )

    if verbose:
        print(f"[RANSAC] centre=({cx:.2f},{cy:.2f})  "
              f"inliers={len(inlier_indices)}/{len(rays)}",
              file=sys.stderr)

    # Guard: reject unreliable results
    inlier_ratio = len(inlier_indices) / len(rays)
    if inlier_ratio < ransac_min_inlier_ratio:
        raise SystemExit(
            f"[ERROR] RANSAC inlier ratio {inlier_ratio:.0%} ({len(inlier_indices)}/{len(rays)}) "
            f"is below threshold {ransac_min_inlier_ratio:.0%} — "
            f"high-score frames do not converge to the same critical object. "
            f"Check your SMVS/vulnerability data or try fewer top-k frames."
        )

    # ---- 5. Angular sector filtering（v2 paper §III-C）----
    # 从 cluster center C 看出去，遍历不同的 θ_center，
    # 选 angular range θ 内包含最多轨迹点的那个方向。
    # angular_range_deg=0 时禁用（用所有 top-k 帧）。
    sector_info: Optional[dict] = None

    if angular_range_rad is not None and angular_range_rad > 0:
        xs = candidates["x"].to_numpy(dtype=float)
        ys = candidates["y"].to_numpy(dtype=float)
        angles = np.arctan2(ys - cy, xs - cx)  # range [-π, π]

        n_frames = len(xs)
        best_count = -1
        best_theta_center = 0.0
        best_mask = np.zeros(n_frames, dtype=bool)

        # 遍历每个可能的扇区中心（-π 到 π，step 步进）
        num_steps = int(math.ceil(2 * math.pi / angular_step_rad))
        for step in range(num_steps):
            theta_center = -math.pi + step * angular_step_rad
            half = angular_range_rad / 2.0
            theta_min = theta_center - half
            theta_max = theta_center + half
            span = theta_max - theta_min

            # 判断每帧是否落在扇区内（vectorized，考虑角度环绕）
            if span <= math.pi:
                mask = (angles >= theta_min) & (angles <= theta_max)
            else:
                mask = (angles >= theta_min) | (angles <= theta_max)

            count = int(np.sum(mask))

            if count > best_count:
                best_count = count
                best_theta_center = theta_center
                best_mask = mask

        # 排除 0 帧的极端情况
        if best_count == 0:
            if verbose:
                print(f"[ANGULAR] no frames in sector (θ_range={angular_range_deg}°); "
                      f"using all {n_frames} frames.", file=sys.stderr)
            best_mask = np.ones(n_frames, dtype=bool)
            best_theta_center = float(np.median(angles))
            best_count = n_frames

        sector_info = {
            "cx": float(cx),
            "cy": float(cy),
            "angular_range_deg": float(angular_range_deg),
            "angular_step_deg": float(angular_step_deg),
            "best_theta_center_deg": float(math.degrees(best_theta_center)),
            "best_count": int(best_count),
            "total_frames": int(n_frames),
            "filtered_frame_indices": [int(i) for i in np.where(best_mask)[0]],
        }

        if verbose:
            print(f"[ANGULAR] best θ_center={math.degrees(best_theta_center):.1f}°  "
                  f"range={angular_range_deg}°  frames_in_sector={best_count}/{n_frames}",
                  file=sys.stderr)

        # 过滤：只用扇区内的帧
        filtered_candidates = candidates.iloc[best_mask].reset_index(drop=True)
        if len(filtered_candidates) < 2:
            print(f"[WARN] Only {len(filtered_candidates)} frames in angular sector; "
                  f"falling back to all top-k frames.", file=sys.stderr)
            filtered_candidates = candidates
            sector_info["filtered_frame_indices"] = list(range(len(candidates)))
            sector_info["fallback"] = True
    else:
        filtered_candidates = candidates
        sector_info = {
            "cx": float(cx),
            "cy": float(cy),
            "angular_range_deg": 0.0,
            "angular_step_deg": float(angular_step_deg),
            "best_theta_center_deg": None,
            "best_count": len(candidates),
            "total_frames": len(candidates),
            "filtered_frame_indices": list(range(len(candidates))),
            "disabled": True,
        }

    # ---- 6. 轨迹直线拟合（只用 angular sector 过滤后的帧）----
    # 原文 §III-C：用扇区内的轨迹点拟合局部直线
    traj_x_arr = filtered_candidates["x"].to_numpy(dtype=float)
    traj_y_arr = filtered_candidates["y"].to_numpy(dtype=float)
    if len(traj_x_arr) < 2:
        raise SystemExit("Need at least two candidate frames to fit trajectory line.")
    if np.std(traj_x_arr) < orig.EPS:
        raise SystemExit("Candidate frames are nearly vertical in x-y; y=mx+n is ill-conditioned.")
    A_traj = np.column_stack([traj_x_arr, np.ones_like(traj_x_arr)])
    m_traj, n_traj = np.linalg.lstsq(A_traj, traj_y_arr, rcond=None)[0]
    traj_line = orig.trajectory_line(float(m_traj), float(n_traj))

    # ---- 7. 过关键物体中心 C 做垂线（angular sector 内的局部直线方向）----
    place_line = orig.perpendicular_line_geometric(float(m_traj), float(cx), float(cy))
    if np.std(traj_x_arr) < orig.EPS:
        place_line = orig.perpendicular_line_geometric(float(m_traj), float(cx), float(cy))

    if verbose:
        print(f"[TRAJ] fitted line: y = {m_traj:.4f}*x + {n_traj:.4f}  (from {len(traj_x_arr)} candidate frames)")
        print(f"[TRAJ] perpendicular through C=({cx:.2f},{cy:.2f}): a={place_line.a:.4f}, b={place_line.b:.4f}, c={place_line.c:.4f}")

    # ---- 8. Spoofer 位置选择 ----
    #    legacy:          固定距离从垂足 H 出发
    #    traj_constrained: 沿垂线扫描，找"脆弱性好 + 轨迹可达"的最优点
    inlier_ray_list = [rays[i] for i in inlier_indices]

    legacy_placement = orig.place_from_trajectory_intersection(
        traj=traj_line,
        place_line=place_line,
        cx=float(cx),
        cy=float(cy),
        distance=spoof_distance,
        candidate_side="same_as_center",
    )
    hx = legacy_placement["trajectory_intersection"]["x"]
    hy = legacy_placement["trajectory_intersection"]["y"]
    p_pos = legacy_placement["candidates"]["positive_normal"]
    p_neg = legacy_placement["candidates"]["negative_normal"]
    selected_legacy = legacy_placement["selected"]

    scan_info: Optional[dict] = None
    if scan_mode == "traj_constrained":
        if ref_traj_df is None or len(ref_traj_df) == 0:
            print("[WARN] --scan-mode traj_constrained requires --ref-traj; falling back to legacy.",
                  file=sys.stderr)
            scan_mode = "legacy"
        elif "x" not in ref_traj_df.columns or "y" not in ref_traj_df.columns:
            print("[WARN] --ref-traj missing x/y columns; falling back to legacy.",
                  file=sys.stderr)
            scan_mode = "legacy"

    if scan_mode == "traj_constrained":
        scan_info = trajectory_constrained_scan(
            critical_x=float(cx),
            critical_y=float(cy),
            traj_m=float(m_traj),
            traj_n=float(n_traj),
            inlier_rays=inlier_ray_list,
            vul_df=vul_df,
            smvs_df=smvs_df,
            traj_df=ref_traj_df,
            scan_min=scan_min,
            scan_max=scan_max,
            scan_step=scan_step,
            dist_limit=dist_limit,
            alpha=alpha,
        )
        if verbose:
            print(f"[SCAN] best spoofer=({scan_info['best_sx']:.2f},{scan_info['best_sy']:.2f})  "
                  f"score={scan_info['best_score']:.4f}  "
                  f"traj_dist={scan_info['best_min_traj_dist']:.2f}m  "
                  f"bi_vuln={scan_info['best_avg_bi_vuln']:.1f}  "
                  f"reachable={scan_info['best_reachable']}  side={scan_info['best_side']}")
            for i, r in enumerate(scan_info["top5"]):
                print(f"  #{i+1}: s=({r['sx']:.1f},{r['sy']:.1f})  "
                      f"score={r['score']:.4f}  dist={r['min_traj_dist']:.1f}m  "
                      f"bi_vuln={r['avg_bi_vuln']:.1f}  reachable={r['reachable']}")

    # 最终选择：优先用扫描结果，否则用 legacy
    if scan_info is not None:
        final_sx, final_sy = scan_info["best_sx"], scan_info["best_sy"]
    else:
        final_sx = float(selected_legacy["spoofer_x"])
        final_sy = float(selected_legacy["spoofer_y"])

    if verbose:
        print(f"[PLACE] foot H=({hx:.2f},{hy:.2f})  "
              f"p_pos=({p_pos['spoofer_x']:.2f},{p_pos['spoofer_y']:.2f})  "
              f"p_neg=({p_neg['spoofer_x']:.2f},{p_neg['spoofer_y']:.2f})  "
              f"selected={'positive_normal' if selected_legacy is p_pos else 'negative_normal' if selected_legacy is p_neg else 'none'}")

    results = {}
    results["trajectory_perpendicular"] = {
        "formula_note": "RANSAC center C + angular-sector filtered trajectory line + perpendicular through C + foot H",
        "angular_sector": sector_info,
        "trajectory_fit": {"m": float(m_traj), "n": float(n_traj), "method": "lstsq_y_mx_n", "frames_used": "angular_sector_filtered"},
        "placement_line_general": orig.asdict(place_line),
        "trajectory_intersection": {"x": float(hx), "y": float(hy)},
        "critical_center": {"x": float(cx), "y": float(cy)},
        "candidates": {
            "positive_normal": p_pos,
            "negative_normal": p_neg,
        },
        "selected": selected_legacy,
        "inlier_ray_count": len(inlier_indices),
    }

    return {
        "method": "bimodal_attack_location_selection",
        "score_column": score_column,
        "avg_v_vul_max": v_stats["avg_v_vul_max"],
        "avg_v_vul_mean": v_stats["avg_v_vul_mean"],
        "spoofer_x": float(final_sx),
        "spoofer_y": float(final_sy),
        "critical_object_center": {"x": float(cx), "y": float(cy)},
        "num_selected_frames": int(len(filtered_candidates)),
        "num_rays": len(rays),
        "num_inlier_rays": len(inlier_indices),
        "ransac_inlier_indices": inlier_indices,
        "intersection_method": "ransac",
        "scf_strength": scf_strength,
        "angular_range_deg": float(angular_range_deg),
        "angular_step_deg": float(angular_step_deg),
        "angular_sector": sector_info,
        "variants": results,
        "scan_info": scan_info,
        "top_frames": [
            {"timestamp": float(r["timestamp"]),
             "x": float(r["x"]), "y": float(r["y"]), "z": float(r["z"]),
             "frame_bi_smvs": float(r["frame_bi_smvs"]),
             "frame_l_smvs": float(r["frame_l_smvs"]) if "frame_l_smvs" in r else float("nan"),
             "frame_v_smvs": float(r["frame_v_smvs"]) if "frame_v_smvs" in r else float("nan"),
             "feature_count": int(float(r["feature_count"])) if "feature_count" in r else 0,
             "sharpness": float(r["sharpness"]) if "sharpness" in r else 0.0,
             "contrast": float(r["contrast"]) if "contrast" in r else 0.0,
             "v_vul_max": float(r["v_vul_max"]) if "v_vul_max" in r else 0.0,
             "v_vul_mean": float(r["v_vul_mean"]) if "v_vul_mean" in r else 0.0,
             "scf": float(r["scf"]) if "scf" in r else float("nan"),
             "_adjusted_score": float(r["_adjusted_score"]) if "_adjusted_score" in r else float(r["frame_bi_smvs"]),
            }
            for _, r in filtered_candidates.iterrows()
        ],
    }


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    smvs_df, vul_df = load_csvs(args.smvs, args.vul)

    # 加载参考轨迹（用于轨迹约束扫描）
    ref_traj_df = None
    if args.ref_traj:
        import os
        if os.path.exists(args.ref_traj):
            ref_traj_df = pd.read_csv(args.ref_traj)
            # 支持 time,x,y,z 或直接 x,y 列
            if "x" not in ref_traj_df.columns and "time" in ref_traj_df.columns:
                ref_traj_df = ref_traj_df.rename(columns={"time": "timestamp"})
            print(f"[INFO] Loaded ref_traj: {len(ref_traj_df)} rows  "
                  f"columns={list(ref_traj_df.columns[:6])}", file=sys.stderr)
        else:
            print(f"[WARN] --ref-traj file not found: {args.ref_traj}", file=sys.stderr)

    result = select_spoofer(
        smvs_df=smvs_df, vul_df=vul_df,
        score_column=args.score_column,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        sigma=args.sigma,
        spoof_distance=args.spoof_distance,
        line_formula=args.line_formula,
        candidate_side=args.candidate_side,
        match_mode=args.match_mode,
        max_match_distance=args.max_match_distance,
        verbose=args.verbose,
        time_window=args.time_window,
        ransac_iterations=args.ransac_iterations,
        ransac_dist_thresh=args.ransac_dist_thresh,
        ransac_min_inlier_ratio=args.ransac_min_inlier_ratio,
        scf_strength=args.scf_strength,
        angular_range_deg=args.angular_range_deg,
        angular_step_deg=args.angular_step_deg,
        scan_mode=args.scan_mode,
        scan_min=args.scan_min,
        scan_max=args.scan_max,
        scan_step=args.scan_step,
        dist_limit=args.dist_limit,
        alpha=args.alpha,
        ref_traj_df=ref_traj_df,
    )

    # distance_threshold: 用扫描到的最近轨迹距离（加一点余量），或固定值
    scan = result.get("scan_info")
    if scan and scan.get("best_reachable"):
        # 动态：最近可达轨迹距离 + 5m 余量
        dyn_threshold = round(scan["best_min_traj_dist"] + 5.0, 1)
        output_data = {
            "spoofer_x": result["spoofer_x"],
            "spoofer_y": result["spoofer_y"],
            "distance_threshold": dyn_threshold,
            "spoofing_range": args.spoofing_range,
            "avg_v_vul_max": result["avg_v_vul_max"],
            "avg_v_vul_mean": result["avg_v_vul_mean"],
            "frame_bi_smvs_threshold": args.score_threshold,
            "selection_info": result,
        }
        print(f"distance_threshold: auto={dyn_threshold}m  (nearest reachable traj dist={scan['best_min_traj_dist']:.1f}m + 5m margin)")
    else:
        output_data = {
            "spoofer_x": result["spoofer_x"],
            "spoofer_y": result["spoofer_y"],
            "distance_threshold": args.distance_threshold,
            "spoofing_range": args.spoofing_range,
            "avg_v_vul_max": result["avg_v_vul_max"],
            "avg_v_vul_mean": result["avg_v_vul_mean"],
            "frame_bi_smvs_threshold": args.score_threshold,
            "selection_info": result,
        }
        if scan:
            print(f"distance_threshold: {args.distance_threshold}m  (WARNING: no reachable position found; scan best traj_dist={scan['best_min_traj_dist']:.1f}m)")

    print("=" * 72)
    print("Bimodal (LiDAR+Visual) spoofer placement selection")
    print("=" * 72)
    print(f"score_column   : {result['score_column']}")
    print(f"avg_v_vul_max  : {result['avg_v_vul_max']:.4f}  "
          f"(视觉补偿能力峰值，越高 → 视觉越能削弱攻击)")
    print(f"avg_v_vul_mean : {result['avg_v_vul_mean']:.4f}")
    print(f"critical center: ({result['critical_object_center']['x']:.4f}, "
          f"{result['critical_object_center']['y']:.4f})")
    sector = result.get("angular_sector", {})
    if sector.get("disabled"):
        print(f"angular sector : disabled (used all {sector.get('total_frames',0)} frames)")
    else:
        print(f"angular sector : θ_center={sector.get('best_theta_center_deg',0):.1f}°  "
              f"range={sector.get('angular_range_deg',0):.1f}°  "
              f"in_sector={sector.get('best_count',0)}/{sector.get('total_frames',0)} frames")
    print(f"selected spoofer: ({result['spoofer_x']}, {result['spoofer_y']})"
          if result['spoofer_x'] is not None else "selected spoofer: not unique")
    scan = result.get("scan_info")
    if scan:
        print(f"trajectory scan  : mode=traj_constrained  alpha={scan['alpha']}  "
              f"dist_limit={scan['dist_limit']}m  "
              f"scan=[{scan['scan_range']['min']}, {scan['scan_range']['max']}] "
              f"step={scan['scan_range']['step']}m")
        print(f"  → best traj_dist={scan['best_min_traj_dist']:.1f}m  "
              f"bi_vuln={scan['best_avg_bi_vuln']:.1f}  "
              f"reachable={scan['best_reachable']}  side={scan['best_side']}")

    if args.output:
        if not args.dry_run:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] wrote {args.output}")
        else:
            print("\n[DRY RUN] Would write:")
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
    else:
        print("\nJSON output:")
        print(json.dumps(output_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
