#!/usr/bin/env python3
"""
select_spoofer_from_bimodal.py
================================

基于双模态（LiDAR + Visual）脆弱性的 spoofer 位置选择。

融合公式：
  Bi-Vul[k] = L-Vul[k] × (1 - V-Vul[k])
  V-Vul[k] = γ × cam_coverage[k] × Q
  Q = 0.30 × track_ratio + 0.40 × parallax_ratio + 0.30 × depth_ratio

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
                             "Required columns: timestamp, x, y, z, frame_bi_smvs, "
                             "frame_l_smvs, frame_v_smvs, last_track_num, avg_parallax, "
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
                        help="Outlier removal in standard deviations. Default: 2.0")
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
) -> Dict:

    # ---- 1. 按双模态 SMVS 排序选择高脆弱性帧 ----
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
        candidates = candidates.nsmallest(min(top_k, len(candidates)), score_column).reset_index(drop=True)
    else:
        candidates = candidates.nlargest(min(top_k, len(candidates)), score_column).reset_index(drop=True)

    # 时间窗口过滤
    if time_window is not None and time_window > 0 and len(candidates) > 0:
        median_t = float(candidates["timestamp"].median())
        candidates = candidates[
            (candidates["timestamp"] >= median_t - time_window) &
            (candidates["timestamp"] <= median_t + time_window)
        ].reset_index(drop=True)
        if len(candidates) < 2:
            if sign_mode == "negative":
                candidates = smvs_df.nsmallest(min(top_k, len(smvs_df)), score_column).reset_index(drop=True)
            else:
                candidates = smvs_df.nlargest(min(top_k, len(smvs_df)), score_column).reset_index(drop=True)

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

        vx = -float(matched["vec_x"])
        vy = -float(matched["vec_y"])
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

    # ---- 4. 射线交点 ----
    rows_list = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            out = orig.intersect_rays(rays[i], rays[j])
            if out is None:
                continue
            ix, iy, t1, t2 = out
            rows_list.append({
                "x": ix, "y": iy,
                "ray_i": i, "ray_j": j,
                "frame_i": rays[i].frame_index,
                "frame_j": rays[j].frame_index,
                "t_i": t1, "t_j": t2,
            })
    if not rows_list:
        raise SystemExit("No valid half-line intersections found.")
    intersections = pd.DataFrame(rows_list)

    # ---- 5. Sigma 过滤离群点 ----
    x_arr = intersections["x"].to_numpy()
    y_arr = intersections["y"].to_numpy()
    mean_x, mean_y = float(np.mean(x_arr)), float(np.mean(y_arr))
    std_x  = float(np.std(x_arr))
    std_y  = float(np.std(y_arr))

    x_ok = np.ones_like(x_arr, dtype=bool) if std_x < orig.EPS else np.abs(x_arr - mean_x) <= sigma * std_x
    y_ok = np.ones_like(y_arr, dtype=bool) if std_y < orig.EPS else np.abs(y_arr - mean_y) <= sigma * std_y
    filtered = intersections.loc[x_ok & y_ok].copy()
    if len(filtered) == 0:
        print("[WARN] Sigma filter removed all intersections; using unfiltered.", file=sys.stderr)
        filtered = intersections

    # ---- 6. 临界物体中心 ----
    cx = float(filtered["x"].mean())
    cy = float(filtered["y"].mean())

    # ---- 7. 轨迹线性拟合 ----
    traj_x = candidates["x"].to_numpy(dtype=float)
    traj_y = candidates["y"].to_numpy(dtype=float)
    if len(traj_x) < 2 or np.std(traj_x) < orig.EPS:
        raise SystemExit("Cannot fit trajectory line: insufficient or degenerate data.")
    A = np.column_stack([traj_x, np.ones_like(traj_x)])
    m, n = np.linalg.lstsq(A, traj_y, rcond=None)[0]
    m, n = float(m), float(n)

    # ---- 8. 垂直线 + spoofer 位置 ----
    traj_line = orig.trajectory_line(m, n)

    if line_formula in ("paper", "both"):
        place_line_paper = orig.perpendicular_line_paper(m, cx, cy)
    if line_formula in ("geometric", "both"):
        place_line_geom = orig.perpendicular_line_geometric(m, cx, cy)

    results = {}
    for variant in ("paper", "geometric"):
        if line_formula == "both" and variant == "both":
            continue
        if line_formula != "both" and line_formula != variant:
            continue

        pl = place_line_paper if variant == "paper" else place_line_geom

        # 轨迹线与放置线的交点
        det = traj_line.a * pl.b - pl.a * traj_line.b
        if abs(det) < orig.EPS:
            continue
        hx = ((-traj_line.c) * pl.b - (-pl.c) * traj_line.b) / det
        hy = (traj_line.a * (-pl.c) - pl.a * (-traj_line.c)) / det

        # 放置线方向向量
        d_perp = np.array([pl.b, -pl.a], dtype=float)
        d_norm = np.linalg.norm(d_perp)
        if d_norm < orig.EPS:
            continue
        d_perp /= d_norm

        # 两个候选位置
        p_pos = np.array([hx, hy]) + spoof_distance * d_perp
        p_neg = np.array([hx, hy]) - spoof_distance * d_perp

        # 选择在临界物体中心同侧的那个
        # 临界物体在轨迹的哪一侧？
        traj_val_center = traj_line.a * cx + traj_line.b * cy + traj_line.c
        pos_side = traj_line.a * p_pos[0] + traj_line.b * p_pos[1] + traj_line.c
        neg_side = traj_line.a * p_neg[0] + traj_line.b * p_neg[1] + traj_line.c

        if traj_val_center * pos_side >= 0:
            selected = {"spoofer_x": float(p_pos[0]), "spoofer_y": float(p_pos[1])}
        elif traj_val_center * neg_side >= 0:
            selected = {"spoofer_x": float(p_neg[0]), "spoofer_y": float(p_neg[1])}
        else:
            selected = None

        formula_note = ("printed Eq.(7): g(x)=-1/m*x+(-m*Cx+Cy)"
                        if variant == "paper"
                        else "geometrically through C: g(x)=-1/m*x+(Cy+Cx/m)")

        results[variant] = {
            "formula_note": formula_note,
            "trajectory_intersection": {"x": float(hx), "y": float(hy)},
            "critical_center": {"x": float(cx), "y": float(cy)},
            "trajectory_fit": {"m": m, "n": n, "model": "y=m*x+n"},
            "selected": selected,
            "candidates": {
                "positive_normal": {"spoofer_x": float(p_pos[0]), "spoofer_y": float(p_pos[1])},
                "negative_normal": {"spoofer_x": float(p_neg[0]), "spoofer_y": float(p_neg[1])},
            },
        }

    primary = results.get("paper") or results.get("geometric") or {}
    sel = primary.get("selected") or {}

    return {
        "method": "bimodal_attack_location_selection",
        "score_column": score_column,
        "avg_v_vul_max": v_stats["avg_v_vul_max"],
        "avg_v_vul_mean": v_stats["avg_v_vul_mean"],
        "spoofer_x": sel.get("spoofer_x"),
        "spoofer_y": sel.get("spoofer_y"),
        "critical_object_center": {"x": float(cx), "y": float(cy)},
        "num_selected_frames": int(len(candidates)),
        "num_rays": len(rays),
        "num_intersections": int(len(intersections)),
        "num_filtered_intersections": int(len(filtered)),
        "variants": results,
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
            }
            for _, r in candidates.iterrows()
        ],
    }


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    smvs_df, vul_df = load_csvs(args.smvs, args.vul)

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
    )

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

    print("=" * 72)
    print("Bimodal (LiDAR+Visual) spoofer placement selection")
    print("=" * 72)
    print(f"score_column   : {result['score_column']}")
    print(f"avg_v_vul_max  : {result['avg_v_vul_max']:.4f}  "
          f"(视觉补偿能力峰值，越高 → 视觉越能削弱攻击)")
    print(f"avg_v_vul_mean : {result['avg_v_vul_mean']:.4f}")
    print(f"critical center: ({result['critical_object_center']['x']:.4f}, "
          f"{result['critical_object_center']['y']:.4f})")
    print(f"selected spoofer: ({result['spoofer_x']}, {result['spoofer_y']})"
          if result['spoofer_x'] is not None else "selected spoofer: not unique")

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
