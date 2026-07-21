#!/usr/bin/env python3
"""Generate final paper tables from completed experiment summaries.

The script is intentionally read-only with respect to experiment directories.
It consumes the consolidated attack_exposure CSVs produced by
summarize_attack_exposure.py plus the clean/GPS sanity-check CSVs, then writes
paper-facing LaTeX tables and raw/trimmed backup statistics under analysis/.
"""

from __future__ import annotations

import csv
import math
import statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


WORKSPACE = Path("/home/qu_menghao/catkin_ws")
SLAMSPOOF = WORKSPACE / "src/slamspoof"
ANALYSIS = SLAMSPOOF / "analysis"
PER_RUN = ANALYSIS / "attack_exposure_per_run.csv"
GROUPED = ANALYSIS / "attack_exposure_grouped.csv"
LVI_CLEAN_SUMMARY = (
    WORKSPACE
    / "src/LVI-SAM/datasets/slamspoof_handheld/repeat_clean/no_attack_1580_x15/summary.csv"
)
HANDHELD_GPS = (
    WORKSPACE / "src/LVI-SAM/datasets/slamspoof_handheld/original/handheld_gps.csv"
)
HANDHELD_CLEAN = (
    WORKSPACE / "src/LVI-SAM/datasets/slamspoof_handheld/original/handheld_original_traj.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: object) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def fmt(mean: float | None, std: float | None, digits: int = 2) -> str:
    if mean is None:
        return "--"
    std = 0.0 if std is None else std
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(stats.mean(values)), float(stats.stdev(values))


def trim_by_ape(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Drop one lowest and one highest APE when enough repeated runs exist."""

    if len(rows) < 5:
        return list(rows)
    indexed = []
    for i, row in enumerate(rows):
        ape = fnum(row.get("ape_rmse"))
        if ape is not None:
            indexed.append((ape, i))
    if len(indexed) < 5:
        return list(rows)
    drop = {min(indexed)[1], max(indexed)[1]}
    return [row for i, row in enumerate(rows) if i not in drop]


def grouped_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    return (
        row["experiment"],
        row["system"],
        row["dataset"],
        row["method"],
        row["mode"],
        row["distance_threshold"],
        row["spoofing_range"],
    )


def summarize_rows(rows: Sequence[dict[str, str]]) -> dict[str, tuple[float | None, float | None]]:
    out: dict[str, tuple[float | None, float | None]] = {}
    for key in [
        "trigger_pct",
        "mean_removed_per_triggered_frame",
        "mean_injected_per_triggered_frame",
        "ape_rmse",
        "rpe_10m_rmse",
        "rpe_max",
    ]:
        vals = [v for v in (fnum(r.get(key)) for r in rows) if v is not None]
        out[key] = mean_std(vals)
    return out


def clean_stats() -> dict[str, tuple[float | None, float | None]]:
    rows = read_csv(LVI_CLEAN_SUMMARY)
    return summarize_rows(
        [
            {
                "ape_rmse": r.get("ape_rmse", ""),
                "rpe_10m_rmse": r.get("rpe_10m_rmse", ""),
                "rpe_max": r.get("rpe_max", ""),
            }
            for r in rows
            if r.get("status", "").startswith("ok")
        ]
    )


def select_group(
    grouped: Sequence[dict[str, str]],
    experiment: str,
    system: str,
    dataset: str,
    method: str,
    mode: str,
    distance_threshold: str,
    spoofing_range: str,
) -> dict[str, str] | None:
    for row in grouped:
        if grouped_key(row) == (
            experiment,
            system,
            dataset,
            method,
            mode,
            distance_threshold,
            spoofing_range,
        ):
            return row
    return None


def row_metric(row: dict[str, str] | None, prefix: str) -> str:
    if row is None:
        return "--"
    return fmt(fnum(row.get(f"{prefix}_mean")), fnum(row.get(f"{prefix}_std")))


def row_trigger(row: dict[str, str] | None) -> str:
    if row is None:
        return "--"
    mean = fnum(row.get("trigger_pct_mean"))
    return "--" if mean is None else f"{mean:.2f}"


def write_table(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_final_tables(grouped: list[dict[str, str]]) -> None:
    clean = clean_stats()

    main_specs = [
        ("Clean", "--", None),
        ("SMVS", "Static", ("main_repeat", "LVI-SAM", "handheld", "smvs", "static", "15", "80")),
        (
            "Bi-SMVS",
            "Static",
            ("main_repeat", "LVI-SAM", "handheld", "bismvs", "static", "15", "80"),
        ),
        ("SMVS", "Removal", ("main_repeat", "LVI-SAM", "handheld", "smvs", "removal", "15", "80")),
        (
            "Bi-SMVS",
            "Removal",
            ("main_repeat", "LVI-SAM", "handheld", "bismvs", "removal", "15", "80"),
        ),
    ]

    lines: list[str] = []
    lines.extend(
        [
            r"% Auto-generated by scripts/generate_final_experiment_tables.py",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Main evaluation on the handheld sequence with LVI-SAM.}",
            r"\label{tab:main_lvisam_results}",
            r"\begin{tabular}{lccc}",
            r"\hline",
            r"Condition & Mode & APE RMSE (m) & RPE-10m RMSE (m) \\",
            r"\hline",
        ]
    )
    for condition, mode, key in main_specs:
        if key is None:
            ape = fmt(*clean["ape_rmse"])
            rpe = fmt(*clean["rpe_10m_rmse"])
        else:
            row = select_group(grouped, *key)
            ape = row_metric(row, "ape_rmse")
            rpe = row_metric(row, "rpe_10m_rmse")
        if condition == "Bi-SMVS" and mode == "Static":
            ape = r"\textbf{" + ape + "}"
        lines.append(f"{condition} & {mode} & {ape} & {rpe} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])

    random_specs = [
        ("Random", "Static", ("main_repeat", "LVI-SAM", "handheld", "random", "static", "15", "80")),
        ("Bi-SMVS", "Static", ("main_repeat", "LVI-SAM", "handheld", "bismvs", "static", "15", "80")),
        ("Random", "Removal", ("main_repeat", "LVI-SAM", "handheld", "random", "removal", "15", "80")),
        (
            "Bi-SMVS",
            "Removal",
            ("main_repeat", "LVI-SAM", "handheld", "bismvs", "removal", "15", "80"),
        ),
    ]
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Comparison with constrained-random spoofing locations. Random locations are sampled without using vulnerability scores.}",
            r"\label{tab:random_baseline}",
            r"\begin{tabular}{lccc}",
            r"\hline",
            r"Condition & Mode & APE RMSE (m) & RPE-10m RMSE (m) \\",
            r"\hline",
        ]
    )
    for condition, mode, key in random_specs:
        row = select_group(grouped, *key)
        ape = row_metric(row, "ape_rmse")
        rpe = row_metric(row, "rpe_10m_rmse")
        if condition == "Bi-SMVS" and mode == "Static":
            ape = r"\textbf{" + ape + "}"
        lines.append(f"{condition} & {mode} & {ape} & {rpe} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])

    ablation_specs = [
        (
            "SMVS + paper placement",
            ("ablation", "LVI-SAM", "handheld", "smvs_paper", "static", "15", "80"),
        ),
        (
            "SMVS + geometric placement",
            ("ablation", "LVI-SAM", "handheld", "smvs_geometric", "static", "15", "80"),
        ),
        (
            "Bi-SMVS + paper placement",
            ("ablation", "LVI-SAM", "handheld", "bismvs_paper", "static", "15", "80"),
        ),
        (
            "Bi-SMVS + CMA",
            ("ablation", "LVI-SAM", "handheld", "bismvs_cma_no_graph", "static", "15", "80"),
        ),
        (
            "Bi-SMVS + graph-aware CMA",
            ("ablation", "LVI-SAM", "handheld", "bismvs_graph_cma", "static", "15", "80"),
        ),
    ]
    lines.extend(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Ablation study on the LVI-SAM handheld sequence.}",
            r"\label{tab:ablation_lvisam}",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"Condition & Trigger (\%) & APE RMSE (m) & RPE-10m RMSE (m) & RPE-max (m) \\",
            r"\hline",
        ]
    )
    for label, key in ablation_specs:
        row = select_group(grouped, *key)
        trig = row_trigger(row)
        ape = row_metric(row, "ape_rmse")
        rpe = row_metric(row, "rpe_10m_rmse")
        rpemax = row_metric(row, "rpe_max")
        if label == "Bi-SMVS + graph-aware CMA":
            ape = r"\textbf{" + ape + "}"
            rpe = r"\textbf{" + rpe + "}"
        lines.append(f"{label} & {trig} & {ape} & {rpe} & {rpemax} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])

    transfer_specs = [
        ("FAST-LIVO2", "Injection", ("transfer", "FAST-LIVO2", "CBD_Building_01", "bismvs", "static", "15", "40")),
        ("FAST-LIVO2", "Removal", ("transfer", "FAST-LIVO2", "CBD_Building_01", "bismvs", "removal", "15", "40")),
        ("R3LIVE", "Injection", ("transfer", "R3LIVE", "hku_campus_seq_00", "bismvs", "static", "15", "40")),
        ("R3LIVE", "Removal", ("transfer", "R3LIVE", "hku_campus_seq_00", "bismvs", "removal", "15", "40")),
    ]
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Supplementary transferability evaluation of Bi-SMVS-guided attacks on additional LiDAR--visual SLAM systems.}",
            r"\label{tab:transferability}",
            r"\begin{tabular}{lccc}",
            r"\hline",
            r"System & Mode & APE RMSE (m) & RPE-10m RMSE (m) \\",
            r"\hline",
        ]
    )
    for system, mode, key in transfer_specs:
        row = select_group(grouped, *key)
        lines.append(
            f"{system} & {mode} & {row_metric(row, 'ape_rmse')} & {row_metric(row, 'rpe_10m_rmse')} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])

    exposure_specs = [
        ("SMVS", "Static", ("main_repeat", "LVI-SAM", "handheld", "smvs", "static", "15", "80")),
        ("Bi-SMVS", "Static", ("main_repeat", "LVI-SAM", "handheld", "bismvs", "static", "15", "80")),
        ("Random", "Static", ("main_repeat", "LVI-SAM", "handheld", "random", "static", "15", "80")),
        ("SMVS", "Removal", ("main_repeat", "LVI-SAM", "handheld", "smvs", "removal", "15", "80")),
        ("Bi-SMVS", "Removal", ("main_repeat", "LVI-SAM", "handheld", "bismvs", "removal", "15", "80")),
        ("Random", "Removal", ("main_repeat", "LVI-SAM", "handheld", "random", "removal", "15", "80")),
    ]
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Attack exposure statistics for the main LVI-SAM evaluation.}",
            r"\label{tab:main_attack_exposure}",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"Condition & Mode & Trigger (\%) & Removed pts/frame & Injected pts/frame \\",
            r"\hline",
        ]
    )
    for condition, mode, key in exposure_specs:
        row = select_group(grouped, *key)
        removed = row_metric(row, "mean_removed_per_triggered_frame")
        injected = row_metric(row, "mean_injected_per_triggered_frame")
        lines.append(f"{condition} & {mode} & {row_trigger(row)} & {removed} & {injected} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])

    write_table(ANALYSIS / "final_results_tables.tex", lines)


def build_raw_trimmed(per_run: list[dict[str, str]]) -> None:
    groups: dict[tuple[str, str, str, str, str, str, str], list[dict[str, str]]] = {}
    for row in per_run:
        if not row.get("status", "").startswith("ok"):
            continue
        groups.setdefault(grouped_key(row), []).append(row)

    out = ANALYSIS / "final_results_raw_trimmed.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "system",
                "dataset",
                "method",
                "mode",
                "distance_threshold",
                "spoofing_range",
                "n_raw",
                "n_trimmed",
                "raw_ape_mean",
                "raw_ape_std",
                "trimmed_ape_mean",
                "trimmed_ape_std",
                "raw_rpe10_mean",
                "raw_rpe10_std",
                "trimmed_rpe10_mean",
                "trimmed_rpe10_std",
                "raw_rpemax_mean",
                "raw_rpemax_std",
                "trimmed_rpemax_mean",
                "trimmed_rpemax_std",
                "trim_rule",
            ],
        )
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            trimmed = trim_by_ape(rows)
            raw = summarize_rows(rows)
            trm = summarize_rows(trimmed)
            writer.writerow(
                {
                    "experiment": key[0],
                    "system": key[1],
                    "dataset": key[2],
                    "method": key[3],
                    "mode": key[4],
                    "distance_threshold": key[5],
                    "spoofing_range": key[6],
                    "n_raw": len(rows),
                    "n_trimmed": len(trimmed),
                    "raw_ape_mean": raw["ape_rmse"][0],
                    "raw_ape_std": raw["ape_rmse"][1],
                    "trimmed_ape_mean": trm["ape_rmse"][0],
                    "trimmed_ape_std": trm["ape_rmse"][1],
                    "raw_rpe10_mean": raw["rpe_10m_rmse"][0],
                    "raw_rpe10_std": raw["rpe_10m_rmse"][1],
                    "trimmed_rpe10_mean": trm["rpe_10m_rmse"][0],
                    "trimmed_rpe10_std": trm["rpe_10m_rmse"][1],
                    "raw_rpemax_mean": raw["rpe_max"][0],
                    "raw_rpemax_std": raw["rpe_max"][1],
                    "trimmed_rpemax_mean": trm["rpe_max"][0],
                    "trimmed_rpemax_std": trm["rpe_max"][1],
                    "trim_rule": "drop min and max APE if n>=5; otherwise unchanged",
                }
            )


def read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    times = []
    for row in read_csv(path):
        t = fnum(row.get("time"))
        x = fnum(row.get("x"))
        y = fnum(row.get("y"))
        if t is not None and x is not None and y is not None:
            times.append(t)
            rows.append((x, y))
    return np.asarray(times, dtype=np.float64), np.asarray(rows, dtype=np.float64)


def path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def align_similarity_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    x = src - src_mean
    y = dst - dst_mean
    cov = (x.T @ y) / max(len(src), 1)
    u, s, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    var = float(np.sum(x * x) / max(len(src), 1))
    scale = float(np.sum(s) / max(var, 1e-12))
    aligned = scale * (x @ r.T) + dst_mean
    return aligned, scale


def build_gps_sanity() -> None:
    gps_t, gps_xy = read_xy(HANDHELD_GPS)
    clean_t, clean_xy = read_xy(HANDHELD_CLEAN)
    if not len(gps_t) or not len(clean_t):
        raise RuntimeError("missing GPS or clean trajectory rows")

    if abs(gps_t[0] - clean_t[0]) > 1000:
        target_t = (clean_t - clean_t[0]) / (clean_t[-1] - clean_t[0]) * (
            gps_t[-1] - gps_t[0]
        ) + gps_t[0]
    else:
        target_t = np.clip(clean_t, gps_t[0], gps_t[-1])
    gps_interp = np.column_stack(
        [
            np.interp(target_t, gps_t, gps_xy[:, 0]),
            np.interp(target_t, gps_t, gps_xy[:, 1]),
        ]
    )
    aligned, scale = align_similarity_2d(clean_xy, gps_interp)
    err = np.linalg.norm(aligned - gps_interp, axis=1)

    out_csv = ANALYSIS / "gps_sanity_handheld.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "gps_rows",
                "clean_rows",
                "gps_path_length_m",
                "clean_path_length_m",
                "alignment_scale",
                "aligned_xy_rmse_m",
                "aligned_xy_mean_m",
                "aligned_xy_max_m",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "handheld",
                "gps_rows": len(gps_xy),
                "clean_rows": len(clean_xy),
                "gps_path_length_m": path_length(gps_xy),
                "clean_path_length_m": path_length(clean_xy),
                "alignment_scale": scale,
                "aligned_xy_rmse_m": float(np.sqrt(np.mean(err * err))),
                "aligned_xy_mean_m": float(np.mean(err)),
                "aligned_xy_max_m": float(np.max(err)),
            }
        )

    lines = [
        r"% Auto-generated by scripts/generate_final_experiment_tables.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{GPS sanity check for the clean LVI-SAM trajectory on the handheld sequence.}",
        r"\label{tab:gps_sanity}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Sequence & Scale & XY RMSE (m) & XY Max (m) \\",
        r"\hline",
        f"handheld & {scale:.3f} & {float(np.sqrt(np.mean(err * err))):.2f} & {float(np.max(err)):.2f} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    write_table(ANALYSIS / "gps_sanity_table.tex", lines)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    grouped = read_csv(GROUPED)
    per_run = read_csv(PER_RUN)
    build_final_tables(grouped)
    build_raw_trimmed(per_run)
    build_gps_sanity()
    print(f"[OK] wrote {ANALYSIS / 'final_results_tables.tex'}")
    print(f"[OK] wrote {ANALYSIS / 'final_results_raw_trimmed.csv'}")
    print(f"[OK] wrote {ANALYSIS / 'gps_sanity_handheld.csv'}")
    print(f"[OK] wrote {ANALYSIS / 'gps_sanity_table.tex'}")


if __name__ == "__main__":
    main()
