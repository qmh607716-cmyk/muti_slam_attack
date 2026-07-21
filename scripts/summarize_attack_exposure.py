#!/usr/bin/env python3
"""Summarize attack exposure and evaluation metrics from repeated runs."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


RE_TOTAL = re.compile(r"Total frames\s*:\s*(\d+)")
RE_TRIGGER = re.compile(r"Triggered frames\s*:\s*(\d+)\s*\(?\s*([0-9.]+)?%?")
RE_MEAN_REMOVED = re.compile(r"Mean removed/frame\s*:\s*([0-9.]+)")
RE_MAX_REMOVED = re.compile(r"Max removed \(frame\)\s*:\s*(\d+)")
RE_MEAN_INJECTED = re.compile(r"Mean injected/frame\s*:\s*([0-9.]+)")
RE_MAX_INJECTED = re.compile(r"Max injected \(frame\)\s*:\s*(\d+)")
RE_OUTPUT_BAG = re.compile(r"Output bag\s*:\s*(.+)")


FIELDNAMES = [
    "experiment",
    "system",
    "dataset",
    "run",
    "method",
    "mode",
    "distance_threshold",
    "spoofing_range",
    "spoofer_x",
    "spoofer_y",
    "status",
    "total_frames",
    "triggered_frames",
    "trigger_pct",
    "mean_removed_per_triggered_frame",
    "max_removed_frame",
    "mean_injected_per_triggered_frame",
    "max_injected_frame",
    "ape_rmse",
    "rpe_10m_rmse",
    "rpe_max",
    "summary_path",
    "run_dir",
    "attack_log",
    "output_bag_from_log",
]


GROUP_FIELDNAMES = [
    "experiment",
    "system",
    "dataset",
    "method",
    "mode",
    "distance_threshold",
    "spoofing_range",
    "n_runs",
    "trigger_pct_mean",
    "trigger_pct_std",
    "mean_removed_per_triggered_frame_mean",
    "mean_removed_per_triggered_frame_std",
    "mean_injected_per_triggered_frame_mean",
    "mean_injected_per_triggered_frame_std",
    "ape_rmse_mean",
    "ape_rmse_std",
    "rpe_10m_rmse_mean",
    "rpe_10m_rmse_std",
    "rpe_max_mean",
    "rpe_max_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect trigger ratio, modified-point statistics, APE, RPE-10m, "
            "and RPE-max from LVI-SAM/FAST-LIVO2/R3LIVE repeated attacks."
        )
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / "catkin_ws",
        help="catkin workspace root",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        action="append",
        default=[],
        help="Extra summary.csv path. Can be specified multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "catkin_ws" / "src" / "slamspoof" / "analysis",
        help="Output directory for generated CSV/LaTeX files.",
    )
    parser.add_argument(
        "--include-old-lvisam",
        action="store_true",
        help="Also include older LVI-SAM non-x15 and 30/80 summary files.",
    )
    return parser.parse_args()


def default_summary_files(workspace: Path, include_old_lvisam: bool) -> List[Path]:
    summaries: List[Path] = []

    lvi_root = workspace / "src" / "LVI-SAM" / "datasets" / "slamspoof_handheld"
    if include_old_lvisam:
        summaries.extend(sorted(lvi_root.glob("repeat_*/*/summary.csv")))
    else:
        for mode_dir in ("repeat_static", "repeat_removal"):
            summaries.extend(sorted((lvi_root / mode_dir).glob("*_1580_x15/summary.csv")))
        summaries.extend(sorted((lvi_root / "param_sweep").glob("**/summary.csv")))
        summaries.extend(sorted((lvi_root / "ablation").glob("**/summary.csv")))

    fast_summary = (
        workspace
        / "datasets"
        / "official"
        / "fast_livo2"
        / "experiments_CBD_Building_01"
        / "summary.csv"
    )
    r3_summary = workspace / "datasets" / "official" / "r3live" / "experiments" / "summary.csv"
    summaries.extend([fast_summary, r3_summary])

    seen = set()
    existing: List[Path] = []
    for path in summaries:
        path = path.expanduser().resolve()
        if path.exists() and path not in seen:
            existing.append(path)
            seen.add(path)
    return existing


def fnum(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def read_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            yield row


def infer_system_dataset(summary_path: Path, row: Dict[str, str]) -> Tuple[str, str]:
    text = str(summary_path)
    input_bag = row.get("input_bag") or row.get("attack_bag") or ""

    if "fast_livo2" in text:
        return "FAST-LIVO2", "CBD_Building_01"
    if "r3live" in text:
        return "R3LIVE", "hku_campus_seq_00"
    if "slamspoof_handheld" in text:
        return "LVI-SAM", "handheld"

    platform = row.get("platform", "").strip()
    return platform or "unknown", Path(input_bag).stem if input_bag else "unknown"


def infer_experiment(summary_path: Path) -> str:
    text = str(summary_path)
    if "ablation" in text:
        return "ablation"
    if "param_sweep" in text:
        return "param_sweep"
    if "slamspoof_handheld" in text:
        return "main_repeat"
    if "official" in text:
        return "transfer"
    return "unknown"


def infer_run_dir(summary_path: Path, row: Dict[str, str]) -> Path:
    eval_dir = row.get("eval_dir", "").strip()
    if eval_dir:
        return Path(eval_dir).expanduser().resolve().parent

    run = row.get("run", "").strip()
    if "slamspoof_handheld" in str(summary_path) and run:
        return summary_path.parent / "runs" / f"run_{run}"
    return summary_path.parent / "runs" / run


def parse_attack_log(log_path: Path) -> Dict[str, Optional[float]]:
    values: Dict[str, Optional[float]] = {
        "total_frames": None,
        "triggered_frames": None,
        "trigger_pct": None,
        "mean_removed_per_triggered_frame": None,
        "max_removed_frame": None,
        "mean_injected_per_triggered_frame": None,
        "max_injected_frame": None,
        "output_bag_from_log": None,
    }
    if not log_path.exists():
        return values

    text = log_path.read_text(errors="replace")

    match = RE_TOTAL.search(text)
    if match:
        values["total_frames"] = float(match.group(1))

    match = RE_TRIGGER.search(text)
    if match:
        values["triggered_frames"] = float(match.group(1))
        if match.group(2) is not None:
            values["trigger_pct"] = float(match.group(2))

    for key, regex in (
        ("mean_removed_per_triggered_frame", RE_MEAN_REMOVED),
        ("max_removed_frame", RE_MAX_REMOVED),
        ("mean_injected_per_triggered_frame", RE_MEAN_INJECTED),
        ("max_injected_frame", RE_MAX_INJECTED),
    ):
        match = regex.search(text)
        if match:
            values[key] = float(match.group(1))

    match = RE_OUTPUT_BAG.search(text)
    if match:
        values["output_bag_from_log"] = match.group(1).strip()

    if values["trigger_pct"] is None:
        total = values["total_frames"]
        triggered = values["triggered_frames"]
        if total and triggered is not None:
            values["trigger_pct"] = triggered / total * 100.0

    return values


def status_is_usable(status: str) -> bool:
    return status.strip().startswith("ok")


def collect_rows(summary_paths: Sequence[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary_path in summary_paths:
        raw_rows = list(read_csv_rows(summary_path))
        rows_by_run: Dict[str, Dict[str, str]] = {}
        for row in raw_rows:
            run_key = row.get("run", "").strip()
            if not run_key:
                continue
            # Some transfer summaries append re-evaluation rows. Keep the last
            # row for each run because it reflects the most recent metrics.
            rows_by_run[run_key] = row

        for row in rows_by_run.values():
            method = (row.get("method") or "").strip()
            mode = (row.get("mode") or "").strip()
            status = (row.get("status") or "").strip()
            if not method or not mode:
                continue
            if mode == "clean" or method in {"fast_livo2", "r3live"}:
                continue

            run_dir = infer_run_dir(summary_path, row)
            attack_log = run_dir / "01_generate_attacked_bag.log"
            exposure = parse_attack_log(attack_log)
            system, dataset = infer_system_dataset(summary_path, row)

            out: Dict[str, object] = {
                "experiment": infer_experiment(summary_path),
                "system": system,
                "dataset": dataset,
                "run": row.get("run", ""),
                "method": method,
                "mode": mode,
                "distance_threshold": row.get("distance_threshold", ""),
                "spoofing_range": row.get("spoofing_range", ""),
                "spoofer_x": row.get("spoofer_x", ""),
                "spoofer_y": row.get("spoofer_y", ""),
                "status": status,
                "ape_rmse": fnum(row.get("ape_rmse")),
                "rpe_10m_rmse": fnum(row.get("rpe_10m_rmse")),
                "rpe_max": fnum(row.get("rpe_max")),
                "summary_path": str(summary_path),
                "run_dir": str(run_dir),
                "attack_log": str(attack_log),
            }
            out.update(exposure)
            rows.append(out)
    return rows


def mean_std(values: Iterable[object]) -> Tuple[Optional[float], Optional[float]]:
    nums = [fnum(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return None, None
    if len(nums) == 1:
        return nums[0], 0.0
    return statistics.mean(nums), statistics.stdev(nums)


def normalize_param(value: object) -> str:
    number = fnum(value)
    if number is None:
        return str(value).strip()
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:g}"


def group_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not status_is_usable(str(row.get("status", ""))):
            continue
        key = (
            str(row.get("experiment", "")),
            str(row.get("system", "")),
            str(row.get("dataset", "")),
            str(row.get("method", "")),
            str(row.get("mode", "")),
            normalize_param(row.get("distance_threshold", "")),
            normalize_param(row.get("spoofing_range", "")),
        )
        groups[key].append(row)

    out: List[Dict[str, object]] = []
    for key, items in groups.items():
        experiment, system, dataset, method, mode, dist, spoof_range = key
        grouped: Dict[str, object] = {
            "experiment": experiment,
            "system": system,
            "dataset": dataset,
            "method": method,
            "mode": mode,
            "distance_threshold": dist,
            "spoofing_range": spoof_range,
            "n_runs": len(items),
        }
        for field in (
            "trigger_pct",
            "mean_removed_per_triggered_frame",
            "mean_injected_per_triggered_frame",
            "ape_rmse",
            "rpe_10m_rmse",
            "rpe_max",
        ):
            mean, std = mean_std(item.get(field) for item in items)
            grouped[f"{field}_mean"] = mean
            grouped[f"{field}_std"] = std
        out.append(grouped)

    def sort_key(row: Dict[str, object]) -> Tuple[int, int, str, int, int, float, float]:
        experiment_order = {"main_repeat": 0, "param_sweep": 1, "ablation": 2, "transfer": 3}
        system_order = {"LVI-SAM": 0, "FAST-LIVO2": 1, "R3LIVE": 2}
        method_order = {
            "smvs": 0,
            "smvs_paper": 1,
            "bismvs_paper": 2,
            "bismvs_cma_no_graph": 3,
            "bismvs_graph_cma": 4,
            "bismvs": 5,
            "random": 6,
        }
        mode_order = {"static": 0, "removal": 1}
        return (
            experiment_order.get(str(row["experiment"]), 99),
            system_order.get(str(row["system"]), 99),
            str(row["dataset"]),
            method_order.get(str(row["method"]), 99),
            mode_order.get(str(row["mode"]), 99),
            fnum(row["distance_threshold"]) or 0.0,
            fnum(row["spoofing_range"]) or 0.0,
        )

    return sorted(out, key=sort_key)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_number(value: object, digits: int = 2) -> str:
    number = fnum(value)
    if number is None:
        return "--"
    return f"{number:.{digits}f}"


def fmt_pm(mean: object, std: object, digits: int = 2) -> str:
    if fnum(mean) is None:
        return "--"
    return f"{fmt_number(mean, digits)} $\\pm$ {fmt_number(std, digits)}"


def latex_method(method: str) -> str:
    return {
        "bismvs": "Bi-SMVS",
        "smvs": "SMVS",
        "random": "Random",
        "smvs_paper": "SMVS+paper",
        "bismvs_paper": "Bi-SMVS+paper",
        "bismvs_cma_no_graph": "Bi-SMVS+CMA",
        "bismvs_graph_cma": "Bi-SMVS+graph+CMA",
    }.get(method, method)


def latex_mode(mode: str) -> str:
    return {"static": "Static", "removal": "Removal"}.get(mode, mode)


def latex_escape(text: object) -> str:
    return str(text).replace("_", "\\_")


def write_latex_table(path: Path, grouped: Sequence[Dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Attack exposure statistics measured during attacked-bag generation.}",
        r"\label{tab:attack_exposure_stats}",
        r"\begin{tabular}{lllccccc}",
        r"\hline",
        r"Experiment & System & Condition & Mode & D/R & Triggered (\%) & Removed pts/frame & Injected pts/frame \\",
        r"\hline",
    ]
    for row in grouped:
        dist = fmt_number(row["distance_threshold"], 0)
        spoof_range = fmt_number(row["spoofing_range"], 0)
        lines.append(
            " & ".join(
                [
                    latex_escape(row["experiment"]),
                    latex_escape(row["system"]),
                    latex_method(str(row["method"])),
                    latex_mode(str(row["mode"])),
                    f"{dist}/{spoof_range}",
                    fmt_pm(row["trigger_pct_mean"], row["trigger_pct_std"], 2),
                    fmt_pm(
                        row["mean_removed_per_triggered_frame_mean"],
                        row["mean_removed_per_triggered_frame_std"],
                        1,
                    ),
                    fmt_pm(
                        row["mean_injected_per_triggered_frame_mean"],
                        row["mean_injected_per_triggered_frame_std"],
                        1,
                    ),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def print_console_summary(grouped: Sequence[Dict[str, object]]) -> None:
    print("experiment,system,dataset,method,mode,D/R,n,trigger_pct,removed_pts,injected_pts,APE,RPE10,RPEmax")
    for row in grouped:
        dist = fmt_number(row["distance_threshold"], 0)
        spoof_range = fmt_number(row["spoofing_range"], 0)
        print(
            ",".join(
                [
                    str(row["experiment"]),
                    str(row["system"]),
                    str(row["dataset"]),
                    str(row["method"]),
                    str(row["mode"]),
                    f"{dist}/{spoof_range}",
                    str(row["n_runs"]),
                    fmt_pm(row["trigger_pct_mean"], row["trigger_pct_std"], 2),
                    fmt_pm(
                        row["mean_removed_per_triggered_frame_mean"],
                        row["mean_removed_per_triggered_frame_std"],
                        1,
                    ),
                    fmt_pm(
                        row["mean_injected_per_triggered_frame_mean"],
                        row["mean_injected_per_triggered_frame_std"],
                        1,
                    ),
                    fmt_pm(row["ape_rmse_mean"], row["ape_rmse_std"], 2),
                    fmt_pm(row["rpe_10m_rmse_mean"], row["rpe_10m_rmse_std"], 2),
                    fmt_pm(row["rpe_max_mean"], row["rpe_max_std"], 2),
                ]
            )
        )


def main() -> None:
    args = parse_args()
    summaries = default_summary_files(args.workspace, args.include_old_lvisam)
    for path in args.summary:
        path = path.expanduser().resolve()
        if path.exists() and path not in summaries:
            summaries.append(path)

    if not summaries:
        raise SystemExit("No summary.csv files found.")

    rows = collect_rows(summaries)
    grouped = group_rows(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run_csv = args.out_dir / "attack_exposure_per_run.csv"
    grouped_csv = args.out_dir / "attack_exposure_grouped.csv"
    latex_path = args.out_dir / "attack_exposure_table.tex"

    write_csv(per_run_csv, rows, FIELDNAMES)
    write_csv(grouped_csv, grouped, GROUP_FIELDNAMES)
    write_latex_table(latex_path, grouped)

    print(f"[OK] summaries scanned: {len(summaries)}")
    print(f"[OK] per-run rows: {len(rows)}")
    print(f"[OK] grouped rows: {len(grouped)}")
    print(f"[OK] wrote {per_run_csv}")
    print(f"[OK] wrote {grouped_csv}")
    print(f"[OK] wrote {latex_path}")
    print()
    print_console_summary(grouped)


if __name__ == "__main__":
    main()
