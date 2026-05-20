#!/usr/bin/env python3
"""
spoofing_sim_lvisam.py
=======================

LVI-SAM-compatible SLAMSpoof attack functions.

Core principles:
1. Attack framework follows SLAMSpoof:
   - angular window: center +/- spoofing_range/2
   - removal: delete points in window + inject noise
   - static: delete points in window + inject false wall
   - dynamic: static with wall distance varying over time

2. LVI-SAM adaptation:
   - Outputs full 22-byte point records, not xyz-only
   - Always operates on complete 22-byte point layout:
     x, y, z, intensity, ring, time

3. Point count models:
   - "original": use SLAMSpoof original formula
   - "equal_replace": inject as many as removed
   - "pure_removal": remove only, no injection

4. Static geometry models (D-SLAMSpoof extension):
   - "original_random": uniform random angles → spread-out flat wall
   - "beam_project": inherits ring/time from affected beams → realistic topology
   - "square": D-SLAMSpoof polar equation → diamond/square shape, concentrated constraints
   - "corner": D-SLAMSpoof L-shape → concentrated constraints at corner edge

5. Oscillating Injection (D-SLAMSpoof extension):
   - cycle t_cycle is auto-computed from M_corr constraint:
     t_cycle = (d_max - d_min) / M_corr * delta_t
   - this is the FASTEST oscillation that avoids being rejected by outlier filtering

6. Physical realism:
   - time field: real scan-phase mapping (azimuth / 360) * scan_period
   - vertical angle: discrete sampling from LiDAR's fixed channel angles
     (16 for VLP-16, 32 for VLP-32C), not continuous values
   - num_lines is controlled by the `vertical_lines` config parameter
     and passed through to all synthesis functions
"""

import numpy as np


# VLP-16 approximate vertical angles, in degrees
_VLP16_VERTICAL_ANGLES = np.array([
    +15.0, +13.0, +11.0,  +9.0,
     +7.0,  +5.0,  +3.0,  +1.0,
     -1.0,  -3.0,  -5.0,  -7.0,
     -9.0, -11.0, -13.0, -15.0,
], dtype=np.float64)

# VLP-32C approximate vertical angles, in degrees (32 channels)
_VLP32C_VERTICAL_ANGLES = np.array([
    +10.00, +7.00, +4.67, +2.67, +1.00, -0.33, -1.67, -2.67,
     -3.67, -4.67, -5.67, -6.67, -7.67, -8.67, -9.67, -10.67,
    -11.67, -12.67, -13.67, -14.67, -15.67, -16.67, -17.67, -18.67,
    -19.67, -20.67, -22.00, -24.00, -26.00, -28.00, -30.00, -32.00,
], dtype=np.float64)

# HDL-64 approximate vertical angles, in degrees (64 channels, from spec sheet)
# Upper 32 channels (rings 0-31): +2.0° down to ~-9.0°
# Lower 32 channels (rings 32-63): -9.0° down to -24.9°
_HDL64_VERTICAL_ANGLES = np.array([
    +2.0,  +1.667, +1.333, +1.0,  +0.667, +0.333,  0.0,  -0.333,
    -0.667,-1.0,   -1.333, -1.667, -2.0,  -2.333,  -2.667, -3.0,
    -3.333,-3.667, -4.0,   -4.333, -4.667, -5.0,   -5.333, -5.667,
    -6.0,  -6.333, -6.667, -7.0,  -7.333, -7.667,  -8.0,  -8.333,
    -8.667, -9.0,  -9.333,  -9.667,-10.0, -10.333, -10.667,-11.0,
    -11.333,-11.667,-12.0, -12.333,-12.667,-13.0,  -13.333,-13.667,
    -14.0, -14.333,-14.667,-15.0, -15.333,-15.667,-16.0, -16.333,
    -16.667,-17.0, -17.333,-17.667,-18.0, -20.0,  -22.0, -24.9,
], dtype=np.float64)


def _read_float32(bin_points, start, end):
    return bin_points[:, start:end].copy().view(np.float32)[:, 0]


def _get_vertical_angles(num_lines: int) -> np.ndarray:
    """
    Return the vertical angle array for the given number of LiDAR lines.
    Supports 16 (VLP-16), 32 (VLP-32C), and 64 (HDL-64).
    """
    if num_lines == 16:
        return _VLP16_VERTICAL_ANGLES
    elif num_lines == 32:
        return _VLP32C_VERTICAL_ANGLES
    elif num_lines == 64:
        return _HDL64_VERTICAL_ANGLES
    else:
        raise ValueError(f"Unsupported lidar lines: {num_lines}. Supported: 16 (VLP-16), 32 (VLP-32C), 64 (HDL-64)")


def _approx_ring_from_z_and_r(z: np.ndarray, r: np.ndarray,
                               num_lines: int = 16) -> np.ndarray:
    r_safe = np.where(r < 0.01, 0.01, r)
    elevation = np.degrees(np.arcsin(np.clip(z / r_safe, -1.0, 1.0)))
    v_angles = _get_vertical_angles(num_lines)
    diff = np.abs(elevation[:, None] - v_angles[None, :])
    return np.argmin(diff, axis=1).astype(np.uint16)


def polar_mask_2d(x: np.ndarray, y: np.ndarray,
                  center_deg: float,
                  half_range_deg: float) -> np.ndarray:
    theta_deg = (np.degrees(np.arctan2(y, x)) + 180.0) % 360.0
    center_mod = center_deg % 360.0
    delta = (theta_deg - center_mod + 180.0) % 360.0 - 180.0
    return np.abs(delta) <= half_range_deg


def _pack_records_18(x, y, z, intensity, tag):
    """Pack Livox 18-byte record: x(4)+y(4)+z(4)+intensity(4)+tag(2)"""
    n = len(x)
    records = np.zeros((n, 18), dtype=np.uint8)
    records[:, 0:4].view(np.float32)[:, 0]   = np.asarray(x, dtype=np.float32)
    records[:, 4:8].view(np.float32)[:, 0]  = np.asarray(y, dtype=np.float32)
    records[:, 8:12].view(np.float32)[:, 0] = np.asarray(z, dtype=np.float32)
    records[:, 12:16].view(np.float32)[:, 0]= np.asarray(intensity, dtype=np.float32)
    records[:, 16:18].view(np.uint16)[:, 0] = np.asarray(tag, dtype=np.uint16)
    return records

def _pack_records_22(x, y, z, intensity, ring, time):
    """Pack Velodyne 22-byte record: x(4)+y(4)+z(4)+intensity(4)+ring(2)+time(4)"""
    n = len(x)
    records = np.zeros((n, 22), dtype=np.uint8)
    records[:, 0:4].view(np.float32)[:, 0]   = np.asarray(x, dtype=np.float32)
    records[:, 4:8].view(np.float32)[:, 0]  = np.asarray(y, dtype=np.float32)
    records[:, 8:12].view(np.float32)[:, 0]  = np.asarray(z, dtype=np.float32)
    records[:, 12:16].view(np.float32)[:, 0]= np.asarray(intensity, dtype=np.float32)
    records[:, 16:18].view(np.uint16)[:, 0] = np.asarray(ring, dtype=np.uint16)
    records[:, 18:22].view(np.float32)[:, 0] = np.asarray(time, dtype=np.float32)
    return records


def _approx_tag_from_z_and_r(z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Approximate Livox tag/line number from z and range."""
    r_safe = np.where(r < 0.01, 0.01, r)
    elev = np.degrees(np.arcsin(np.clip(z / r_safe, -1.0, 1.0)))
    # Livox Avia has 3 non-duplicate lines; use bucket edges from typical specs
    # Line 0: ~+15° to ~+5°,  Line 1: ~+5° to ~-5°,  Line 2: ~-5° to ~-15°
    lines = np.zeros_like(elev, dtype=np.uint16)
    lines[elev > 5.0]   = 0
    lines[(elev <= 5.0) & (elev > -5.0)] = 1
    lines[elev <= -5.0]  = 2
    return lines


def _original_noise_count(spoofing_range_deg: float,
                          horizontal_resolution: float,
                          vertical_lines: float,
                          spoofing_rate: float) -> int:
    """
    Original removal/HFR spoofed noise point count formula:
    int((spoofing_range / horizontal_resolution) * vertical_lines * spoofing_rate)
    """
    return max(0, int((spoofing_range_deg / horizontal_resolution) * vertical_lines * spoofing_rate))


def _original_static_count(spoofing_range_deg: float,
                           horizontal_resolution: float,
                           vertical_lines: float) -> int:
    """
    Original static false-wall injection point count formula:
    int((spoofing_range / horizontal_resolution) * vertical_lines)
    """
    return max(0, int((spoofing_range_deg / horizontal_resolution) * vertical_lines))




def _synth_noise_records(n: int,
                         center_deg: float,
                         half_range_deg: float,
                         rng: np.random.Generator,
                         num_lines: int = 16,
                         point_step: int = 22,
                         lidar_scan_period: float = 0.1) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, point_step), dtype=np.uint8)

    theta_deg = rng.uniform(center_deg - half_range_deg,
                            center_deg + half_range_deg,
                            size=n)
    theta_deg = theta_deg % 360.0
    theta_rad = np.radians(theta_deg)

    r = rng.uniform(1.0, 50.0, size=n)

    v_angles = _get_vertical_angles(num_lines)
    elev_deg = rng.choice(v_angles, size=n)
    elev_rad = np.radians(elev_deg)

    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)
    z = r * np.sin(elev_rad)

    intensity = rng.uniform(10.0, 50.0, size=n)
    ring = _approx_ring_from_z_and_r(z, r, num_lines=num_lines)

    if point_step == 18:
        # Livox format: no time field, tag instead of ring
        tag = _approx_tag_from_z_and_r(z, r)
        time = np.zeros(n, dtype=np.float32)
        return _pack_records_18(x, y, z, intensity, tag)
    else:
        # Velodyne format: has ring + time fields
        time = ((theta_deg % 360.0) / 360.0 * lidar_scan_period).astype(np.float32)
        return _pack_records_22(x, y, z, intensity, ring, time)


def _synth_wall_records_original(n: int,
                                 wall_distance: float,
                                 center_deg: float,
                                 half_range_deg: float,
                                 rng: np.random.Generator,
                                 intensity_value: float = 120.0,
                                 num_lines: int = 16,
                                 point_step: int = 22,
                                 lidar_scan_period: float = 0.1) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, point_step), dtype=np.uint8)

    theta_deg = rng.uniform(center_deg - half_range_deg,
                            center_deg + half_range_deg,
                            size=n)
    theta_deg = theta_deg % 360.0
    theta_rad = np.radians(theta_deg)

    v_angles = _get_vertical_angles(num_lines)
    elev_deg = rng.choice(v_angles, size=n)
    elev_rad = np.radians(elev_deg)

    r = np.full(n, wall_distance, dtype=np.float32)
    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)
    z = r * np.sin(elev_rad)

    intensity = np.full(n, intensity_value, dtype=np.float32)
    ring = _approx_ring_from_z_and_r(z, r, num_lines=num_lines)

    if point_step == 18:
        tag = _approx_tag_from_z_and_r(z, r)
        return _pack_records_18(x, y, z, intensity, tag)
    else:
        time = ((theta_deg % 360.0) / 360.0 * lidar_scan_period).astype(np.float32)
        return _pack_records_22(x, y, z, intensity, ring, time)


def _beam_project_wall_records(source_records: np.ndarray,
                               wall_distance: float,
                               intensity_value = None,
                               point_step: int = 22) -> np.ndarray:
    if source_records.shape[0] == 0:
        return np.zeros((0, point_step), dtype=np.uint8)

    x0 = _read_float32(source_records, 0, 4)
    y0 = _read_float32(source_records, 4, 8)
    z0 = _read_float32(source_records, 8, 12)

    norm = np.sqrt(x0 * x0 + y0 * y0 + z0 * z0)
    norm_safe = np.where(norm < 1e-3, 1e-3, norm)

    x = x0 / norm_safe * wall_distance
    y = y0 / norm_safe * wall_distance
    z = z0 / norm_safe * wall_distance
    r = wall_distance  # all injected points are at wall_distance

    if intensity_value is None:
        intensity = _read_float32(source_records, 12, 16)
    else:
        intensity = np.full(source_records.shape[0], intensity_value, dtype=np.float32)

    if point_step == 18:
        tag = _approx_tag_from_z_and_r(z, r)
        return _pack_records_18(x, y, z, intensity, tag)
    else:
        ring = source_records[:, 16:18].copy().view(np.uint16)[:, 0]
        time = source_records[:, 18:22].copy().view(np.float32)[:, 0]
        return _pack_records_22(x, y, z, intensity, ring, time)


# ---------------------------------------------------------------------------
# D-SLAMSpoof Constraint-Forging Injection: Square / Corner geometry
# ---------------------------------------------------------------------------
# Polar equation: d_fake = S / (|sin(θ')| + |cos(θ')|)
# This is the boundary of a diamond/square centered at origin in polar coords.
#
# θ' = θ - rotate_rad  controls the orientation:
#   rotate=0        → L-shaped corner (two adjacent edges facing the LiDAR)
#   rotate=π/4      → planar wall (two opposite edges, perpendicular to radial)
#   rotate=π/2      → rotated L-shape
#
# S controls the average distance (scales the whole shape).
# The shape concentrates all geometric constraints along the edges that are
# perpendicular to the LiDAR's radial direction → consistent directional bias
# in scan matching → persistent pose drift.

def _square_wall_records(
    n: int,
    scale_S: float,
    center_deg: float,
    half_range_deg: float,
    rotate_rad: float,
    rng: np.random.Generator,
    intensity_value: float = 120.0,
    num_lines: int = 16,
    point_step: int = 22,
    lidar_scan_period: float = 0.1,
) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, point_step), dtype=np.uint8)

    theta_deg = rng.uniform(
        center_deg - half_range_deg,
        center_deg + half_range_deg,
        size=n,
    )
    theta_deg = theta_deg % 360.0
    theta_rad = np.radians(theta_deg)

    theta_prime = theta_rad - rotate_rad

    eps = 1e-6
    d_fake = scale_S / (np.abs(np.sin(theta_prime)) + np.abs(np.cos(theta_prime)) + eps)

    r = d_fake.astype(np.float32)

    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)

    v_angles = _get_vertical_angles(num_lines)
    elev_deg = rng.choice(v_angles, size=n)
    elev_rad = np.radians(elev_deg)
    z = r * np.sin(elev_rad)

    intensity = np.full(n, intensity_value, dtype=np.float32)
    ring = _approx_ring_from_z_and_r(z, r, num_lines=num_lines)

    if point_step == 18:
        tag = _approx_tag_from_z_and_r(z, r)
        return _pack_records_18(x, y, z, intensity, tag)
    else:
        time = ((theta_deg % 360.0) / 360.0 * lidar_scan_period).astype(np.float32)
        return _pack_records_22(x, y, z, intensity, ring, time)


def _square_beam_project_records(
    source_records: np.ndarray,
    scale_S: float,
    center_deg: float,
    half_range_deg: float,
    rotate_rad: float,
    intensity_value: float = 120.0,
    point_step: int = 22,
) -> np.ndarray:
    if source_records.shape[0] == 0:
        return np.zeros((0, point_step), dtype=np.uint8)

    x0 = _read_float32(source_records, 0, 4)
    y0 = _read_float32(source_records, 4, 8)
    z0 = _read_float32(source_records, 8, 12)

    theta_rad = np.arctan2(y0, x0)
    theta_prime = theta_rad - rotate_rad

    eps = 1e-6
    d_fake = scale_S / (np.abs(np.sin(theta_prime)) + np.abs(np.cos(theta_prime)) + eps)

    norm = np.sqrt(x0 * x0 + y0 * y0 + z0 * z0)
    norm_safe = np.where(norm < 1e-3, 1e-3, norm)

    x = x0 / norm_safe * d_fake
    y = y0 / norm_safe * d_fake
    z = z0 / norm_safe * d_fake
    r = d_fake

    if intensity_value is None:
        intensity = _read_float32(source_records, 12, 16)
    else:
        intensity = np.full(source_records.shape[0], intensity_value, dtype=np.float32)

    if point_step == 18:
        tag = _approx_tag_from_z_and_r(z, r)
        return _pack_records_18(x, y, z, intensity, tag)
    else:
        ring = source_records[:, 16:18].copy().view(np.uint16)[:, 0]
        time = source_records[:, 18:22].copy().view(np.float32)[:, 0]
        return _pack_records_22(x, y, z, intensity, ring, time)


def _optimal_cycle_from_mcorr(
    d_min: float,
    d_max: float,
    M_corr: float,
    delta_t: float,
) -> float:
    """
    D-SLAMSpoof Oscillating Injection: compute the OPTIMAL cycle time.

    From D-SLAMSpoof Eq. (4):
        t_cycle = (d_max - d_min) / M_corr * delta_t

    This is the FASTEST oscillation that still stays within the correspondence
    distance gate M_corr. Faster oscillation → injected points get rejected;
    slower oscillation → less accumulated drift per unit time.

    Args:
        d_min: minimum injection distance (meters)
        d_max: maximum injection distance (meters, must be within LiDAR range)
        M_corr: maximum correspondence distance threshold of the SLAM algorithm
                (typically 1.0~2.0m for FAST-LIO2/KISS-ICP; use 1.0 as conservative default)
        delta_t: LiDAR scan period (seconds, typically 0.05~0.1s for VLP-16)

    Returns:
        optimal t_cycle in seconds
    """
    delta_d = d_max - d_min
    if delta_d <= 0:
        return float('inf')
    t_cycle = (delta_d / M_corr) * delta_t
    return max(t_cycle, delta_t)  # at least one scan period


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def removal_injection(
    bin_points: np.ndarray,
    center_deg: float,
    half_range_deg: float,
    rng: np.random.Generator,
    point_count_model: str = "original",
    horizontal_resolution: float = 0.1,
    vertical_lines: float = 16.0,
    spoofing_rate: float = 0.1,
    point_step: int = 22,
    lidar_scan_period: float = 0.1,
) -> np.ndarray:
    x = _read_float32(bin_points, 0, 4)
    y = _read_float32(bin_points, 4, 8)

    mask = polar_mask_2d(x, y, center_deg, half_range_deg)
    kept = bin_points[~mask]

    n_removed = int(mask.sum())

    if point_count_model == "pure_removal":
        return kept

    if point_count_model == "equal_replace":
        n_noise = n_removed
    else:
        spoofing_range_deg = half_range_deg * 2.0
        n_noise = _original_noise_count(
            spoofing_range_deg,
            horizontal_resolution,
            vertical_lines,
            spoofing_rate,
        )

    noise_records = _synth_noise_records(
        n_noise, center_deg, half_range_deg, rng,
        num_lines=int(vertical_lines), point_step=point_step,
        lidar_scan_period=lidar_scan_period,
    )

    if kept.shape[0] == 0:
        return noise_records
    if noise_records.shape[0] == 0:
        return kept
    return np.concatenate([kept, noise_records], axis=0)


def static_injection(
    bin_points: np.ndarray,
    center_deg: float,
    half_range_deg: float,
    wall_distance: float,
    rng: np.random.Generator,
    point_count_model: str = "original",
    horizontal_resolution: float = 0.1,
    vertical_lines: float = 16.0,
    static_geometry_model: str = "original_random",
    wall_intensity: float = 120.0,
    square_scale_S: float = None,
    square_rotate_rad: float = None,
    point_step: int = 22,
    lidar_scan_period: float = 0.1,
) -> np.ndarray:
    """
    static false-wall injection.

    point_count_model:
      - original: use original formula n_injection = range/resolution*vertical_lines
      - equal_replace: inject as many as removed
      - pure_removal: remove only, no injection

    static_geometry_model:
      - original_random: uniform random angles in window → spread-out flat wall
      - beam_project: inherits ring/time from affected beams → realistic topology
      - square: D-SLAMSpoof polar equation → concentrated constraint geometry
      - corner: alias for square with rotate=0 → L-shape
      - planar: square with rotate=π/4 → planar wall perpendicular to radial

    D-SLAMSpoof square geometry (model="square"/"corner"/"planar"):
      - square_scale_S: scaling constant S (controls average distance).
                        If None, defaults to wall_distance * 1.414 (≈ diagonal of square)
      - square_rotate_rad: rotation of polar equation.
                           0 (default)       → corner/L-shape (adjacent edges)
                           np.pi/4           → planar wall (opposite edges)
                           np.pi/2           → rotated L-shape
    """
    x = _read_float32(bin_points, 0, 4)
    y = _read_float32(bin_points, 4, 8)

    mask = polar_mask_2d(x, y, center_deg, half_range_deg)
    kept = bin_points[~mask]
    affected = bin_points[mask]
    n_removed = int(mask.sum())

    if point_count_model == "pure_removal":
        return kept

    if point_count_model == "equal_replace":
        n_wall = n_removed
    else:
        spoofing_range_deg = half_range_deg * 2.0
        n_wall = _original_static_count(
            spoofing_range_deg,
            horizontal_resolution,
            vertical_lines,
        )

    if n_wall <= 0:
        return kept

    # ── D-SLAMSpoof square / corner / planar geometry ───────────────────
    square_models = {"square", "corner", "planar"}
    if static_geometry_model in square_models:

        # Determine rotation
        if static_geometry_model == "corner":
            rotate_rad = 0.0  # L-shape: adjacent edges facing LiDAR
        elif static_geometry_model == "planar":
            rotate_rad = np.pi / 4.0  # planar wall: opposite edges
        else:
            # "square": use explicit rotate_rad if provided, else default to corner
            rotate_rad = float(square_rotate_rad) if square_rotate_rad is not None else 0.0

        # Scale S: controls average injection distance
        # For a square with edge length L, the diamond boundary d_fake ranges
        # from L/2 (at edges aligned with axes) to L/√2 (at 45°).
        # We want average distance ≈ wall_distance, so set S ≈ wall_distance * √2
        # This makes d_fake ≈ wall_distance * √2 / (|sin|+|cos|)
        if square_scale_S is not None:
            scale_S = float(square_scale_S)
        else:
            scale_S = wall_distance * np.sqrt(2.0)

        if static_geometry_model == "beam_project" and affected.shape[0] > 0:
            replace = affected.shape[0] < n_wall
            idx = rng.choice(affected.shape[0], size=n_wall, replace=replace)
            selected = affected[idx]
            wall_records = _square_beam_project_records(
                selected,
                scale_S,
                center_deg,
                half_range_deg,
                rotate_rad,
                intensity_value=wall_intensity,
                point_step=point_step,
            )
        else:
            wall_records = _square_wall_records(
                n_wall,
                scale_S,
                center_deg,
                half_range_deg,
                rotate_rad,
                rng,
                intensity_value=wall_intensity,
                num_lines=int(vertical_lines),
                point_step=point_step,
                lidar_scan_period=lidar_scan_period,
            )

        if kept.shape[0] == 0:
            return wall_records
        if wall_records.shape[0] == 0:
            return kept
        return np.concatenate([kept, wall_records], axis=0)

    # ── Original geometry models ───────────────────────────────────────
    if static_geometry_model == "beam_project":
        if affected.shape[0] == 0:
            wall_records = np.zeros((0, point_step), dtype=np.uint8)
        else:
            replace = affected.shape[0] < n_wall
            idx = rng.choice(affected.shape[0], size=n_wall, replace=replace)
            selected = affected[idx]
            wall_records = _beam_project_wall_records(
                selected,
                wall_distance,
                intensity_value=wall_intensity,
                point_step=point_step,
            )
    else:
        wall_records = _synth_wall_records_original(
            n_wall,
            wall_distance,
            center_deg,
            half_range_deg,
            rng,
            intensity_value=wall_intensity,
            num_lines=int(vertical_lines),
            point_step=point_step,
            lidar_scan_period=lidar_scan_period,
        )

    if kept.shape[0] == 0:
        return wall_records
    if wall_records.shape[0] == 0:
        return kept
    return np.concatenate([kept, wall_records], axis=0)


def dynamic_injection(
    bin_points: np.ndarray,
    center_deg: float,
    half_range_deg: float,
    timestamp: float,
    rng: np.random.Generator,
    wall_dist_min: float = 5.0,
    wall_dist_max: float = 25.0,
    spoofing_cycle: float = None,
    point_count_model: str = "original",
    horizontal_resolution: float = 0.1,
    vertical_lines: float = 16.0,
    static_geometry_model: str = "original_random",
    wall_intensity: float = 120.0,
    square_scale_S: float = None,
    square_rotate_rad: float = None,
    M_corr: float = 1.0,
    lidar_scan_period: float = 0.1,
    auto_cycle: bool = True,
    point_step: int = 22,
) -> np.ndarray:
    """
    Dynamic false-wall injection (wall distance varies over time).

    D-SLAMSpoof Oscillating Injection:
      When auto_cycle=True (default), the cycle t_cycle is automatically
      computed from the M_corr constraint (Eq. 4 in D-SLAMSpoof):

          t_cycle = (d_max - d_min) / M_corr * Δt

      This gives the FASTEST oscillation that avoids being rejected by
      the SLAM's outlier correspondence filtering, maximizing attack
      effectiveness per unit time.

      When auto_cycle=False, the user-specified spoofing_cycle is used
      (legacy behavior, may be slower than optimal or rejected if too fast).

    Args:
      M_corr: maximum correspondence distance of the target SLAM algorithm.
              Conservative defaults:
                FAST-LIO2: ~1.0m
                KISS-ICP:  ~1.0~2.0m
                LVI-SAM:   ~1.0m (LiDAR factor dominates)
              Increase if the SLAM uses a larger outlier rejection threshold.
      lidar_scan_period: Δt between consecutive scans (VLP-16: 0.1s, HDL-64: 0.05s)
    """
    if auto_cycle and spoofing_cycle is None:
        # Compute optimal cycle from M_corr constraint (D-SLAMSpoof Eq. 4)
        spoofing_cycle = _optimal_cycle_from_mcorr(
            wall_dist_min, wall_dist_max, M_corr, lidar_scan_period
        )
    elif spoofing_cycle is None:
        spoofing_cycle = 2.0  # fallback default

    # Linear distance sweep within one cycle (D-SLAMSpoof Eq. 5)
    frac = (timestamp % spoofing_cycle) / spoofing_cycle
    wall_dist = (wall_dist_max - wall_dist_min) * frac + wall_dist_min

    return static_injection(
        bin_points,
        center_deg,
        half_range_deg,
        wall_dist,
        rng,
        point_count_model=point_count_model,
        horizontal_resolution=horizontal_resolution,
        vertical_lines=vertical_lines,
        static_geometry_model=static_geometry_model,
        wall_intensity=wall_intensity,
        square_scale_S=square_scale_S,
        square_rotate_rad=square_rotate_rad,
        point_step=point_step,
        lidar_scan_period=lidar_scan_period,
    )


# Original API names preserved, but LVI-SAM editor should not use them
def spoof_main(pointcloud, largest_score_angle, spoofing_range):
    raise NotImplementedError("Use removal_injection() for LVI-SAM editor.")


def injection_main(pointcloud, largest_score_angle, spoofing_range, wall_dist):
    raise NotImplementedError("Use static_injection() for LVI-SAM editor.")


def dynamic_injection_main(pointcloud, timestamp, largest_score_angle, spoofing_range):
    raise NotImplementedError("Use dynamic_injection() for LVI-SAM editor.")
