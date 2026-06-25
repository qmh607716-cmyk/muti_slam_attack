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

    # center_deg uses the (atan2(y,x)+180)%360 convention shared with
    # polar_mask_2d. cos/sin expect atan2 convention, so subtract 180.
    center_atn2 = (center_deg - 180.0) % 360.0
    theta_deg = rng.uniform(center_atn2 - half_range_deg,
                            center_atn2 + half_range_deg,
                            size=n)
    theta_deg = theta_deg % 360.0
    theta_rad = np.radians(theta_deg)

    r = rng.uniform(1.0, 50.0, size=n)

    v_angles = _get_vertical_angles(num_lines)
    elev_deg = rng.choice(v_angles, size=n)
    elev_rad = np.radians(elev_deg)

    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)
    # Use tan(elev) to preserve physical elevation angle (Pitch = arctan(z/r)).
    # Using sin would compress the elevation: arctan(sin(elev)) ≠ elev.
    z = r * np.tan(elev_rad)

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
                                 lidar_scan_period: float = 0.1,
                                 intensities: np.ndarray = None) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, point_step), dtype=np.uint8)

    # center_deg uses (atan2(y,x)+180)%360 convention; convert to atan2 for cos/sin.
    center_atn2 = (center_deg - 180.0) % 360.0
    theta_deg = rng.uniform(center_atn2 - half_range_deg,
                            center_atn2 + half_range_deg,
                            size=n)
    theta_deg = theta_deg % 360.0
    theta_rad = np.radians(theta_deg)

    v_angles = _get_vertical_angles(num_lines)
    elev_deg = rng.choice(v_angles, size=n)
    elev_rad = np.radians(elev_deg)

    r = np.full(n, wall_distance, dtype=np.float32)
    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)
    # Use tan(elev) to preserve physical elevation angle (Pitch = arctan(z/r)).
    # Using sin would compress the elevation: arctan(sin(elev)) ≠ elev.
    z = r * np.tan(elev_rad)

    # Sample from real intensity distribution; fall back to intensity_value if unavailable
    if intensities is not None and intensities.shape[0] > 0:
        intensity = intensities.astype(np.float32)
    else:
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
                               intensity_value=None,
                               point_step: int = 22,
                               intensities: np.ndarray = None) -> np.ndarray:
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

    # Priority: passed-in intensities > source_records intensities > intensity_value fallback
    if intensities is not None and intensities.shape[0] == source_records.shape[0]:
        intensity = intensities.astype(np.float32)
    else:
        orig_int = _read_float32(source_records, 12, 16)
        if orig_int.max() > 0:
            intensity = orig_int
        elif intensity_value is not None:
            intensity = np.full(source_records.shape[0], intensity_value, dtype=np.float32)
        else:
            intensity = np.full(source_records.shape[0], 120.0, dtype=np.float32)

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
    intensities: np.ndarray = None,
) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, point_step), dtype=np.uint8)

    # center_deg uses (atan2(y,x)+180)%360 convention; convert to atan2 for cos/sin.
    center_atn2 = (center_deg - 180.0) % 360.0
    theta_deg = rng.uniform(
        center_atn2 - half_range_deg,
        center_atn2 + half_range_deg,
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
    # Use tan(elev) to preserve physical elevation angle (Pitch = arctan(z/r)).
    z = r * np.tan(elev_rad)

    # Sample from real intensity distribution; fall back to wall_intensity if unavailable
    if intensities is not None and intensities.shape[0] > 0:
        intensity = intensities.astype(np.float32)
    else:
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
    intensities: np.ndarray = None,
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

    # Use 2D horizontal norm so the injected points project onto a VERTICAL wall
    # (same elevation as the original beam), preserving the ring/topology.
    # Using 3D norm would compress horizontal range for high-elevation beams,
    # turning the square into a "diamond sphere" with warped edges.
    norm_2d = np.sqrt(x0 * x0 + y0 * y0)
    norm_safe = np.where(norm_2d < 1e-3, 1e-3, norm_2d)

    x = x0 / norm_safe * d_fake
    y = y0 / norm_safe * d_fake
    z = z0 / norm_safe * d_fake
    r = d_fake

    # Priority: passed-in intensities > source_records intensities > wall_intensity fallback
    if intensities is not None and intensities.shape[0] == source_records.shape[0]:
        intensity = intensities.astype(np.float32)
    else:
        orig_int = _read_float32(source_records, 12, 16)
        if orig_int.max() > 0:
            intensity = orig_int
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

    # Sample intensity from sector's original intensity distribution
    # to avoid mismatch (e.g. Kitti uses 0-1 normalized, not raw 0-255).
    # Falls back to wall_intensity if sector has no points or all-zero intensities.
    if affected.shape[0] > 0:
        orig_intensities = _read_float32(affected, 12, 16)
        if orig_intensities.max() > 0:
            # Sample from the real intensity distribution of sector points
            sector_intensities = rng.choice(orig_intensities, size=n_wall, replace=True)
        else:
            sector_intensities = None  # all-zero intensities -> use fallback
    else:
        sector_intensities = None

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
                intensities=None,  # inherit from selected source records
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
                intensities=sector_intensities,
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
                intensities=None,  # inherit from selected source records
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
            intensities=sector_intensities,
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


# ---------------------------------------------------------------------------
# Adaptive Injection: Hessian-Eigenvector-Guided Attack Geometry
# ---------------------------------------------------------------------------

def _adaptive_wall_distance(
    l_vul_in_window: float,
    l_vul_global: float,
    base_wall_dist: float,
    wall_dist_min: float,
    wall_dist_max: float,
) -> float:
    """
    Compute adaptive wall distance based on local vulnerability strength.

    D-SLAMSpoof principle: weaker local structure → more room for injected
    constraints to dominate → place wall closer (stronger pull toward fake).
    However, placing it TOO close may be unrealistic (LiDAR would see it).
    We use a sigmoid-like mapping: low L-Vul → closer wall.

    Args:
        l_vul_in_window: average L-Vul (dot_eigen_value) in the attack window
        l_vul_global:    global average L-Vul for normalisation
        base_wall_dist:  user-specified base distance
        wall_dist_min:   minimum allowed injection distance
        wall_dist_max:   maximum allowed injection distance

    Returns:
        adaptive wall distance in metres
    """
    if l_vul_global <= 0:
        return base_wall_dist

    ratio = l_vul_in_window / l_vul_global

    # Sigmoid compression: ratio 0→1 maps to dist in [base*0.6, base*1.4]
    # Clamp ratio to avoid overflow
    ratio = float(np.clip(ratio, 0.01, 10.0))
    t = 1.0 / (1.0 + np.exp(-3.0 * (ratio - 1.0)))

    dist = base_wall_dist * (0.6 + 0.8 * t)
    return float(np.clip(dist, wall_dist_min, wall_dist_max))


def _eigenvector_to_rotate_rad(
    eigenvec_xyz: np.ndarray,
    dot_eig: np.ndarray,
    center_deg: float,
    half_range_deg: float,
) -> float:
    """
    Infer the optimal injection shape orientation (rotate_rad) from Hessian eigenvectors.

    Key insight: each point's translation eigenvector gives the dominant direction
    along which that point resists motion. Points with low dot_eigen_value have
    the global minimum eigenvector direction dominating their local Hessian, meaning
    they are the weakest constraints. Injecting a fake wall perpendicular to the
    minimum-eigenvalue direction forces scan matching to accumulate drift in
    that specific direction.

    Algorithm:
      1. Keep only points in the attack angular window (where injection happens).
      2. Filter to those with low dot_eigen_value (weakest 30% of constraints).
      3. Compute the median 2D azimuth of their eigenvector xy-projection.
      4. Convert to rotate_rad: the eigenvector direction is the direction of
         weak constraint; we want the injection shape to have its edge
         perpendicular to this so that residuals point along the eigenvector.

    rotate_rad = 0    → corner shape with edges along x/y axes
    rotate_rad = θ    → corner rotated by θ → edge perpendicular to θ

    Returns rotate_rad in radians.
    """
    if eigenvec_xyz.shape[0] == 0 or dot_eig.shape[0] == 0:
        return 0.0

    theta_deg = (np.degrees(np.arctan2(eigenvec_xyz[:, 1], eigenvec_xyz[:, 0])) + 180.0) % 360.0
    center_mod = center_deg % 360.0
    delta = (theta_deg - center_mod + 180.0) % 360.0 - 180.0
    in_window = np.abs(delta) <= half_range_deg

    if not np.any(in_window):
        return 0.0

    eig_in = eigenvec_xyz[in_window]
    dot_in = dot_eig[in_window]

    threshold = np.percentile(dot_in, 30.0)
    weak_mask = dot_in <= threshold

    if not np.any(weak_mask):
        weak_mask = dot_in == dot_in.min()

    eig_weak = eig_in[weak_mask]

    median_x = float(np.median(eig_weak[:, 0]))
    median_y = float(np.median(eig_weak[:, 1]))

    if abs(median_x) < 1e-6 and abs(median_y) < 1e-6:
        return 0.0

    eig_theta = np.arctan2(median_y, median_x)

    rotate_rad = eig_theta + np.pi / 4.0
    return float(rotate_rad)


def adaptive_injection(
    bin_points: np.ndarray,
    center_deg: float,
    half_range_deg: float,
    timestamp: float,
    eigenvec_xyz: np.ndarray,
    dot_eig: np.ndarray,
    rng: np.random.Generator,
    base_wall_dist: float = 15.0,
    wall_dist_min: float = 5.0,
    wall_dist_max: float = 25.0,
    spoofing_cycle: float = None,
    M_corr: float = 1.0,
    point_count_model: str = "original",
    horizontal_resolution: float = 0.1,
    vertical_lines: float = 16.0,
    wall_intensity: float = 120.0,
    square_scale_S: float = None,
    lidar_scan_period: float = 0.1,
    auto_cycle: bool = True,
    point_step: int = 22,
) -> np.ndarray:
    """
    Adaptive Injection: dynamically chooses injection geometry guided by
    per-frame G-ICP Hessian eigenvector analysis.

    This is the core of the Hessian-eigenvector-guided adaptive attack.
    For each frame, the function:
      1. Computes the optimal injection shape orientation (rotate_rad) from
         Hessian eigenvectors of the weak-constraint points in the window.
      2. Computes an adaptive wall distance from local L-Vul strength.
      3. Optionally oscillates the wall radially (D-SLAMSpoof Oscillating Injection)
         using the optimal cycle derived from M_corr.
      4. Injects a corner-shaped (rotate_rad-oriented) false wall.

    The key innovation over D-SLAMSpoof:
      - D-SLAMSpoof uses a FIXED rotate_rad (0 for corner, pi/4 for planar).
      - adaptive_injection infers rotate_rad PER FRAME from the G-ICP Hessian,
        automatically aligning with the actual weakest constraint direction.

    Args:
        bin_points:       raw point cloud (N, point_step) uint8 binary
        center_deg:       attack angle in LiDAR frame (degrees, [0, 360))
        half_range_deg:   half-width of attack angular window
        timestamp:        bag relative time (seconds) for oscillation
        eigenvec_xyz:     (M, 3) per-point translation eigenvector xyz
        dot_eig:         (M,) per-point dot(global_min_vec, local_max_vec)
        rng:              seeded RNG
        base_wall_dist:   base injection distance (metres)
        wall_dist_min:    minimum oscillation bound
        wall_dist_max:    maximum oscillation bound
        spoofing_cycle:   oscillation period; if None and auto_cycle=True,
                          computed from M_corr (D-SLAMSpoof Eq. 4)
        M_corr:           max correspondence distance for auto_cycle (m)
        point_count_model: "original" | "equal_replace" | "pure_removal"
        horizontal_resolution: degrees per fake point (for count model)
        vertical_lines:   number of LiDAR channels (for count model)
        wall_intensity:   intensity value for injected points
        square_scale_S:   polar equation scale S; if None, auto-computed
        lidar_scan_period: LiDAR scan period (s)
        auto_cycle:       if True, compute optimal t_cycle from M_corr
        point_step:       22 for Velodyne, 18 for Livox

    Returns:
        modified point cloud as (N', point_step) uint8 binary
    """
    x = _read_float32(bin_points, 0, 4)
    y = _read_float32(bin_points, 4, 8)
    z = _read_float32(bin_points, 8, 12)

    mask = polar_mask_2d(x, y, center_deg, half_range_deg)
    kept = bin_points[~mask]

    if point_count_model == "pure_removal":
        return kept

    n_removed = int(mask.sum())

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

    # ── Step 1: Eigenvector-guided rotate_rad ──────────────────────────────
    rotate_rad = _eigenvector_to_rotate_rad(eigenvec_xyz, dot_eig, center_deg, half_range_deg)

    # ── Step 2: L-Vul-guided adaptive wall distance ────────────────────────
    dot_in_window = np.zeros(len(dot_eig), dtype=bool) if len(dot_eig) == 0 else None

    if dot_in_window is not None:
        theta_pts = (np.degrees(np.arctan2(y[mask], x[mask])) + 180.0) % 360.0
        center_mod = center_deg % 360.0
        delta_pts = (theta_pts - center_mod + 180.0) % 360.0 - 180.0

        if dot_eig.shape[0] > 0 and eigenvec_xyz.shape[0] == dot_eig.shape[0]:
            theta_eig = (np.degrees(np.arctan2(eigenvec_xyz[:, 1], eigenvec_xyz[:, 0])) + 180.0) % 360.0
            delta_eig = (theta_eig - center_mod + 180.0) % 360.0 - 180.0
            eig_in_window = np.abs(delta_eig) <= half_range_deg

            dot_in_window = dot_eig[eig_in_window] if np.any(eig_in_window) else dot_eig
        else:
            dot_in_window = dot_eig

        l_vul_local = float(np.mean(dot_in_window)) if len(dot_in_window) > 0 else 0.5
        l_vul_global = float(np.mean(dot_eig)) if len(dot_eig) > 0 else 1.0
    else:
        l_vul_local = 0.5
        l_vul_global = 1.0

    adaptive_dist = _adaptive_wall_distance(
        l_vul_local, l_vul_global,
        base_wall_dist, wall_dist_min, wall_dist_max,
    )

    # ── Step 3: Oscillating Injection ───────────────────────────────────────
    if auto_cycle and spoofing_cycle is None:
        spoofing_cycle = _optimal_cycle_from_mcorr(
            wall_dist_min, wall_dist_max, M_corr, lidar_scan_period,
        )
    elif spoofing_cycle is None:
        spoofing_cycle = 2.0

    frac = (timestamp % spoofing_cycle) / spoofing_cycle
    wall_dist = (wall_dist_max - wall_dist_min) * frac + wall_dist_min

    # ── Step 4: Square geometry with adaptive rotate_rad ───────────────────
    scale_S = square_scale_S if square_scale_S is not None else adaptive_dist * np.sqrt(2.0)

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
