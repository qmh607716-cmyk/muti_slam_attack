#!/usr/bin/env python3
"""
calc_bimodal_smvs.py
=====================

Bimodal (LiDAR + Visual) Scan Matching Vulnerability Score.

Core framework:
    L-Vul[k] : LiDAR-only vulnerability (GICP Hessian eigenvalue bucket)
    V-Vul[k] : Visual compensation capability (camera quality per bucket)
    Bi-Vul[k] = L-Vul[k] × (1 - V-Vul[k] × L-Vul_norm[k])
    L-Vul_norm[k] = L-Vul[k] / l_vul_max  (per-frame max for normalisation)

    Visual quality good  → V-Vul high  → attack weakened  → Bi-Vul low
    Visual quality poor / attack outside FOV → V-Vul low → attack strong → Bi-Vul ≈ L-Vul

V-Vul model:
    V-Vul[k] = γ × cam_coverage[k] × Q[k]
    Q[k] = w_track × feature_density[k]
         + w_optical × flow_consistency[k]
         + w_depth × depth_quality          # global scalar (same for all buckets)
         + w_spatial × spatial_dist         # global scalar (same for all buckets)
         + w_parallax × parallax            # global scalar (same for all buckets)

    Where:
      - cam_coverage: hierarchical FOV (center=1.0, edge=0.7, blind=0.0)
      - feature_density: ORB keypoint count projected into LiDAR buckets
      - flow_consistency: Farneback optical flow direction coherence per bucket
      - depth_quality: weighted ratio of LiDAR points inside camera FOV
      - spatial_dist: 6×6 grid feature coverage
      - parallax: ORB descriptor matching median distance

References:
    SLAMSpoof, ICRA 2025
    VINS-Mono, T-RO 2019
    LVI-SAM, ICRA 2021
"""

import numpy as np

# ---------------------------------------------------------------------------
# Camera intrinsics (LVI-SAM params_camera.yaml)
# ---------------------------------------------------------------------------
_CAM_FX = 669.894
_CAM_FY = 669.145
_CAM_U0 = 377.946
_CAM_V0 = 279.637
_CAM_W  = 720
_CAM_H  = 540

# Horizontal / vertical FOV (radians)
_FOV_H = 2.0 * np.arctan(_CAM_W / 2.0 / _CAM_FX)   # ≈ 1.07 rad
_FOV_V = 2.0 * np.arctan(_CAM_H / 2.0 / _CAM_FY)   # ≈ 0.86 rad

# ---------------------------------------------------------------------------
# V-Vul parameters
# ---------------------------------------------------------------------------
# Hierarchical FOV boundaries (radians, relative to optical axis)
_FOV_CENTER = _FOV_H * 0.15   # ≈ 9.2°: center clear zone → coverage = 1.0
_FOV_EDGE   = _FOV_H * 0.40   # ≈ 24.5°: edge transition zone → coverage = 0.7
# > FOV_EDGE: blind zone → coverage = 0.0

# Component weights (normalized, sum to 1.0)
_W_TRACK    = 0.20   # ORB feature density (per bucket)
_W_OPTICAL  = 0.30   # optical flow consistency (per bucket)
_W_DEPTH    = 0.20   # depth assist (LiDAR FOV coverage)
_W_SPATIAL  = 0.15   # feature spatial distribution
_W_PARALLAX = 0.15   # parallax information

# Maximum visual compensation coefficient
# Increased to 0.70: higher V-Vul → stronger Bi-Vul suppression,
# forcing selection into truly common-vulnerability zones where vision also fails.
_GAMMA = 0.70

# V-Vul floor: removed. Zero visual compensation means maximum Bi-Vul penalty,
# which correctly selects areas where both LiDAR AND vision are vulnerable.
_VUL_FLOOR = 0.0

# Temporal EMA smoothing coefficient (0=no smoothing, 1=heavy smoothing)
_EMA_ALPHA = 0.60


# ---------------------------------------------------------------------------
# Helpers: polar bucket
# ---------------------------------------------------------------------------

def cartesian2polar(x, y):
    """World frame → polar (r, theta_deg), theta ∈ [0, 360)"""
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(y, x)) + 180.0
    return r, theta


def polar_buckets(theta_deg, step_deg=5.0):
    """Return bucket index (0..n_buckets-1) for each point."""
    n_buckets = int(360.0 / step_deg)
    idx = (theta_deg / step_deg).astype(np.int32) % n_buckets
    return idx, n_buckets


def bucket_score(theta_deg, value, step_deg=5.0):
    """Bucket-sum value by azimuth angle."""
    idx, n = polar_buckets(theta_deg, step_deg)
    scores = np.zeros(n, dtype=np.float64)
    np.add.at(scores, idx, value)
    centers = (np.arange(n) + 0.5) * step_deg
    return scores, centers


# ---------------------------------------------------------------------------
# V-Vul core
# ---------------------------------------------------------------------------

def _hierarchical_coverage(theta_cam_rad: float) -> float:
    """
    Hierarchical FOV coverage model.

    |theta| <= FOV_CENTER  →  1.0 (center clear zone)
    FOV_CENTER < |theta| <= FOV_EDGE  →  linear 1.0 → 0.7
    |theta| > FOV_EDGE     →  0.0 (blind zone)
    """
    abs_t = abs(theta_cam_rad)
    if abs_t <= _FOV_CENTER:
        return 1.0
    if abs_t <= _FOV_EDGE:
        t = (abs_t - _FOV_CENTER) / (_FOV_EDGE - _FOV_CENTER)
        return 1.0 - 0.3 * t
    return 0.0


def _project_kp_to_lidar_bucket(
    kp_list,
    img_w=_CAM_W, img_h=_CAM_H,
    robot_yaw=0.0,
    lidar_to_cam_yaw=-0.04,
    step_deg=5.0,
):
    """
    Project ORB keypoints into LiDAR local frame bucket indices,
    returning per-bucket feature count.

    Approximate: feature at 5m depth (real depth can be obtained from LiDAR projection).
    """
    n_buckets = int(360.0 / step_deg)
    counts = np.zeros(n_buckets, dtype=np.float64)

    if not kp_list:
        return counts

    fx, fy, u0, v0 = _CAM_FX, _CAM_FY, _CAM_U0, _CAM_V0
    cam_world = robot_yaw + lidar_to_cam_yaw

    for kp in kp_list:
        u, v = kp.pt[0], kp.pt[1]
        x_c = (u - u0) / fx
        y_c = (v - v0) / fy
        z_c = 5.0
        r_inv = z_c / np.sqrt(x_c**2 + y_c**2 + 1.0)
        x_l = r_inv * x_c
        y_l = r_inv * y_c

        theta_lidar = np.arctan2(y_l, x_l)
        theta_lidar_deg = np.degrees(theta_lidar) + 180.0
        idx = int(theta_lidar_deg / step_deg) % n_buckets
        counts[idx] += 1.0

    return counts


def _optical_flow_per_bucket(
    prev_gray, curr_gray,
    robot_yaw=0.0,
    lidar_to_cam_yaw=-0.04,
    step_deg=5.0,
):
    """
    Compute per-bucket optical flow consistency (motion stability).

    Uses Farneback dense optical flow, divided by bucket:
      - motion magnitude (information)
      - direction consistency (circular std → 0 means all pixels move same way)
    """
    try:
        import cv2
    except ImportError:
        n_buckets = int(360.0 / step_deg)
        return np.zeros(n_buckets, dtype=np.float64)

    n_buckets = int(360.0 / step_deg)
    scores = np.zeros(n_buckets, dtype=np.float64)

    if prev_gray is None or curr_gray is None:
        return scores

    h, w = curr_gray.shape[:2]
    cam_world = robot_yaw + lidar_to_cam_yaw

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    fx_flow, fy_flow = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx_flow**2 + fy_flow**2)

    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    theta_cam_h = (u_coords - _CAM_U0) / _CAM_FX
    phi_cam_v   = (v_coords - _CAM_V0) / _CAM_FY
    theta_lidar = theta_cam_h + cam_world
    idx_map = ((np.degrees(theta_lidar) + 180.0) / step_deg).astype(np.int32) % n_buckets

    valid_v = np.abs(phi_cam_v) <= (_FOV_V / 2.0)

    for k in range(n_buckets):
        mask = (idx_map == k) & valid_v
        if not np.any(mask):
            continue
        mag_k  = mag[mask]
        fx_k   = fx_flow[mask]
        fy_k   = fy_flow[mask]
        median_mag = np.median(mag_k)
        if median_mag <= 0.5:
            continue
        angles = np.arctan2(fy_k, fx_k)
        sin_m  = np.mean(np.sin(angles))
        cos_m  = np.mean(np.cos(angles))
        r_bar  = np.sqrt(sin_m**2 + cos_m**2)
        info   = np.clip(median_mag / 5.0, 0.0, 1.0)
        scores[k] = r_bar * info

    max_s = scores.max()
    if max_s > 0:
        scores /= max_s
    return scores


def _feature_spatial_distribution(kp_list, img_w=_CAM_W, img_h=_CAM_H):
    """Feature spatial distribution: 6×6 grid coverage ratio."""
    if not kp_list:
        return 0.0
    rows, cols = 6, 6
    cell_w = img_w / cols
    cell_h = img_h / rows
    occupied = np.zeros((rows, cols), dtype=bool)
    for kp in kp_list:
        u, v = kp.pt[0], kp.pt[1]
        c = min(int(u / cell_w), cols - 1)
        r = min(int(v / cell_h), rows - 1)
        occupied[r, c] = True
    return float(occupied.sum() / (rows * cols))


def _parallax_approximate(kp_curr, kp_prev, curr_gray, prev_gray,
                           img_w=_CAM_W, img_h=_CAM_H):
    """Approximate parallax: median distance from ORB descriptor brute-force matching."""
    try:
        import cv2
    except ImportError:
        return 0.0
    if len(kp_curr) < 5 or len(kp_prev) < 5:
        return 0.0
    if curr_gray is None or prev_gray is None:
        return 0.0
    orb = cv2.ORB_create(nfeatures=100)
    # Pass real grayscale images so ORB can extract descriptors at the
    # detected keypoint locations (the earlier detect step used real images).
    _, desc_c = orb.compute(curr_gray, kp_curr)
    _, desc_p = orb.compute(prev_gray, kp_prev)
    if desc_c is None or desc_p is None or len(desc_c) == 0:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_c, desc_p)
    if len(matches) < 3:
        return 0.0
    median_d = np.median([m.distance for m in matches])
    return float(np.clip(1.0 - median_d / 128.0, 0.0, 1.0))


def compute_visual_vul(
    robot_yaw: float = 0.0,
    kp_list=None,
    prev_kp_list=None,
    prev_gray=None,
    curr_gray=None,
    lidar_xyz_in_cam_fov: float = None,
    gamma: float = _GAMMA,
    step_deg: float = 5.0,
) -> tuple:
    """
    Compute per-bucket V-Vul.

    Formula:
        V-Vul[k] = γ × cam_coverage[k] × Q[k]
        Q[k] = w_track × feature_density[k]
             + w_optical × flow_consistency[k]
             + w_depth × depth_quality
             + w_spatial × spatial_dist
             + w_parallax × parallax

    Args:
        robot_yaw:            robot yaw (radians)
        kp_list:             current frame ORB keypoint list
        prev_kp_list:        previous frame ORB keypoint list
        prev_gray:            previous frame grayscale image
        curr_gray:            current frame grayscale image
        lidar_xyz_in_cam_fov: count of LiDAR points inside camera FOV (scalar)
        gamma:               visual max compensation coefficient
        step_deg:            bucket resolution

    Returns:
        v_vul:  shape (n_buckets,)  visual compensation capability
        info:   dict  debug information
    """
    kp_list = kp_list or []
    n_buckets = int(360.0 / step_deg)

    lidar_to_cam_yaw = -0.04
    cam_world = robot_yaw + lidar_to_cam_yaw

    # Global quantities (not per-bucket)
    depth_quality = 0.0
    if lidar_xyz_in_cam_fov is not None:
        depth_quality = float(np.clip(lidar_xyz_in_cam_fov / 5000.0, 0.0, 1.0))

    spatial_dist = _feature_spatial_distribution(kp_list)
    parallax     = _parallax_approximate(kp_list, prev_kp_list,
                                           curr_gray, prev_gray)

    # Per-bucket quantities
    feature_density = _project_kp_to_lidar_bucket(
        kp_list, _CAM_W, _CAM_H, robot_yaw, lidar_to_cam_yaw, step_deg)
    if feature_density.max() > 0:
        feature_density /= feature_density.max()

    flow_consistency = _optical_flow_per_bucket(
        prev_gray, curr_gray, robot_yaw, lidar_to_cam_yaw, step_deg)

    # Per-bucket V-Vul[k]
    v_vul = np.zeros(n_buckets, dtype=np.float64)
    for k in range(n_buckets):
        theta_lidar_deg = (k + 0.5) * step_deg
        theta_lidar_rad = np.radians(theta_lidar_deg)
        theta_cam = theta_lidar_rad - cam_world
        theta_cam = np.arctan2(np.sin(theta_cam), np.cos(theta_cam))

        coverage = _hierarchical_coverage(theta_cam)
        if coverage <= 0:
            v_vul[k] = 0.0
            continue

        Q_k = (_W_TRACK    * feature_density[k] +
               _W_OPTICAL  * flow_consistency[k] +
               _W_DEPTH    * depth_quality +
               _W_SPATIAL  * spatial_dist +
               _W_PARALLAX * parallax)
        Q_k = float(np.clip(Q_k, 0.0, 1.0))

        v_vul[k] = gamma * coverage * Q_k

    v_vul = np.clip(v_vul, _VUL_FLOOR, gamma)

    info = {
        "n_features": len(kp_list),
        "depth_quality": depth_quality,
        "spatial_dist": spatial_dist,
        "parallax": parallax,
        "feature_density_max": float(feature_density.max()),
        "flow_consistency_max": float(flow_consistency.max()),
    }
    return v_vul, info


# ---------------------------------------------------------------------------
# LiDAR modality vulnerability (L-Vul)
# ---------------------------------------------------------------------------

def compute_lidar_vul_from_hessian(xyz, dot_eigen_value, step_deg=5.0) -> np.ndarray:
    """Point-level vulnerability based on G-ICP Hessian."""
    _, theta = cartesian2polar(xyz[:, 0], xyz[:, 1])
    l_vul, _ = bucket_score(theta, dot_eigen_value, step_deg)
    return l_vul


# ---------------------------------------------------------------------------
# Bimodal fusion (Bi-Vul)
# ---------------------------------------------------------------------------

def fuse_bimodal(l_vul, v_vul, l_vul_max: float = None) -> np.ndarray:
    """
    Fuse L-Vul and V-Vul.

    L-Vul-aware fusion with doubled visual penalty:
        Bi-Vul[k] = L-Vul[k] × (1 - 2.0 × V-Vul[k] × L-Vul_norm[k])
    where L-Vul_norm[k] = L-Vul[k] / l_vul_max.

    Doubled penalty (2×) strongly suppresses areas where vision compensates,
    forcing selection into truly common-vulnerability zones.

    Clamped to non-negative: Bi-Vul >= 0.

    If l_vul_max is None, degrades to:
        Bi-Vul[k] = L-Vul[k] × (1 - 2.0 × V-Vul[k])
    """
    if l_vul_max is None or l_vul_max <= 0:
        bi_vul = l_vul * (1.0 - 2.0 * v_vul)
        return np.maximum(bi_vul, 0.0)

    l_vul_norm = l_vul / l_vul_max
    bi_vul = l_vul * (1.0 - 2.0 * v_vul * l_vul_norm)
    return np.maximum(bi_vul, 0.0)


# ---------------------------------------------------------------------------
# Frame-level SMVS
# ---------------------------------------------------------------------------

def _circular_distance(i, j, n):
    d = abs(i - j)
    return min(d, n - d)


def frame_smvs(
    vul,
    step_deg=5.0,
    d_th=None,
) -> float:
    """
    Frame-level SMVS (applies to L-Vul, V-Vul, or Bi-Vul).

    HFR mode (d_th=None → 8):   attack range ≈ ±40° (80° total)
    AHFR mode (d_th=2):         attack range ≈ ±10° (20° total)
    """
    if d_th is None:
        d_th = 8
    n = len(vul)
    center_idx = int(np.argmax(vul))

    score = 0.0
    for k in range(n):
        dist = _circular_distance(k, center_idx, n)
        if dist <= d_th:
            weight = -dist + d_th
            score += vul[k] * weight
    return float(score)


def frame_lidar_smvs(l_vul, step_deg=5.0, d_th=None) -> float:
    """LiDAR-only frame-level SMVS."""
    return frame_smvs(l_vul, step_deg, d_th)


def frame_visual_smvs(v_vul, step_deg=5.0, d_th=None) -> float:
    """Visual-only frame-level SMVS."""
    return frame_smvs(v_vul, step_deg, d_th)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def vulnerable_direction(vul, step_deg=5.0) -> float:
    """Return azimuth of the highest vulnerability bucket (degrees, [0, 360))."""
    idx = int(np.argmax(vul))
    return float(idx + 0.5) * step_deg


def vec_from_angle(angle_deg):
    """Azimuth → unit vector (vec_x, vec_y)."""
    rad = np.radians(angle_deg)
    return float(np.cos(rad)), float(np.sin(rad))


# ---------------------------------------------------------------------------
# Temporal EMA filter
# ---------------------------------------------------------------------------

class VVulEMAFilter:
    """
    V-Vul temporal EMA smoothing filter.

    Usage:
        ema = VVulEMAFilter(alpha=0.6)
        for frame in frames:
            v_vul_raw = compute_visual_vul(...)
            v_vul_smooth = ema.update(v_vul_raw)
    """

    def __init__(self, alpha: float = _EMA_ALPHA):
        self.alpha = alpha
        self._prev: np.ndarray = None

    def update(self, v_vul: np.ndarray) -> np.ndarray:
        """EMA smoothing: ema = α·new + (1-α)·prev."""
        if self._prev is None:
            self._prev = v_vul.copy()
            return v_vul.copy()
        smoothed = self.alpha * v_vul + (1.0 - self.alpha) * self._prev
        self._prev = smoothed.copy()
        return smoothed

    def reset(self):
        self._prev = None


# ---------------------------------------------------------------------------
# VINS feature msg parsing
# ---------------------------------------------------------------------------

def compute_visual_vul_from_feature_msg(feature_cloud_msg) -> dict:
    """Parse tracking quality from VINS feature PointCloud."""
    n_features = len(feature_cloud_msg.points)
    if n_features == 0:
        return {"last_track_num": 0, "avg_parallax": 0.0, "depth_ratio": 0.0}

    u_ch  = [ch for ch in feature_cloud_msg.channels if ch.name == "u"]
    vx_ch = [ch for ch in feature_cloud_msg.channels if ch.name == "velocity_x"]
    vy_ch = [ch for ch in feature_cloud_msg.channels if ch.name == "velocity_y"]

    u_arr  = np.array([c.values[0] for c in u_ch],  dtype=np.float64) if u_ch  else np.array([], dtype=np.float64)
    vx_arr = np.array([c.values[0] for c in vx_ch], dtype=np.float64) if vx_ch else np.array([], dtype=np.float64)
    vy_arr = np.array([c.values[0] for c in vy_ch], dtype=np.float64) if vy_ch else np.array([], dtype=np.float64)

    last_track_num = len(u_arr)

    if len(vx_arr) > 0 and len(vy_arr) > 0:
        avg_parallax = float(np.mean(np.sqrt(vx_arr**2 + vy_arr**2)))
    else:
        avg_parallax = 0.0

    if hasattr(feature_cloud_msg, 'points') and len(feature_cloud_msg.points) > 0:
        depths = np.array([p.x for p in feature_cloud_msg.points], dtype=np.float64)
        depth_ratio = float(np.mean(depths > 0))
    else:
        depth_ratio = 0.0

    return {"last_track_num": last_track_num, "avg_parallax": avg_parallax, "depth_ratio": depth_ratio}
