#!/usr/bin/env python3

import numpy as np
import pandas as pd
import small_gicp
import sys

def random_sampling(array, sample_rate):

    num_sample = int(array.shape[0] * sample_rate)

    indices_1 = np.random.choice(array.shape[0], num_sample, replace=False)
    sampled_array1 = array[indices_1]

    indices_2 = np.random.choice(array.shape[0], num_sample, replace=False)
    sampled_array2 = array[indices_2]

    return sampled_array1, sampled_array2

def points_noise(array, scale_translation, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    noise_x = rng.normal(0, scale_translation, array.shape[0])
    noise_y = rng.normal(0, scale_translation, array.shape[0])
    noise_z = rng.normal(0, scale_translation, array.shape[0])

    array_noised = array.copy()
    array_noised[:, 0] += noise_x
    array_noised[:, 1] += noise_y
    array_noised[:, 2] += noise_z

    return array_noised

def calc_localizability(hessian_matrix):
    cov_matrix = np.linalg.pinv(hessian_matrix)

    cov_eigen_value, cov_eigen_vector = np.linalg.eig(cov_matrix)
    localizability_zhen = float(np.min(cov_eigen_value))

    localizability_kondo = float(np.linalg.det(cov_matrix) ** 0.5)
    return localizability_zhen, localizability_kondo

def calc_factor(source_points, target_points, localizability):

    source, source_tree = small_gicp.preprocess_points(source_points, downsampling_resolution=0.3)
    target, target_tree = small_gicp.preprocess_points(target_points, downsampling_resolution=0.3)

    result = small_gicp.align(target, source, target_tree)
    result = small_gicp.align(target, source, target_tree, result.T_target_source)

    factors = [small_gicp.GICPFactor()]
    rejector = small_gicp.DistanceRejector()

    hessian = np.asarray(result.H)

    if localizability:
        l_z, l_k = calc_localizability(hessian)
    else:
        l_z, l_k = 0.0, 0.0

    hessian_rotation    = hessian[0:3, 0:3]
    hessian_translation = hessian[3:6, 3:6]

    rot_eigen_value,    rot_eigen_vector    = np.linalg.eig(hessian_rotation)
    trans_eigen_value,  trans_eigen_vector  = np.linalg.eig(hessian_translation)

    rot_global_min_idx    = np.argmin(rot_eigen_value)
    trans_global_min_idx  = np.argmin(trans_eigen_value)

    rot_global_min_vector   = rot_eigen_vector[:, rot_global_min_idx]
    trans_global_min_vector = trans_eigen_vector[:, trans_global_min_idx]

    list_xyz = []
    list_cov_eigen_value = []

    for i in range(source.size()):
        succ, H, b, e = factors[0].linearize(target, source, target_tree, result.T_target_source, i, rejector)
        if succ:
            H_arr = np.asarray(H)
            rot_point_eigen_value,    rot_point_eigen_vector    = np.linalg.eig(H_arr[0:3, 0:3])
            trans_point_eigen_value,  trans_point_eigen_vector  = np.linalg.eig(H_arr[3:6, 3:6])

            rot_local_max_idx    = np.argmax(rot_point_eigen_value)
            trans_local_max_idx  = np.argmax(trans_point_eigen_value)

            rot_local_max_vector   = rot_point_eigen_vector[:, rot_local_max_idx]
            trans_local_max_vector = trans_point_eigen_vector[:, trans_local_max_idx]

            dp_rot   = np.dot(rot_global_min_vector,   rot_local_max_vector)
            dp_trans = np.dot(trans_global_min_vector, trans_local_max_vector)

            list_cov_eigen_value.append(abs(dp_trans))
            list_xyz.append(source.points()[i, 0:3])

    return np.array(list_xyz), np.array(list_cov_eigen_value), l_z, l_k


def execute_gicp(array, num_iteration, sample_rate, scale_translation):
    rng = np.random.default_rng()

    pc1, pc2 = random_sampling(array, sample_rate)

    # 存储上一次迭代的结果，用于多轮迭代时做平均
    prev_xyz  = None
    prev_eig  = None

    for counter in range(1, int(num_iteration) + 1):
        source = pc1
        target = points_noise(pc2, scale_translation, rng)

        xyz, dot_eig, _, _ = calc_factor(source, target, localizability=False)

        if counter < num_iteration:
            # 中间迭代：保存结果用于后续平均
            prev_xyz = xyz
            prev_eig = dot_eig
        else:
            # 最后一次迭代
            if prev_xyz is not None and prev_xyz.shape == xyz.shape:
                # 多轮迭代：对前后两次结果做平均（平滑噪声）
                final_xyz = (prev_xyz + xyz) * 0.5
                final_eig = (prev_eig + dot_eig) * 0.5
            else:
                # 单轮迭代或形状不匹配：直接用最后一次结果
                final_xyz = xyz
                final_eig = dot_eig

    return final_xyz, final_eig

