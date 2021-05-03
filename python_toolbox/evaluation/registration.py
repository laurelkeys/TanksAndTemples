# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

import os
import copy

import numpy as np
import open3d as o3d

from trajectory_io import convert_trajectory_to_pointcloud

try:
    o3d_registration = o3d.registration
except AttributeError:
    o3d_registration = o3d.pipelines.registration

MAX_POINT_NUMBER = 4e6


def read_mapping(filename):
    mapping = []
    with open(filename, "r") as f:
        n_sampled_frames = int(f.readline())
        n_total_frames = int(f.readline())
        mapping = np.zeros(shape=(n_sampled_frames, 2))
        metastr = f.readline()
        for iter in range(n_sampled_frames):
            metadata = list(map(int, metastr.split()))
            mapping[iter, :] = metadata
            metastr = f.readline()
    return [n_sampled_frames, n_total_frames, mapping]


def gen_sparse_trajectory(mapping, f_trajectory):
    sparse_traj = []
    for m in mapping:
        sparse_traj.append(f_trajectory[int(m[1] - 1)])
    return sparse_traj


def trajectory_alignment(map_file, traj_to_register, gt_traj_col, gt_trans):
    traj_pcd_col = convert_trajectory_to_pointcloud(gt_traj_col)
    traj_pcd_col.transform(gt_trans)

    correspondence = o3d.utility.Vector2iVector(
        np.asarray([[x, x] for x in range(len(gt_traj_col))])
    )

    rr = o3d_registration.RANSACConvergenceCriteria()
    rr.max_iteration = 100000
    try:
        rr.max_validation = 100000  # open3d <0.12
    except AttributeError:
        rr.confidence = 0.999

    if len(traj_to_register) > 1600:
        # in this case a log file was used which contains
        # every movie frame (see tutorial for details)
        if map_file is None:
            assert gt_traj_col.endswith("_COLMAP_SfM.log")
            map_file = gt_traj_col[: -len("_COLMAP_SfM.log")] + "_mapping_reference.txt"
        n_sampled_frames, n_total_frames, mapping = read_mapping(map_file)
        traj_col2 = gen_sparse_trajectory(mapping, traj_to_register)
        traj_to_register_pcd = convert_trajectory_to_pointcloud(traj_col2)
    else:
        traj_to_register_pcd = convert_trajectory_to_pointcloud(traj_to_register)

    randomvar = 0.0
    nr_of_cam_pos = len(traj_to_register_pcd.points)
    rand_number_added = np.asanyarray(traj_to_register_pcd.points) * (
        np.random.rand(nr_of_cam_pos, 3) * randomvar - randomvar / 2.0 + 1
    )

    traj_to_register_pcd_rand = o3d.geometry.PointCloud()
    for elem in list(rand_number_added):
        traj_to_register_pcd_rand.points.append(elem)

    # Rough registration based on aligned colmap SfM data
    reg = o3d_registration.registration_ransac_based_on_correspondence(
        source=traj_to_register_pcd_rand,
        target=traj_pcd_col,
        corres=correspondence,
        max_correspondence_distance=0.2,
        estimation_method=o3d_registration.TransformationEstimationPointToPoint(with_scaling=True),
        ransac_n=6,
        criteria=rr,
    )
    return reg.transformation


def crop_and_downsample(
    pcd,
    crop_volume,
    down_sample_method="voxel",
    voxel_size=0.01,
    trans=np.identity(4),
):
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(trans)
    pcd_crop = crop_volume.crop_point_cloud(pcd_copy)
    if down_sample_method == "voxel":
        # return voxel_down_sample(pcd_crop, voxel_size)
        return pcd_crop.voxel_down_sample(voxel_size)
    elif down_sample_method == "uniform":
        n_points = len(pcd_crop.points)
        if n_points > MAX_POINT_NUMBER:
            ds_rate = int(round(n_points / float(MAX_POINT_NUMBER)))
            return pcd_crop.uniform_down_sample(ds_rate)
    return pcd_crop


def registration_unif(
    source,
    gt_target,
    init_trans,
    crop_volume,
    threshold,
    max_itr,
    max_size=4 * MAX_POINT_NUMBER,
    verbose=True,
):
    if verbose:
        print("[Registration] threshold: %f" % threshold)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    s = crop_and_downsample(
        source,
        crop_volume,
        down_sample_method="uniform",
        trans=init_trans,
    )
    t = crop_and_downsample(
        gt_target,
        crop_volume,
        down_sample_method="uniform",
        trans=np.identity(4),
    )
    reg = o3d_registration.registration_icp(
        s,
        t,
        max_correspondence_distance=threshold,
        init=np.identity(4),
        estimation_method=o3d_registration.TransformationEstimationPointToPoint(with_scaling=True),
        criteria=o3d_registration.ICPConvergenceCriteria(1e-6, max_itr),
    )
    reg.transformation = np.matmul(reg.transformation, init_trans)
    return reg


def registration_vol_ds(
    source,
    gt_target,
    init_trans,
    crop_volume,
    voxel_size,
    threshold,
    max_itr,
    verbose=True,
):
    if verbose:
        print("[Registration] voxel_size: %f, threshold: %f" % (voxel_size, threshold))
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    s = crop_and_downsample(
        source,
        crop_volume,
        down_sample_method="voxel",
        voxel_size=voxel_size,
        trans=init_trans,
    )
    t = crop_and_downsample(
        gt_target,
        crop_volume,
        down_sample_method="voxel",
        voxel_size=voxel_size,
        trans=np.identity(4),
    )
    reg = o3d_registration.registration_icp(
        s,
        t,
        max_correspondence_distance=threshold,
        init=np.identity(4),
        estimation_method=o3d_registration.TransformationEstimationPointToPoint(with_scaling=True),
        criteria=o3d_registration.ICPConvergenceCriteria(1e-6, max_itr),
    )
    reg.transformation = np.matmul(reg.transformation, init_trans)
    return reg
