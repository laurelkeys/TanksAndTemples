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
import argparse

# this script requires Open3D python binding
# please follow the intructions in setup.py before running this script.
import numpy as np
import open3d as o3d

from plot import plot_graph
from config import scenes_tau_dict
from evaluation import EvaluateHisto
from registration import registration_unif, registration_vol_ds, trajectory_alignment
from trajectory_io import read_trajectory


def evaluate(
    scene_name,
    out_dir,
    dTau,
    gt_ply_path,  # Ground-truth (gt)
    gt_log_path,
    est_ply_path,  # Estimate (est) reconstruction
    est_log_path,
    alignment_txt_path,  # Transformation matrix to align 'est' with 'gt'
    crop_json_path,  # Area cropping for the 'gt' point cloud
    plot_stretch,
    map_file=None,
):
    # Load reconstruction and according ground-truth
    est_pcd = o3d.io.read_point_cloud(est_ply_path)  # "source"
    gt_pcd = o3d.io.read_point_cloud(gt_ply_path)  # "target"

    transform = np.loadtxt(alignment_txt_path)  # ('gt_trans')
    est_traj = read_trajectory(est_log_path)  # ('traj_to_register')
    gt_traj = read_trajectory(gt_log_path)  # ('gt_traj_col')

    traj_transformation = trajectory_alignment(map_file, est_traj, gt_traj, transform)

    # Refine alignment by using the actual 'gt' and 'est' point clouds
    # Big pointclouds will be downsampled to 'dTau' to speed up alignment
    vol = o3d.visualization.read_selection_polygon_volume(crop_json_path)

    # Registration refinment in 3 iterations
    r2 = registration_vol_ds(
        est_pcd,
        gt_pcd,
        init_trans=traj_transformation,
        crop_volume=vol,
        voxel_size=dTau,
        threshold=dTau * 80,
        max_itr=20,
    )
    r3 = registration_vol_ds(
        est_pcd,
        gt_pcd,
        init_trans=r2.transformation,
        crop_volume=vol,
        voxel_size=dTau / 2,
        threshold=dTau * 20,
        max_itr=20,
    )
    r = registration_unif(
        est_pcd,
        gt_pcd,
        init_trans=r3.transformation,
        crop_volume=vol,
        threshold=2 * dTau,
        max_itr=20,
    )

    # Generate histograms and compute P/R/F1
    # [precision, recall, fscore, edges_source, cum_source, edges_target, cum_target]
    return EvaluateHisto(
        est_pcd,
        gt_pcd,
        trans=r.transformation,
        crop_volume=vol,
        voxel_size=dTau / 2,
        threshold=dTau,
        filename_mvs=out_dir,
        plot_stretch=plot_stretch,
        scene_name=scene_name,
    )


def run_evaluation(dataset_dir, traj_path, ply_path, out_dir, scene_dTau=None):
    scene = os.path.basename(os.path.normpath(dataset_dir))

    if scene_dTau is None and scene not in scenes_tau_dict:
        print(dataset_dir, scene)
        raise Exception("invalid dataset-dir, not in scenes_tau_dict")

    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    dTau = scene_dTau if scene_dTau is not None else scenes_tau_dict[scene]
    # put the crop-file, the GT file, the COLMAP SfM log file and
    # the alignment of the according scene in a folder of
    # the same scene name in the dataset_dir
    colmap_ref_logfile = os.path.join(dataset_dir, scene + "_COLMAP_SfM.log")
    alignment = os.path.join(dataset_dir, scene + "_trans.txt")
    gt_filen = os.path.join(dataset_dir, scene + ".ply")
    cropfile = os.path.join(dataset_dir, scene + ".json")
    map_file = os.path.join(dataset_dir, scene + "_mapping_reference.txt")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(ply_path)
    print(gt_filen)
    plot_stretch = 5

    [precision, recall, fscore, edges_source, cum_source, edges_target, cum_target] = evaluate(
        scene,
        out_dir,
        dTau,
        gt_ply_path=gt_filen,
        gt_log_path=colmap_ref_logfile,
        est_ply_path=ply_path,
        est_log_path=traj_path,
        alignment_txt_path=alignment,
        crop_json_path=cropfile,
        plot_stretch=plot_stretch,
        map_file=map_file,
    )

    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % precision)
    print("recall : %.4f" % recall)
    print("f-score : %.4f" % fscore)
    print("==============================")

    plot_graph(
        scene,
        fscore,
        dTau,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="path to a dataset/scene directory containing X.json, X.ply, ...",
    )
    parser.add_argument(
        "--traj-path",
        type=str,
        required=True,
        help="path to trajectory file (see `convert_to_logfile.py` to create this file)",
    )
    parser.add_argument(
        "--ply-path",
        type=str,
        required=True,
        help="path to reconstruction ply file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="output directory (default: an evaluation directory is created in the directory of the ply file)",
    )
    args = parser.parse_args()

    if args.out_dir.strip() == "":
        args.out_dir = os.path.join(os.path.dirname(args.ply_path), "evaluation")

    run_evaluation(
        dataset_dir=args.dataset_dir,
        traj_path=args.traj_path,
        ply_path=args.ply_path,
        out_dir=args.out_dir,
    )
