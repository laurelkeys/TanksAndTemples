import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_NAME = "Ignatius"

try:
    sys.path.append(os.path.dirname(os.path.dirname(HERE)))
except Exception as e:
    raise e
finally:
    from tanksandtemples_evaluator import TanksAndTemplesEvaluator


def main(dTau, out_dir, plot_stretch):
    print("===========================")
    print("Evaluating %s" % SCENE_NAME)
    print("===========================")

    gt_ply_path = os.path.join(HERE, "Ignatius.ply")
    gt_log_path = os.path.join(HERE, "Ignatius_COLMAP_SfM.log")
    est_ply_path = os.path.join(HERE, "Ignatius_COLMAP.ply")
    est_log_path = os.path.join(HERE, "Ignatius_COLMAP_SfM.log")
    align_txt_path = os.path.join(HERE, "Ignatius_trans.txt")
    crop_json_path = os.path.join(HERE, "Ignatius.json")

    print(est_ply_path)  # estimate reconstruction, i.e. source
    print(gt_ply_path)  # ground-truth, i.e. reference / target

    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = TanksAndTemplesEvaluator.evaluate_reconstruction(
        SCENE_NAME,
        out_dir,
        dTau,
        gt_ply_path,
        gt_log_path,
        est_ply_path,
        est_log_path,
        align_txt_path,
        crop_json_path,
        plot_stretch,
        map_file=None,
        verbose=True,
    )

    print("==============================")
    print("evaluation result : %s" % SCENE_NAME)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % precision)
    print("recall : %.4f" % recall)
    print("f-score : %.4f" % fscore)
    print("==============================")

    TanksAndTemplesEvaluator.plot_graph(
        SCENE_NAME,
        fscore,
        dTau,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
        show_figure=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Runs the single-file evaluation script on {SCENE_NAME}"
    )

    parser.add_argument("--dtau", type=float, default=0.003)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--plot_stretch", type=int, default=5)

    args = parser.parse_args()

    dTau = args.dtau
    out_dir = args.out_dir
    plot_stretch = args.plot_stretch

    if out_dir is None:
        out_dir = os.path.join(HERE, "evaluation")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    main(dTau, out_dir, plot_stretch)

# NOTE you can compare the results of tanksandtemples_evaluator.py with the original
# evaluation scripts by running the following commands in python_toolbox/evaluation/:
# python run.py --dataset-dir ../../test_data/Ignatius
#               --traj-path ../../test_data/Ignatius/Ignatius_COLMAP_SfM.log
#               --ply-path ../../test_data/Ignatius/Ignatius_COLMAP.ply
