import argparse

import pandas as pd
import torch
from matplotlib import pyplot as plt

from punches_lib import datasets
from punches_lib.ii_loss import eval as eval_ii
from punches_lib.ii_loss import models
from punches_lib.radam import RAdam


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_outlier_scores_valid", type=str, default="model/model_ii.pth_valid.pth", help="path to the outlier scores of the validation set (default: model/model_ii.pth_valid.pth).")
    parser.add_argument("--path_outlier_scores_crops", type=str, default="model/model_ii.pth_crops.pth", help="path to the outlier scores of the ood set (default: model/model_ii.pth_crops.pth).")
    parser.add_argument("--path_outlier_scores_ood", type=str, default="model/model_ii.pth_ood.pth", help="path to the outlier scores of the ood set (default: model/model_ii.pth_ood.pth).")
    parser.add_argument("--by", type=float, default=0.5, help="interval for threshold grid search (default: 0.5).")
    parser.add_argument("--save_path", type=str, default="model/model_ii_results.csv", help="path where the results will be stored as a csv (default: model/model_ii_results.csv).")
    return parser.parse_args()

def main():
    args = get_args()

    scores_valid = torch.load(args.path_outlier_scores_valid, map_location="cpu")
    scores_crops = torch.load(args.path_outlier_scores_crops, map_location="cpu")
    scores_ood = torch.load(args.path_outlier_scores_ood, map_location="cpu")

    all_scores = torch.cat((scores_valid, scores_crops, scores_ood))
    max_score = all_scores.max().item()

    num_steps = int(max_score / args.by) + 1
    thresholds = torch.linspace(0, max_score, num_steps)
    eval_results = eval_ii.eval_multiple_outlier_scores_series_on_thresholds((scores_valid, scores_crops, scores_ood), (torch.lt, torch.gt, torch.gt), thresholds, series_names=("validation", "crops", "ood"))

    eval_results = pd.DataFrame(eval_results)
    eval_results["recall"] = eval_results.validation_N_corr / (eval_results.validation_N_corr + (eval_results.ood_N - eval_results.ood_N_corr))
    eval_results["f1"] = 2 * eval_results.recall * eval_results.validation_pct / (eval_results.recall + eval_results.validation_pct)
    eval_results["sens_spec"] = 2 * eval_results.validation_pct * eval_results.ood_pct / (eval_results.validation_pct + eval_results.ood_pct)

    
    print(eval_results.nlargest(1, "sens_spec"))
    eval_results.to_csv(args.save_path)
    

if __name__ == "__main__":
    main()
