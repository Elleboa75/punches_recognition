import argparse
from sqlite3 import paramstyle

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
    parser.add_argument("--save_path", type=str, default="model/model_ii_hist.png", help="path where the hist will be saved (default: model/model_ii_hist.png).")
    plt.hist()
    return parser.parse_args()

def main():
    args = get_args()

    scores_valid = torch.load(args.path_outlier_scores_valid)
    scores_crops = torch.load(args.path_outlier_scores_crops)
    scores_ood = torch.load(args.path_outlier_scores_ood)

    plt.hist(scores_valid, bins=100, label="valid", alpha=0.5, density=True)
    plt.hist(scores_crops, bins=100, label="crops", alpha=0.5, density=True)
    plt.hist(scores_ood, bins=100, label="ood", alpha=0.5, density=True)
    plt.legend(loc="upper right")
    plt.savefig(args.save_path)