import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import argparse
from typing import Union

from punches_lib.gan import models, features,  testing, utils
from punches_lib import datasets



def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discriminator_path", type=str, required=True, help="Path where the params for the discriminator are stored.")
    parser.add_argument("--input_channel_dim", type=int, default=512, help="Number of channels of the input data (the features representation of the images --- default: 512).")
    parser.add_argument("--base_width", type=int, default=64, help="Base width (i.e., minimum number of output channels) per hidden conv layer in discriminator and generator (default: 64).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluating the features (default: 128).")
    parser.add_argument("--validset_root", type=str, default=None, help="Root where the validation data are stored (default: None).")
    parser.add_argument("--openset_root", type=str, default=None, help="Root where the open data are stored (default: None).")
    parser.add_argument("--path_features_train", type=str, default=None, help="Path from where the features for training the GAN will be loaded from. If None, features will be calculated at runtime but not saved. Features must be loaded and cannot be recalculated.")
    parser.add_argument("--path_features_valid", type=str, default=None, help="Path from where the features for the validation dataset will be loaded from. If None, features will be calculated at runtime but not saved. For recalculating the features and saving them in this path, toggle the switch --force_feats_recalculation (default: None).")
    parser.add_argument("--path_features_open", type=str, default=None, help="Path from where the features for the open data dataset will be loaded from. If None, features will be calculated at runtime but not saved. For recalculating the features and saving them in this path, toggle the switch --force_feats_recalculation (default: None).")
    parser.add_argument("--force_feats_recalculation", action="store_true", default=False, help="Force recalculation of the features even if --backbone_network_feats is passed")
    parser.add_argument("--backbone_network_feats", type=str, choices=["resnet18", "resnet34", "resnet50", None], default=None, help="Backbone network for obtaining the features (default: None).")
    parser.add_argument("--backbone_network_params", type=str, default=None, help="Path to the state_dict containing the parameters for the pretrained backbone (default: None)")
    parser.add_argument("--folder_save_outputs", type=str, default=None, help="Folder where to save the outputs after discriminator evaluation (default: None).")
    parser.add_argument("--save_hist_path", type=str, default="hist.png", help="Path where to save the histogram (default: hist.png).")
    parser.add_argument("--by", type=float, default=0.05, help="Calculation of performance: increment used for swiping the axis 0-1 in search of a threshold (default: 0.05).")
    parser.add_argument("--save_performance_path", type=str, default="performance.csv", help="Path where to save the performance as a CSV file (default: performance.csv).")
    args = parser.parse_args()
    return args

def main():
    args = load_args()
    print(args)

    # INSTANTIATE DISCRIMINATOR AND LOAD WEIGHTS
    netD = models.DiscriminatorFunnel(num_channels=args.input_channel_dim, base_width=args.base_width)
    x = torch.load(args.discriminator_path)
    netD.load_state_dict(torch.load(args.discriminator_path))

    # OBTAIN THE FEATURES AND DATALOADERS
    dataset_valid = datasets.get_dataset(args.validset_root, transforms=datasets.get_bare_transforms()) if args.validset_root is not None else None
    dataset_open = datasets.get_dataset(args.openset_root, transforms=datasets.get_bare_transforms()) if args.openset_root is not None else None
    valid_features = features.get_features(args.path_features_valid, args.force_feats_recalculation, dataset_valid, args.backbone_network_feats, args.backbone_network_params, args.batch_size, num_classes=19) if dataset_valid is not None else None
    open_features = features.get_features(args.path_features_open, args.force_feats_recalculation, dataset_open, args.backbone_network_feats, args.backbone_network_params, args.batch_size, num_classes=19) if dataset_open is not None else None
    train_features = features.get_features(args.path_features_train, False, None, args.backbone_network_feats, args.backbone_network_params, args.batch_size, num_classes=19) if args.path_features_train is not None else None

    trainloader = DataLoader(datasets.BasicDataset(train_features), batch_size=args.batch_size, shuffle=False, num_workers=4) if train_features is not None else None
    validloader = DataLoader(datasets.BasicDataset(valid_features), batch_size=args.batch_size, shuffle=False, num_workers=4) if valid_features is not None else None
    openloader = DataLoader(datasets.BasicDataset(open_features), batch_size=args.batch_size, shuffle=False, num_workers=4) if open_features is not None else None
    
    outs_train = testing.get_outputs(netD, trainloader).squeeze() if trainloader is not None else None
    outs_valid = testing.get_outputs(netD, validloader).squeeze() if validloader is not None else None
    outs_open = testing.get_outputs(netD, openloader).squeeze() if openloader is not None else None


    if (fold:=args.folder_save_outputs) is not None:
        os.makedirs(fold, exist_ok=True)
        torch.save(outs_train, os.path.join(fold, "train.pt"))
        torch.save(outs_valid, os.path.join(fold, "validouts_valid.pt"))
        torch.save(outs_open, os.path.join(fold, "open.pt"))

    if (fold:=os.path.dirname(args.save_hist_path)) != "":
        os.makedirs(fold, exist_ok=True)
    testing.plot_hist(outs_open, outs_valid, args.save_hist_path, "Discriminator validation")

    perf = testing.get_performance(outs_valid, outs_open, increment=args.by)
    if (fold:=os.path.dirname(args.save_performance_path)) != "":
        os.makedirs(fold, exist_ok=True)
    perf.to_csv(args.save_performance_path)

    print("Best\n", perf.nlargest(1, "W"))
    
if __name__ == "__main__":
    main()
