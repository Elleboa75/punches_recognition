'''
Inspired by OpenGAN: https://github.com/aimerykong/OpenGAN/
'''
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import argparse
from typing import Union

from punches_lib.gan import models, features, train, utils, models_experimental as me
from punches_lib import datasets


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for the GAN training (default: 300).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128).") 
    parser.add_argument("--generator_type", type=str, default="classic", choices=["classic", "stretch", "stretch_small", "stretch_super_small", "experimental"], help="Type of generator to use. 'classic' leaves the spatial dimension untouched, while 'stretch' increases the spatial dimensions from 1 to 8 (default: classic).")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimension of the latent space for the generator (default: 100).")
    parser.add_argument("--base_width", type=int, default=64, help="Base width (i.e., minimum number of output channels) per hidden conv layer in discriminator and generator (default: 64).")
    parser.add_argument("--input_channel_dim", type=int, default=512, help="Number of channels of the input data (the features representation of the images --- default: 512).")
    parser.add_argument("--lr", type=float, default=0.0001, help="The base learning rate for the generator (default: 0.0001)")
    parser.add_argument("--lr_modifier_D", type=float, default=0.2, help="The learning rate modifier for the discriminator, i.e., lr_discriminator = --lr * --lr_modifier_D (default: 0.2)")
    parser.add_argument("--adam_beta1", type=float, default=0.5, help="Value for Adam's beta1 hyperparameter (default: 0.5)")
    parser.add_argument("--adam_beta2", type=float, default=.999, help="Value for Adam's beta1 hyperparameter (default: 0.999)")
    parser.add_argument("--label_smoothing", type=float, default=0.15, help="Label smoothing. Leave to 0 for no effect. (default: 0.15)")
    parser.add_argument("--model_name", type=str, default="GAN", help="Model name (used for saving purposes --- default: 'GAN').")
    parser.add_argument("--save_folder_params", type=str, default="models", help="Folder where the parameters will be saved. The name of the file will be `<model_name>_id.pt`, where id is a progressive int starting from 0 (default: 'models').")
    parser.add_argument("--path_features_train", type=str, default=None, help="Path from where the features for training the GAN will be loaded from. If None, features will be calculated at runtime but not saved. For recalculating the features and saving them in this path, toggle the switch --force_feats_recalculation (default: None).")
    parser.add_argument("--backbone_network_feats", type=str, choices=["resnet18", "resnet34", "resnet50", None], default=None, help="Backbone network for obtaining the features (default: None).")
    parser.add_argument("--backbone_network_params", type=str, default=None, help="Path to the state_dict containing the parameters for the pretrained backbone (default: None)")
    parser.add_argument("--trainset_root", type=str, default=None, help="root where the data are stored (default: None).")
    parser.add_argument("--force_feats_recalculation", action="store_true", default=False, help="Force recalculation of the features even if --backbone_network_feats is passed")
    parser.add_argument("--save_figure_path", type=str, default=None, help="Path where the figure for the losses (default: None).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (default: cuda).")
    args = parser.parse_args()
    return args

def check_args(args):
    if args.path_features_train is None or args.force_feats_recalculation:
        assert args.backbone_network_feats is not None and args.backbone_network_params is not None and args.trainset_root is not None, f"If --path_features_train is not specified, then all of --beckbone_network_feats, --backbone_network_params, and --trainset_root must be specified."
  
def main():
    args = load_args()
    print(args)

    # INSTANTIATE THE MODELS
    discriminator_class = models.DiscriminatorFunnel
    if args.generator_type == "classic":
        generator_class = models.Generator
        spatial_dim_noise = 8
    elif args.generator_type == "stretch":
        generator_class = models.GeneratorStretch
        spatial_dim_noise = 1
    elif args.generator_type == "stretch_small":
        generator_class = models.GeneratorStretchSmall
        spatial_dim_noise = 1
    elif args.generator_type == "stretch_super_small":
        generator_class = models.GeneratorStretchSuperSmall
        spatial_dim_noise = 2
    elif args.generator_type == "experimental":
        generator_class = me.Generator
        discriminator_class = me.Discriminator
        spatial_dim_noise = 1
    else:
        raise ValueError(f"Invalid generator_type {args.generator_type}")
    netG = generator_class(latent_dim=args.latent_dim, base_width=args.base_width)
    netG.apply(models.weights_init)
    netD = discriminator_class(num_channels=args.input_channel_dim, base_width=args.base_width)
    netD.apply(models.weights_init)

    trainset = datasets.get_dataset(args.trainset_root, transforms=datasets.get_bare_transforms())
    train_features = features.get_features(args.path_features_train, args.force_feats_recalculation, trainset, args.backbone_network_feats, args.backbone_network_params, args.batch_size)
    
    # PREPARE THE TRAINING
    trainloader = DataLoader(datasets.BasicDataset(train_features), batch_size=args.batch_size, shuffle=True, num_workers=4)
    adam_beta = (args.adam_beta1, args.adam_beta2)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr*args.lr_modifier_D, betas=adam_beta)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=adam_beta)

    params_savefile = os.path.join(args.save_folder_params, args.model_name)

    G_losses, D_losses = train.train(
        generator=netG,
        discriminator=netD,
        optimizer_g=optimizerG,
        optimizer_d=optimizerD, 
        trainloader=trainloader,
        epochs=args.epochs,
        device="cuda",
        label_smoothing_factor=args.label_smoothing,
        loss_fn=torch.nn.BCELoss(),
        latent_dim=args.latent_dim,
        save_path=params_savefile,
        spatial_dim_noise=spatial_dim_noise
    )

    if args.save_figure_path is not None:
        utils.loss_plotter(loss_g=G_losses, loss_d=D_losses, figpath=args.save_figure_path)
    
if __name__ == "__main__":
    main()

    
