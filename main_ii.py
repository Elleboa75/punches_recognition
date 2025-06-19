import argparse
import torch
from matplotlib import pyplot as plt
from punches_lib import datasets
from punches_lib.ii_loss import ii_loss, models, train, eval as eval_ii
from punches_lib.cnn import eval
from punches_lib.radam import RAdam

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size for training (default: 32).')
    parser.add_argument('--epochs', type = int, default = 20, help = 'number of epochs to train (default: 20).')
    parser.add_argument('--dim_latent', type = int, default = 32, help = 'dimension of the latent space where II-loss is computed (default: 32).')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate (default: 0.001).')
    parser.add_argument('--lr_decay_gamma', type = float, default = 0.1, help = 'learning rate decay factor (default: 0.1).')
    parser.add_argument('--lr_decay_epochs', type = int, nargs = '*', default = [
        10,
        15], help = 'learning rate decay epochs (default: 10 and 15).')
    parser.add_argument('--lambda_ii', type = float, default = 1, help = 'weight of the II-loss (default: 1).')
    parser.add_argument('--delta_ii', type = float, default = float('inf'), help = 'delta (margin) for the II-loss. If infinite, the II-loss is unbounded (default: infinite).')
    parser.add_argument('--root_train', type = str, default = 'data/train', help = 'root of training data (default: data/train).')
    parser.add_argument('--root_test', type = str, default = 'data/test', help = 'root of testing data (default: data/test).')
    parser.add_argument('--root_openset', type = str, default = 'data/openset', help = 'root of ood data (default: data/openset).')
    parser.add_argument('--model_path', type = str, default = 'model/model_ii.pth', help = 'path to save model (default: model/model.pth).')
    parser.add_argument('--use_pretrained', action = 'store_true', default = False, help = 'use pretrained model. The weigths used depend on the next arg (default: False).')
    parser.add_argument('--pretrained_params_path', type = str, default = None, help = 'path to pretrained params. Ignored if --use_pretrained is not set. If --use_pretrained is set and this arg is left to None, defaults to loading the ImageNet-pretrained params from torchvision (default: None).')
    parser.add_argument('--model_class', type = str, default = 'resnet18', choices = [
        'resnet18',
        'resnet34',
        'resnet50'], help = 'model class (default: resnet18).')
    parser.add_argument('--device', type = str, default = None, help = 'device to use (default: None -> use CUDA if available).')
    parser.add_argument('--load_trained_model', type = str, default = None, help = 'path to trained model. Bypasses all training args (default: None).')
    parser.add_argument('--alternate_backprop', action = 'store_true', default = False, help = 'alternate backprop between II Loss and CE Loss (default: False).')
    return parser.parse_args()


def main():

    args = get_args()
    trainloader = datasets.get_dataloader(args.root_train, args.batch_size, num_workers = 8, transforms = datasets.get_bare_transforms())
    num_classes = len(trainloader.dataset.classes)
    net = models.ResNetCustom(num_classes, args.model_class, dim_latent = args.dim_latent)


if __name__ == '__main__':
    main()
