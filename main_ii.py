import argparse
import torch
from matplotlib import pyplot as plt
from punches_lib import datasets
from punches_lib.ii_loss import ii_loss, models, train, eval as eval_ii
from punches_lib.cnn import eval
from punches_lib.radam import RAdam

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training (default: 32).")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train (default: 20).")
    parser.add_argument("--dim_latent", type=int, default=32, help="dimension of the latent space where II-loss is computed (default: 32).")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001).")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.1, help="learning rate decay factor (default: 0.1).")
    parser.add_argument("--lr_decay_epochs", type=int, nargs="*", default=[10, 15], help="learning rate decay epochs (default: 10 and 15).")
    parser.add_argument("--lambda_ii", type=float, default=1, help="weight of the II-loss (default: 1).")
    parser.add_argument("--root_train", type=str, default="data/train", help="root of training data (default: data/train).")
    parser.add_argument("--root_test", type=str, default="data/test", help="root of testing data (default: data/test).")
    parser.add_argument("--root_openset", type=str, default="data/openset", help="root of ood data (default: data/openset).")
    parser.add_argument("--model_path", type=str, default="model/model_ii.pth", help="path to save model (default: model/model.pth).")
    parser.add_argument("--use_pretrained", action="store_true", default=False, help="use pretrained model. The weigths used depend on the next arg (default: False).")
    parser.add_argument("--pretrained_params_path", type=str, default=None, help="path to pretrained params. Ignored if --use_pretrained is not set. If --use_pretrained is set and this arg is left to None, defaults to loading the ImageNet-pretrained params from torchvision (default: None).")
    parser.add_argument("--model_class", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"],help="model class (default: resnet18).")
    parser.add_argument("--device", type=str, default=None, help="device to use (default: None -> use CUDA if available).")
    return parser.parse_args()

def main():
    args = get_args()
    trainloader = datasets.get_dataloader(args.root_train, args.batch_size, num_workers=8, transforms=datasets.get_bare_transforms())
    num_classes = len(trainloader.dataset.classes)

    pretrained = args.use_pretrained
    pretrained_params = None
    if args.use_pretrained and args.pretrained_params_path is not None:
        pretrained = False
        pretrained_params = torch.load(args.pretrained_params_path, map_location="cpu")
    elif args.use_prerained:
        raise NotImplementedError("Loading pretrained params from torchvision is not implemented yet.")
    net = models.ResNetCustom(num_classes, args.model_class, dim_latent=args.dim_latent)
    if pretrained_params is not None:
        # load state dict with strict set to False because the model has 2 FC heads instead of 1
        net.load_state_dict(pretrained_params, strict=False)


    ii_loss_fn = ii_loss.IILoss()
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = RAdam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_gamma)

    train.train_model(net, trainloader, ii_loss_fn, ce_loss_fn, args.epochs, optimizer, scheduler, args.device, lambda_scale=args.lambda_ii)
    torch.save(net.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

    train_data_means = eval_ii.get_mean_embeddings(trainloader, net, device=args.device)

    testloader = datasets.get_dataloader(args.root_test, args.batch_size, num_workers=8, transforms=datasets.get_bare_transforms(), shuffle=False)
    # for now, test only on accuracy
    eval.test_model(net, testloader, args.device)

    outlier_scores_test = eval_ii.eval_outlier_scores(testloader, net, train_data_means, device=args.device)

    extraloader = datasets.get_dataloader(args.root_openset, args.batch_size, num_workers=8, transforms=datasets.get_bare_transforms(), shuffle=False)

    outlier_scores_extra = eval_ii.eval_outlier_scores(extraloader, net, train_data_means, device=args.device)

    plt.hist(outlier_scores_test.detach().cpu().numpy(), bins=100, alpha=.5, label="test")
    plt.hist(outlier_scores_extra.detach().cpu().numpy(), bins=100, alpha=.5, label="extra")
    plt.legend(loc="upper right")
    plt.savefig("outlier_scores.png")



if __name__ == "__main__":
    main()