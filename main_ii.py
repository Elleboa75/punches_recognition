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
    parser.add_argument("--delta_ii", type=float, default=float("inf"), help="delta (margin) for the II-loss. If infinite, the II-loss is unbounded (default: infinite).")
    parser.add_argument("--root_train", type=str, default="data/train", help="root of training data (default: data/train).")
    parser.add_argument("--root_test", type=str, default="data/test", help="root of testing data (default: data/test).")
    parser.add_argument("--root_openset", type=str, default="data/openset", help="root of ood data (default: data/openset).")
    parser.add_argument("--model_path", type=str, default="model/model_ii.pth", help="path to save model (default: model/model.pth).")
    parser.add_argument("--use_pretrained", action="store_true", default=False, help="use pretrained model. The weigths used depend on the next arg (default: False).")
    parser.add_argument("--pretrained_params_path", type=str, default=None, help="path to pretrained params. Ignored if --use_pretrained is not set. If --use_pretrained is set and this arg is left to None, defaults to loading the ImageNet-pretrained params from torchvision (default: None).")
    parser.add_argument("--model_class", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"],help="model class (default: resnet18).")
    parser.add_argument("--device", type=str, default=None, help="device to use (default: None -> use CUDA if available).")
    parser.add_argument("--load_trained_model", type=str, default=None, help="path to trained model. Bypasses all training args (default: None).")
    parser.add_argument("--alternate_backprop", action="store_true", default=False, help="alternate backprop between II Loss and CE Loss (default: False).")
    return parser.parse_args()

def main():
    args = get_args()
    trainloader = datasets.get_dataloader(args.root_train, args.batch_size, num_workers=8, transforms=datasets.get_bare_transforms())
    num_classes = len(trainloader.dataset.classes)

    net = models.ResNetCustom(num_classes, args.model_class, dim_latent=args.dim_latent)

    if args.load_trained_model is not None:
        net.load_state_dict(torch.load(args.load_trained_model))
    else:
        pretrained = args.use_pretrained
        pretrained_params = None
        if args.use_pretrained and args.pretrained_params_path is not None:
            pretrained = False
            pretrained_params = torch.load(args.pretrained_params_path, map_location="cpu")
        elif args.use_pretrained:
            raise NotImplementedError("Loading pretrained params from torchvision is not implemented yet.")
        
        if pretrained_params is not None:
            # load state dict with strict set to False because the model has 2 FC heads instead of 1
            net.load_state_dict(pretrained_params, strict=False)


        ii_loss_fn = ii_loss.IILoss(delta=args.delta_ii)
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = RAdam(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_gamma)

        train.train_model(net, trainloader, ii_loss_fn, ce_loss_fn, args.epochs, optimizer, scheduler, args.device, lambda_scale=args.lambda_ii, alternate_backprop=args.alternate_backprop)
        torch.save(net.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

    print("Getting trainset means")
    train_data_means = eval_ii.get_mean_embeddings(trainloader, net, device=args.device,)
    torch.save(train_data_means, f"{args.model_path}_means.pth")

    print("Evaluating accuracy on testset")
    testloader = datasets.get_dataloader(args.root_test, args.batch_size, num_workers=8, transforms=datasets.get_bare_transforms(), shuffle=False)
    # for now, test only on accuracy
    #eval.test_model(net, testloader, loss_fn=None, device=args.device)

    print("Getting outlier scores for testset")
    outlier_scores_test = eval_ii.eval_outlier_scores(testloader, net, train_data_means, device=args.device)
    torch.save(outlier_scores_test, f"{args.model_path}_outliers_score_test.pth")

    extraloader = datasets.get_dataloader(args.root_openset, args.batch_size, num_workers=8, transforms=datasets.get_bare_transforms(), shuffle=False)

    print("Getting outlier scores for ood set")
    outlier_scores_extra = eval_ii.eval_outlier_scores(extraloader, net, train_data_means, device=args.device)
    torch.save(outlier_scores_extra, f"{args.model_path}_outliers_score_ood.pth")

    plt.hist(outlier_scores_test.detach().cpu().numpy(), bins=100, alpha=.5, density=True, label="test")
    plt.hist(outlier_scores_extra.detach().cpu().numpy(), bins=100, alpha=.5, density=True, label="extra")
    plt.legend(loc="upper right")
    plt.savefig(f"{args.model_path}_outlier_scores.png")



if __name__ == "__main__":
    main()