import argparse
import torch
from matplotlib import pyplot as plt
from punches_lib import datasets
from punches_lib.ii_loss import ii_loss, models, train, eval as eval_ii
from punches_lib.radam import RAdam

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training (default: 32).")
    parser.add_argument("--num_classes", type=int, default=19, help="number of classes in the dataset (default: 19).")
    parser.add_argument("--root_valid", type=str, default="data/test", help="root of testing data (default: data/test).")
    parser.add_argument("--root_crops", type=str, default="data/openset", help="root of ood data (default: data/openset).")
    parser.add_argument("--pretrained_params_path", type=str, default=None, help="path to pretrained params. Ignored if --use_pretrained is not set. If --use_pretrained is set and this arg is left to None, defaults to loading the ImageNet-pretrained params from torchvision (default: None).")
    parser.add_argument("--model_class", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"],help="model class (default: resnet18).")
    parser.add_argument("--dim_latent", type=int, default=32, help="dimension of latent space (default: 32).")
    parser.add_argument("--device", type=str, default=None, help="device to use (default: None -> use CUDA if available).")
    parser.add_argument("--mean_embedding_path", type=str, default=None, help="path to the mean embedding. If specified, will use this mean embedding instead of computing the mean embedding from the training data. (default: None).")
    parser.add_argument("--root_train", type=str, default=None, help="root of training data, to use in case the mean embeddings are not provided (default: None).")
    parser.add_argument("--base_path", type=str, default="model/model_ii.pth", help="path to save the scores. _valid.pth and _crops.pth will be added to the filename (default: model/model.pth).")
    parser.add_argument("--calc_valid_accuracy", action="store_true", help="if set, will calculate the accuracy of the model on the validation set (default: False).")
    return parser.parse_args()

def main():
    args = get_args()

    assert args.mean_embedding_path is not None or args.root_train is not None, "Need at least one of mean_embedding_path or root_train to be provided. Both are None."
    assert args.mean_embedding_path is None or args.root_train is None, f"Only one of mean_embedding_path ({args.mean_embedding_path}) or root_train ({args.root_train}) can be provided."

    net = models.ResNetCustom(args.num_classes, args.model_class, dim_latent=args.dim_latent)
    net.load_state_dict(torch.load(args.pretrained_params_path))

    if args.root_train is not None:
        trainloader = datasets.get_dataloader(args.root_train, args.batch_size, shuffle=False, transforms=datasets.get_bare_transforms())
        mean_embeddings = eval_ii.get_mean_embeddings(trainloader, net, device=args.device)
    else:
        mean_embeddings = torch.load(args.mean_embedding_path)

    validloader = datasets.get_dataloader(args.root_valid, args.batch_size, shuffle=False, transforms=datasets.get_bare_transforms())
    cropsloader = datasets.get_dataloader(args.root_crops, args.batch_size, shuffle=False, transforms=datasets.get_bare_transforms())

    outlier_scores_valid = eval_ii.eval_outlier_scores(validloader, net, mean_embeddings, device=args.device)
    torch.save(outlier_scores_valid, f"{args.base_path}_valid.pth")

    outlier_scores_crops = eval_ii.eval_outlier_scores(cropsloader, net, mean_embeddings, device=args.device)
    torch.save(outlier_scores_crops, f"{args.base_path}_crops.pth")

    if args.calc_valid_accuracy:
        eval_ii.test_model(net, validloader, device=args.device)

if __name__ == "__main__":
    main()




    

    
        