import argparse

import torch
from matplotlib import pyplot as plt

from punches_lib import datasets
from punches_lib.ii_loss import eval as eval_ii
from punches_lib.ii_loss import models
from punches_lib.radam import RAdam

from punches_lib import utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training (default: 128).")
    parser.add_argument("--num_classes", type=int, default=19, help="number of classes in the dataset (default: 19).")
    parser.add_argument("--classif_threshold", type=float, required=True, help="Classification threshold acting on outlier score (if outlier score larger than threshold, instance is classified as OOD).")
    parser.add_argument("--root_test", type=str, default="data/test", help="root of testing data (default: data/test).")
    parser.add_argument("--root_ood_test", type=str, default="data/openset", help="root of ood data for testing (default: data/openset_test).")
    parser.add_argument("--root_crops", type=str, default="data/crops", help="root of crops data (default: data/crops).")
    parser.add_argument("--pretrained_params_path", type=str, default=None, help="path to pretrained params. Ignored if --use_pretrained is not set. If --use_pretrained is set and this arg is left to None, defaults to loading the ImageNet-pretrained params from torchvision (default: None).")
    parser.add_argument("--model_class", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"],help="model class (default: resnet18).")
    parser.add_argument("--dim_latent", type=int, default=32, help="dimension of latent space (default: 32).")
    parser.add_argument("--device", type=str, default=None, help="device to use (default: None -> use CUDA if available).")
    parser.add_argument("--mean_embedding_path", type=str, default=None, help="path to the mean embedding. If specified, will use this mean embedding instead of computing the mean embedding from the training data. (default: None).")
    parser.add_argument("--root_train", type=str, default=None, help="root of training data, to use in case the mean embeddings are not provided (default: None).")
    #parser.add_argument("--base_path", type=str, default="model/model_ii.pth", help="path to save the scores. _valid.pth and _crops.pth will be added to the filename (default: model/model.pth).")
    parser.add_argument("--calc_test_accuracy", action="store_true", help="if set, will calculate the accuracy of the model on the validation set (default: False).")
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

    testloader = datasets.get_dataloader(args.root_test, args.batch_size, shuffle=False, transforms=datasets.get_bare_transforms())
    oodloader = datasets.get_dataloader(args.root_ood_test, args.batch_size, shuffle=False, transforms=datasets.get_bare_transforms())
    cropsloader = datasets.get_dataloader(args.root_crops, args.batch_size, shuffle=False, transforms=datasets.get_bare_transforms())

    outlier_scores_test = eval_ii.eval_outlier_scores(testloader, net, mean_embeddings, device=args.device)
    outlier_scores_ood = eval_ii.eval_outlier_scores(oodloader, net, mean_embeddings, device=args.device)
    outlier_scores_crops = eval_ii.eval_outlier_scores(cropsloader, net, mean_embeddings, device=args.device)

    test_non_ood = outlier_scores_test < args.classif_threshold
    sensitivity = test_non_ood.sum().item() / len(outlier_scores_test)
    specificity = (outlier_scores_ood >= args.classif_threshold).sum().item() / len(outlier_scores_ood)
    sens_spec = 2 * sensitivity * specificity / (sensitivity + specificity)
    print(f"Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f} | Sens<->Spec: {sens_spec:.4f}")

    non_ood_punch_id = {cl: 0 for cl in range(len(testloader.dataset.classes))}
    for outlier_score, punch_id in zip(outlier_scores_test, testloader.dataset.targets):
        if outlier_score < args.classif_threshold:
            non_ood_punch_id[punch_id] += 1


    class_id_to_punch_id = {id:name for name, id in testloader.dataset.class_to_idx.items()}
    print("PER-PUNCH OOD ACCURACY")
    targets = torch.Tensor(testloader.dataset.targets).int()
    for cl, count in non_ood_punch_id.items():
        num_items = len(targets[targets == cl])
        print(f"Class: {cl} [ID: {class_id_to_punch_id[cl]}] - correct: {count} - num items: {num_items}-| accuracy: {count/num_items:.4f}")
  

    crops_accuracy = (outlier_scores_crops >= args.classif_threshold).sum().item() / len(outlier_scores_crops)
    print(f"Crops accuracy: {crops_accuracy:.4f}")


    if args.calc_test_accuracy:
        # print(f"Testing model - {(1-test_non_ood.int()).abs().sum().item()} samples removed from testset")
        # filtered_testset = utils.subset_imagefolder(testloader.dataset, test_non_ood)
        # filtered_testloader = torch.utils.data.DataLoader(filtered_testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        eval_ii.test_model(net, testloader, device=args.device)

if __name__ == "__main__":
    main()
    




