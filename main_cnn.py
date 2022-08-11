from webbrowser import get
import torch
import argparse

from punches_lib import models, datasets
from punches_lib.cnn import train, eval

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training (default: 32).")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train (default: 20).")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001).")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.1, help="learning rate decay factor (default: 0.1).")
    parser.add_argument("--lr_decay_epochs", type=int, nargs="*", default=[10, 15], help="learning rate decay epochs (default: 10 and 15).")
    parser.add_argument("--root_train", type=str, default="data/train", help="root of training data (default: data/train).")
    parser.add_argument("--root_test", type=str, default="data/test", help="root of testing data (default: data/test).")
    parser.add_argument("--model_path", type=str, default="model/model.pth", help="path to save model (default: model/model.pth).")
    parser.add_argument("--use_pretrained", action="store_true", default=False, help="use pretrained model (default: False).")
    parser.add_argument("--model_class", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"],help="model class (default: resnet18).")
    parser.add_argument("--device", type=str, default=None, help="device to use (default: None -> use CUDA if available).")
    return parser.parse_args()

def main():
    args = get_args()
    trainloader = datasets.get_dataloader(args.root_train, args.batch_size, args.num_workers, transforms=datasets.get_bare_transforms())
    num_classes = len(trainloader.dataset.classes)
    net = models.get_model(args.model_class, args.use_pretrained, num_classes=num_classes)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_gamma)

    train.train_model(net, trainloader, loss_fn, optimizer, args.epochs, scheduler, device=None)
    torch.save(net.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

    testloader = datasets.get_dataloader(args.root_test, args.batch_size, args.num_workers, transforms=datasets.get_bare_transforms())
    eval.evaluate_model(net, testloader, loss_fn=loss_fn, device=None)


