import argparse
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models, transforms, datasets
from PIL import Image
from pytorch_grad_cam import GradCAM, XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_args():
    parser = argparse.ArgumentParser(description="Generate GradCAM and XGradCAM saliency maps for a dataset.")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model weights file.')
    parser.add_argument('--test_root', type=str, required=True,
                        help='Root directory for test images (ImageFolder structure).')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save the saliency map outputs.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use, e.g., "cuda" or "cpu". Defaults to CUDA if available.')
    parser.add_argument('--resize', type=int, default=256,
                        help='Resize shorter side to this size (default: 256).')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Center crop size (default: 224).')
    return parser.parse_args()


def visualize_and_save_cam(cam, input_image, base_dir, class_name, method, layer_names, image_name):
    """
    Overlays the CAM heatmap onto the input image and saves the result.
    layer_names: list of layer names used; joined for folder naming.
    """
    # Undo normalization for visualization.
    image_np = input_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image_np = np.clip(image_np * np.array([0.229, 0.224, 0.225]) +
                       np.array([0.485, 0.456, 0.406]), 0, 1)
    image_np = (255 * image_np).astype(np.uint8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, 0.6, image_np, 0.4, 0)

    layers_str = "_".join(layer_names)
    out_dir = os.path.join(base_dir, class_name, method, layers_str)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{image_name}_{method}.png"
    cv2.imwrite(os.path.join(out_dir, filename), superimposed_img)


def main():
    args = get_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(root=args.test_root, transform=transform)

    # Initialize the model and load weights.
    model = models.resnet18(weights=None)
    num_classes = len(test_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    # Default mapping: identity mapping; user can modify if needed.
    mapping = {i: i for i in range(num_classes)}

    # Gather convolutional layers
    conv_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    target_layers_info = [(name, module) for name, module in conv_layers if "layer3" in name or "layer4" in name]
    if not target_layers_info:
        raise ValueError("No target layers found for GradCAM (looking for 'layer3' or 'layer4').")
    target_layers = [module for _, module in target_layers_info]
    target_layer_names = [name for name, _ in target_layers_info]

    # Process each image
    for image_path, target in test_dataset.samples:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)

        class_name = test_dataset.classes[target]
        mapped_class = mapping.get(target, target)
        target_category = [ClassifierOutputTarget(mapped_class)]
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # GradCAM
        with GradCAM(model=model, target_layers=target_layers) as gradcam:
            cam_grad = gradcam(input_tensor=input_image, targets=target_category)[0]
        # XGradCAM
        with XGradCAM(model=model, target_layers=target_layers) as xgradcam:
            cam_xgrad = xgradcam(input_tensor=input_image, targets=target_category)[0]

        # Save outputs
        visualize_and_save_cam(cam_grad, input_image, args.save_dir, class_name, "gradcam", target_layer_names, image_name)
        visualize_and_save_cam(cam_xgrad, input_image, args.save_dir, class_name, "xgradcam", target_layer_names, image_name)

    print(f"Saliency maps saved under {args.save_dir}")

if __name__ == "__main__":
    main()
