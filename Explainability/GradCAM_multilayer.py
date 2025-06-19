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

def visualize_and_save_cam(cam, input_image, base_dir, class_name, method, layer_names, image_name):
    """
    Overlays the CAM heatmap onto the input image and saves the result.
    The layer_names is a list of layers used, shown as a combined string.
    """
    # Undo normalization for visualization.
    image_np = input_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image_np = np.clip(image_np * np.array([0.229, 0.224, 0.225]) +
                       np.array([0.485, 0.456, 0.406]), 0, 1)
    image_np = (255 * image_np).astype(np.uint8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, 0.6, image_np, 0.4, 0)

    # Combine layer names for folder naming.
    layers_str = "_".join(layer_names)
    out_dir = os.path.join(base_dir, class_name, method, layers_str)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{image_name}_{method}.png"
    cv2.imwrite(os.path.join(out_dir, filename), superimposed_img)

def main(model_path, test_dataset, save_dir, transform):
    # Mapping for GradCAM target purposes.
    mapping = {0: 0, 1: 5, 2: 10, 3: 15}

    # Initialize the model and load weights.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 27)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval().cuda()

    # Gather all convolutional layers.
    conv_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    # Select only layers from layer3 and layer4.
    target_layers_info = [(name, module) for name, module in conv_layers if "layer3" in name or "layer4" in name]
    target_layers = [module for name, module in target_layers_info]
    target_layer_names = [name for name, module in target_layers_info]

    # Process each image individually.
    for image_path, target in test_dataset.samples:
        # Load the image and apply the transform.
        image = Image.open(image_path).convert('RGB')
        input_image = transform(image)
        input_image = input_image.unsqueeze(0).cuda()

        test_label = target
        # Get the class name from the dataset.
        class_name = test_dataset.classes[test_label]
        # Use the mapping for GradCAM target.
        mapped_class = mapping[test_label]
        target_category = [ClassifierOutputTarget(mapped_class)]

        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Generate GradCAM using the selected layers.
        with GradCAM(model=model, target_layers=target_layers) as gradcam:
            cam_grad = gradcam(input_tensor=input_image, targets=target_category)[0]

        # Generate XGradCAM using the selected layers.
        with XGradCAM(model=model, target_layers=target_layers) as xgradcam:
            cam_xgrad = xgradcam(input_tensor=input_image, targets=target_category)[0]

        # Save CAM overlays.
        visualize_and_save_cam(cam_grad, input_image, save_dir, class_name, "gradcam", target_layer_names, image_name)
        visualize_and_save_cam(cam_xgrad, input_image, save_dir, class_name, "xgradcam", target_layer_names, image_name)

if __name__ == "__main__":
    model_path = "../cnn_model/model_perturbed_ImageNet.pth"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(root='../new_dataset/Test', transform=transform)
    save_dir = "Saliency_maps_multi_layer"

    main(model_path, test_dataset, save_dir, transform)
