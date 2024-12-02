import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress tracking
import cv2
from punches_lib.ii_loss.models import ResNetCustom  # Adjust the import path as needed
from x_grad_cam import XGradCAM  # Import XGrad-CAM class


class GradCAM:
    def __init__(self, model_path, model_class="resnet18", device=None):
        """
        Initialization of the GradCAM class.
        Args:
            model_path (str): Path to the saved model.
            model_class (str): Model architecture class (e.g., 'resnet18').
            device (str): Device to load the model on (default: CUDA if available).
        """
        self.model_class = model_class
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNetCustom(num_classes=19, architecture=self.model_class)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval().to(self.device)

        # Grad-CAM specific hooks
        self.gradients = None
        self.activations = None

    def _save_gradient(self, grad):
        self.gradients = grad

    def forward_pass(self, input_tensor):
        """
        Forward pass through the model to capture gradients and activations.
        Args:
            input_tensor (torch.Tensor): Input tensor to the model.
        Returns:
            torch.Tensor: Model predictions.
        """
        # Hook into the last convolutional layer (layer4[1].conv2 in ResNet)
        last_conv_layer = self.model.layer4[1].conv2  # Update this if your model structure is different
        last_conv_layer.register_forward_hook(self._activation_hook)
        last_conv_layer.register_full_backward_hook(self._gradient_hook)

        return self.model(input_tensor)

    def _activation_hook(self, module, input, output):
        # Capture activations
        self.activations = output

    def _gradient_hook(self, module, grad_input, grad_output):
        # Capture gradients
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class):
        """
        Generate the Grad-CAM heatmap.
        Args:
            input_tensor (torch.Tensor): Input tensor for the model.
            target_class (int): Class index for which to generate the heatmap.
        Returns:
            np.ndarray: Heatmap for the target class.
        """
        output = self.forward_pass(input_tensor)

        # Assuming output[0] contains the actual logits
        logits = output[0]

        # Zero gradients for the model
        self.model.zero_grad()
        target = logits[:, target_class]
        target.backward(retain_graph=True)

        # Compute the Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Average over the height and width

        # Ensure proper multiplication: match the number of channels
        weights = weights.expand_as(self.activations)  # Expanding weights to match activations

        # Now compute the weighted sum
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True).squeeze()

        # Apply ReLU
        cam = F.relu(cam)

        # Convert to numpy
        cam = cam.cpu().detach().numpy()

        # Normalize to [0, 1]
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam

    def visualize_cam(self, input_image, heatmap, save_path=None):
        """
        Visualize the Grad-CAM heatmap overlaid on the input image.
        Args:
            input_image (np.ndarray): Original image.
            heatmap (np.ndarray): Grad-CAM heatmap.
            save_path (str): Path to save the visualization (optional).
        """
        heatmap_resized = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(input_image, 0.6, heatmap_color, 0.4, 0)

        plt.imshow(overlay)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()


def process_test_folder(test_folder, output_base_folder, gradcam, xgradcam, transform, device):
    """
    Process all images in the test folder and generate Grad-CAM and XGrad-CAM visualizations.
    Args:
        test_folder (str): Path to the test folder containing subfolders of images.
        output_base_folder (str): Base path to save Explainability/grad_cam outputs.
        gradcam (GradCAM): Grad-CAM instance.
        xgradcam (XGradCAM): XGrad-CAM instance.
        transform (transforms.Compose): Transformations for the input images.
        device (str): Device to use for processing.
    """
    gradcam_output_folder = os.path.join(output_base_folder, "Explainability", "grad_cam")
    xgradcam_output_folder = os.path.join(output_base_folder, "Explainability", "xgrad_cam")

    os.makedirs(gradcam_output_folder, exist_ok=True)
    os.makedirs(xgradcam_output_folder, exist_ok=True)

    for subdir, _, files in os.walk(test_folder):
        for file in tqdm(files, desc=f"Processing {subdir}"):

            image_path = os.path.join(subdir, file)
            if not (file.endswith(".jpg") or file.endswith(".png") or file.endswith(".tif")):
                continue

            input_image = Image.open(image_path).convert("RGB")
            input_tensor = transform(input_image).unsqueeze(0).to(device)

            # Pass the image through the model
            output = gradcam.model(input_tensor)

            # Get logits and find the target class
            logits = output[0]
            target_class = torch.argmax(logits, dim=1).item()

            # Grad-CAM
            heatmap = gradcam.generate_cam(input_tensor, target_class)
            relative_subdir = os.path.relpath(subdir, test_folder)  # Keeps relative subfolder structure
            output_subfolder = os.path.join(gradcam_output_folder, relative_subdir)
            os.makedirs(output_subfolder, exist_ok=True)
            gradcam_path = os.path.join(output_subfolder, f"gradcam_{file}")
            gradcam.visualize_cam(np.array(input_image), heatmap, save_path=gradcam_path)

            print(f"Saved Grad-CAM visualization for {image_path} to {gradcam_path}")  # Debug print

            # XGrad-CAM
            heatmap_xgrad = xgradcam.generate_xgrad_cam(input_tensor, target_class)
            output_subfolder_xgrad = os.path.join(xgradcam_output_folder, relative_subdir)
            os.makedirs(output_subfolder_xgrad, exist_ok=True)
            xgradcam_path = os.path.join(output_subfolder_xgrad, f"xgradcam_{file}")
            xgradcam.visualize_xgrad_cam(np.array(input_image), heatmap_xgrad, save_path=xgradcam_path)

            print(f"Saved XGrad-CAM visualization for {image_path} to {xgradcam_path}")  # Debug print



if __name__ == "__main__":
    model_path = "../model/model_ii.pth"
    model_class = "resnet18"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gradcam = GradCAM(model_path=model_path, model_class=model_class, device=device)
    xgradcam = XGradCAM(model_path=model_path, model_class=model_class, device=device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_folder = "../data/Test/"
    output_base_folder = "."

    process_test_folder(test_folder, output_base_folder, gradcam, xgradcam, transform, device)
