import os
import torch
import torch.nn.functional as F
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from punches_lib.ii_loss.models import ResNetCustom
from pytorch_grad_cam import XGradCAM, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMManual:
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
        self.model = ResNetCustom(num_classes=19, architecture=self.model_class)  # Matches saved model output classes
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {}

        # Adjust the state_dict to fit the current model structure
        for key, value in state_dict.items():
            if key.startswith("fc2."):
                # Skip fc2 keys since they are not present in the saved model
                continue
            elif key.startswith("fc1.weight") and value.size(0) == 19:
                # Match the size of fc1 to the loaded model's expected size
                new_state_dict[key] = value
            elif key.startswith("fc1.bias") and value.size(0) == 19:
                new_state_dict[key] = value
            else:
                # Retain other keys as-is
                new_state_dict[key] = value

        # Load adjusted state_dict
        self.model.load_state_dict(new_state_dict, strict=False)
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

    def generate_cam(self, input_tensor, target_class, input_image):
        """
        Generate the Grad-CAM heatmap.
        Args:
            input_tensor (torch.Tensor): Input tensor for the model.
            target_class (int): Class index for which to generate the heatmap.
        Returns:
            np.ndarray: Heatmap for the target class.
        """
        pass
def process_test_folder(test_folder, output_base_folder, gradcam, transform, device):
    """
    Process all images in the test folder and generate Grad-CAM visualizations.
    Args:
        test_folder (str): Path to the test folder containing subfolders of images.
        output_base_folder (str): Base path to save Explainability/grad_cam outputs.
        gradcam (GradCAM): Grad-CAM instance.
        transform (transforms.Compose): Transformations for the input images.
        device (str): Device to use for processing.
    """
    gradcam_output_folder = os.path.join(output_base_folder, "Explainability", "grad_cam")

    os.makedirs(gradcam_output_folder, exist_ok=True)

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
            gradcam = GradCAM(model = model, target_layers = [model.layer4[1]])  # Use appropriate layer

            # Pass the image through the model
            output = model(input_tensor)

            # Get logits and find the target class
            logits = output[0]
            target_class = torch.argmax(logits, dim = 1).item()

            # Wrap the target class index in a ClassifierOutputTarget
            targets = [ClassifierOutputTarget(target_class)]  # Correct way to specify the target

            # Grad-CAM
            grayscale_cam = gradcam(input_tensor = input_tensor, targets = targets)

            # Since grayscale_cam is a batch, we get the first item
            grayscale_cam = grayscale_cam[0, :]
            # Normalize the image to the range [0, 1] and convert to np.float32
            input_image = np.array(input_image) / 255.0  # Normalize to [0, 1]
            input_image = input_image.astype(np.float32)  # Ensure it's of type np.float32

            # Resize the heatmap to match the input image size
            grayscale_cam_resized = cv2.resize(grayscale_cam, (input_image.shape[1], input_image.shape[0]))

            # Visualize and save XGrad-CAM output
            relative_subdir = os.path.relpath(subdir, test_folder)
            output_subfolder_grad = os.path.join(gradcam_output_folder, relative_subdir)
            os.makedirs(output_subfolder_grad, exist_ok = True)
            gradcam_path = os.path.join(output_subfolder_grad, f"gradcam_{file}")

            # Use matplotlib to save the image
            plt.imshow(show_cam_on_image(input_image, grayscale_cam_resized, use_rgb = True))
            plt.axis('off')
            plt.savefig(gradcam_path, bbox_inches = 'tight', pad_inches = 0)
            plt.close()

            print(f"Saved Grad-CAM visualization for {image_path} to {gradcam_path}")  # Debug print


def process_test_folder_xgrad(test_folder, output_base_folder, model, transform, device):
    """
    Process all images in the test folder and generate XGrad-CAM visualizations.
    Args:
        test_folder (str): Path to the test folder containing subfolders of images.
        output_base_folder (str): Base path to save Explainability/xgrad_cam outputs.
        model (nn.Module): The trained model.
        transform (transforms.Compose): Transformations for the input images.
        device (str): Device to use for processing.
    """
    xgradcam_output_folder = os.path.join(output_base_folder, "Explainability", "xgrad_cam")

    os.makedirs(xgradcam_output_folder, exist_ok=True)

    for subdir, _, files in os.walk(test_folder):
        for file in tqdm(files, desc=f"Processing {subdir}"):

            image_path = os.path.join(subdir, file)
            if not (file.endswith(".jpg") or file.endswith(".png") or file.endswith(".tif")):
                continue

            input_image = Image.open(image_path).convert("RGB")
            input_tensor = transform(input_image).unsqueeze(0).to(device)

            # Initialize XGrad-CAM
            xgradcam = XGradCAM(model=model, target_layers=[model.layer4[1]])  # Use appropriate layer

            # Pass the image through the model
            output = model(input_tensor)

            # Get logits and find the target class
            logits = output[0]
            target_class = torch.argmax(logits, dim=1).item()

            # Wrap the target class index in a ClassifierOutputTarget
            targets = [ClassifierOutputTarget(target_class)]  # Correct way to specify the target

            # XGrad-CAM
            grayscale_cam = xgradcam(input_tensor=input_tensor, targets=targets)

            # Since grayscale_cam is a batch, we get the first item
            grayscale_cam = grayscale_cam[0, :]

            # Normalize the image to the range [0, 1] and convert to np.float32
            input_image = np.array(input_image) / 255.0
            input_image = input_image.astype(np.float32)

            # Resize the heatmap to match the input image size
            grayscale_cam_resized = cv2.resize(grayscale_cam, (input_image.shape[1], input_image.shape[0]))

            # Visualize and save XGrad-CAM output
            relative_subdir = os.path.relpath(subdir, test_folder)
            output_subfolder_xgrad = os.path.join(xgradcam_output_folder, relative_subdir)
            os.makedirs(output_subfolder_xgrad, exist_ok=True)
            xgradcam_path = os.path.join(output_subfolder_xgrad, f"xgradcam_{file}")

            # Use matplotlib to save the image
            plt.imshow(show_cam_on_image(input_image, grayscale_cam_resized, use_rgb=True))
            plt.axis('off')
            plt.savefig(xgradcam_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"Saved XGrad-CAM visualization for {image_path} to {xgradcam_path}")  # Debug print

if __name__ == "__main__":
    model_path = "../cnn_model/model.pth"
    model_class = "resnet18"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gradcam = GradCAMManual(model_path=model_path, model_class=model_class, device=device)
    # Load your model for XGrad-CAM

    state_dict = torch.load(model_path, map_location = device)
    new_state_dict = {}

    # Adjust the state_dict to fit the current model structure
    for key, value in state_dict.items():
        if key.startswith("fc2."):
            # Skip fc2 keys since they are not present in the saved model
            continue
        elif key.startswith("fc1.weight") and value.size(0) == 19:
            # Match the size of fc1 to the loaded model's expected size
            new_state_dict[key] = value
        elif key.startswith("fc1.bias") and value.size(0) == 19:
            new_state_dict[key] = value
        else:
            # Retain other keys as-is
            new_state_dict[key] = value

    model = ResNetCustom(num_classes = 19, architecture = model_class)
    model.load_state_dict(new_state_dict, strict = False)
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_folder = "../data/Test/"
    output_base_folder = "."

    # Process images for Grad-CAM
    process_test_folder(test_folder, output_base_folder, gradcam, transform, device)
    process_test_folder_xgrad(test_folder, output_base_folder, model, transform, device)
