import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch_grad_cam.base_cam import BaseCAM
from punches_lib.ii_loss.models import ResNetCustom  # Adjust the import path as needed


class XGradCAM:
    def __init__(self, model_path, model_class="resnet18", device=None):
        """
        Initialization of the XGradCAM class.
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

        # XGrad-CAM specific hooks
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

    def generate_xgrad_cam(self, input_tensor, target_class):
        """
        Generate the XGrad-CAM heatmap.
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

        # Compute the XGrad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Average over the height and width

        # Ensure proper multiplication: match the number of channels
        weights = weights.expand_as(self.activations)  # Expanding weights to match activations

        # Now compute the weighted sum
        xgrad_cam = torch.sum(weights * self.activations, dim=1, keepdim=True).squeeze()

        # Apply ReLU
        xgrad_cam = F.relu(xgrad_cam)

        # Convert to numpy
        xgrad_cam = xgrad_cam.cpu().detach().numpy()

        # Normalize to [0, 1]
        xgrad_cam = (xgrad_cam - np.min(xgrad_cam)) / (np.max(xgrad_cam) - np.min(xgrad_cam) + 1e-8)
        return xgrad_cam

    def visualize_xgrad_cam(self, input_image, heatmap, save_path=None):
        """
        Visualize the XGrad-CAM heatmap overlaid on the input image.
        Args:
            input_image (np.ndarray): Original image.
            heatmap (np.ndarray): XGrad-CAM heatmap.
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

