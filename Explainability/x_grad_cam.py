from pytorch_grad_cam import XGradCAM
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

class XGradCAM:
    def __init__(self, model, target_layers):
        """
        Initialize the XGradCAMWrapper with the model and target layers.
        Args:
            model: PyTorch model to be used for generating Grad-CAM.
            target_layers: List of layers where the CAM will be computed.
        """
        self.model = model
        self.target_layers = target_layers
        # Initialize XGradCAM with the model and target layers
        self.cam = XGradCAM(model=self.model, target_layers=self.target_layers)

    def generate_xgrad_cam(self, input_tensor, target_class):
        """
        Generate the XGrad-CAM heatmap.
        Args:
            input_tensor (torch.Tensor): Preprocessed input tensor for the model.
            target_class (int): Class index for which to generate the heatmap.
        Returns:
            np.ndarray: Heatmap for the target class.
        """
        # Generate the heatmap using the forward() method of XGradCAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=[target_class])

        # Return the first (and only) heatmap
        return grayscale_cam[0]

    def visualize_xgrad_cam(self, input_image, heatmap, save_path=None):
        """
        Visualize the XGrad-CAM heatmap overlaid on the input image.
        Args:
            input_image (np.ndarray): Original image.
            heatmap (np.ndarray): XGrad-CAM heatmap.
            save_path (str): Path to save the visualization (optional).
        """
        # Resize the heatmap to the input image size
        heatmap_resized = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))

        # Apply color map to heatmap (Jet color map in this case)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Combine the heatmap with the original image (overlay)
        overlay = cv2.addWeighted(input_image, 0.6, heatmap_color, 0.4, 0)

        # Plot the final image
        plt.imshow(overlay)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()
