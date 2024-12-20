from PIL import Image
import numpy as np
import torch
import quantus

class FaithfulnessValidator:
    def __init__(self, model):
        """
        Initialize the FaithfulnessValidator class.
        Args:
            model (torch.nn.Module): The PyTorch model to validate explanations against.
        """
        self.model = model
        self.metrics = {
            "Pixel-Flipping": quantus.PixelFlipping(),
            "Faithfulness Correlation": quantus.FaithfulnessCorrelation(),
            "Faithfulness Estimate": quantus.FaithfulnessEstimate(),
            "Region Perturbation": quantus.RegionPerturbation(), #TODO: try ROAD
        }

    def load_heatmap(self, heatmap_path, target_size=(224, 224)):
        """
        Load a heatmap from a file (now supports .tif files) and resize it.
        Args:
            heatmap_path (str): Path to the heatmap file.
            target_size (tuple): Target size to resize the heatmap.
        Returns:
            np.ndarray: Loaded and resized heatmap as a NumPy array.
        """
        # Open the heatmap (which is a .tif image file) and convert it to grayscale
        heatmap_image = Image.open(heatmap_path)
        # Convert to grayscale (if it's not already) and resize to target size
        heatmap = heatmap_image.convert('L').resize(target_size, Image.Resampling.LANCZOS)
        return np.array(heatmap)

    def validate(self, input_tensor, target_class, heatmap_path):
        """
        Validate the faithfulness of a heatmap using Quantus metrics.
        Args:
            input_tensor (torch.Tensor): The preprocessed input tensor.
            target_class (int): The target class for the explanation.
            heatmap_path (str): Path to the saved heatmap.
        Returns:
            dict: A dictionary of metric names and their corresponding scores.
        """
        self.model.eval()
        # Load the heatmap and resize it to the same size as the input image
        heatmap = self.load_heatmap(heatmap_path, target_size=(input_tensor.shape[2], input_tensor.shape[3]))
        print("Heatmap loaded and resized successfully")

        # Ensure the input tensor is on the correct device
        input_tensor = input_tensor.to(next(self.model.parameters()).device)

        # Prepare results dictionary
        results = {}

        for metric_name, metric in self.metrics.items():
            score = metric(
                model=self.model,
                x_batch=input_tensor,
                y_batch=torch.tensor([target_class]),
                a_batch=torch.tensor([heatmap]),  # Pass heatmap as a tensor
                device=next(self.model.parameters()).device,
            )
            results[metric_name] = score
        print(results)
        return results
