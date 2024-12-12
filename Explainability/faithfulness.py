import numpy as np
import quantus
import torch

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
            "Region Perturbation": quantus.RegionPerturbation(),
        }

    def load_heatmap(self, heatmap_path):
        """
        Load a heatmap from a file.
        Args:
            heatmap_path (str): Path to the heatmap file.
        Returns:
            np.ndarray: Loaded heatmap.
        """
        return np.load(heatmap_path)

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
        # Load the heatmap
        heatmap = self.load_heatmap(heatmap_path)

        # Ensure the input tensor is on the correct device
        input_tensor = input_tensor.to(next(self.model.parameters()).device)

        # Prepare results dictionary
        results = {}

        for metric_name, metric in self.metrics.items():
            score = metric(
                model=self.model,
                x_batch=input_tensor,
                y_batch=torch.tensor([target_class]),
                a_batch=torch.tensor([heatmap]),
                device=next(self.model.parameters()).device,
            )
            results[metric_name] = score
        print(results)
        return results
