import torch
import os
import argparse
from torchvision import models, transforms
from PIL import Image
from faithfulness import FaithfulnessValidator  # Assuming this is the path to your class


def load_model(model_path, device):
    """
    Load the pre-trained model from a file.
    Args:
        model_path (str): Path to the model file.
        device (torch.device): Device to load the model to.
    Returns:
        torch.nn.Module: Loaded PyTorch model.
    """
    model = models.resnet50(pretrained = False)  # Adjust to the model type you're using
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def preprocess_image(image_path):
    """
    Preprocess an image to make it ready for the model.
    Args:
        image_path (str): Path to the image file.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def main(model_path, heatmaps_folder, images_folder, device):
    """
    Main function to validate Grad-CAM heatmaps using faithfulness metrics.
    Args:
        model_path (str): Path to the model.
        heatmaps_folder (str): Folder where the Grad-CAM heatmaps are saved.
        images_folder (str): Folder with the images corresponding to the heatmaps.
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
    """
    # Load model
    model = load_model(model_path, device)

    # Initialize FaithfulnessValidator
    validator = FaithfulnessValidator(model)

    # Iterate over the images folder and corresponding heatmaps
    for image_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_name)

        # Only process image files (adjust the condition based on your image types)
        if image_name.endswith(('.jpg', '.png', '.tif')):
            print(f"Processing image: {image_name}")

            # Preprocess the image
            input_tensor = preprocess_image(image_path).to(device)

            # Get the target class by running the image through the model
            with torch.no_grad():
                output = model(input_tensor)
                target_class = torch.argmax(output, dim = 1).item()

            # Load the corresponding heatmap (adjust for your heatmap file naming convention)
            heatmap_path = os.path.join(heatmaps_folder, f"gradcam_{image_name.split('.')[0]}.npy")

            if os.path.exists(heatmap_path):
                # Validate the heatmap
                results = validator.validate(input_tensor, target_class, heatmap_path)

                # You can print the results or save them to a file
                print(f"Validation results for {image_name}: {results}")
            else:
                print(f"Heatmap for {image_name} not found!")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description = "Validate Grad-CAM heatmaps")
    parser.add_argument("--model-path", type = str, required = True, help = "Path to the trained model file")
    parser.add_argument("--heatmaps-folder", type = str, required = True, help = "Folder with the saved heatmaps")
    parser.add_argument("--images-folder", type = str, required = True, help = "Folder with the images")
    parser.add_argument("--device", type = str, default = "cpu", choices = ["cpu", "cuda"], help = "Device to run the model on")

    args = parser.parse_args()

    # Set the device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Run the main function
    main(args.model_path, args.heatmaps_folder, args.images_folder, device)
