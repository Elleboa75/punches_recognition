from psd_tools import PSDImage
from PIL import Image
import os


def parse_labels(file_path):
    """
    Parameters
    ----------
    file_path: Path to the labels.txt file

    Returns
    crops: a list of coordinates and the class of the crop
    """
    crops = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cls = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    dx = float(parts[3])
                    dy = float(parts[4])
                    crops.append((cls, x, y, dx, dy))
                except ValueError:
                    print(f"Skipping invalid line: {line}")
    return crops


def split_image_psb(image_path, labels_path, output_dir):
    # Load the PSB file
    psd = PSDImage.open(image_path)
    image = psd.composite()  # Flatten into a PIL Image object
    width, height = image.size

    # Parse the labels
    crops = parse_labels(labels_path)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each crop and save it as a new image
    for idx, (cls, x, y, dx, dy) in enumerate(crops):
        # Convert normalized coordinates to pixel values
        left = int((x - dx / 2) * width)
        top = int((y - dy / 2) * height)
        right = int((x + dx / 2) * width)
        bottom = int((y + dy / 2) * height)

        # Ensure the crop coordinates are within the image bounds
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)

        # Crop the image
        crop = image.crop((left, top, right, bottom))

        # Save the crop
        output_path = os.path.join(output_dir, f"crop_{idx}_class_{cls}.png")
        crop.save(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    # Example usage
    image_path = "PSB images/07_Traino_S_Domenico_2.psb"
    labels_path = "Labels/07_Traino_S_Domenico_2.txt"
    output_dir = "Test"

    split_image_psb(image_path, labels_path, output_dir)
