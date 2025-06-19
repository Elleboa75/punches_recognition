import argparse
from psd_tools import PSDImage
from PIL import Image
import os


def parse_labels(file_path):
    """
    Parameters
    ----------
    file_path: Path to the labels.txt file

    Returns
    -------
    crops: a list of tuples (cls, x, y, dx, dy)
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
                    print(f"Skipping invalid line: {line.strip()}")
    return crops


def split_image_psb(image_path, labels_path, output_dir):
    """
    Splits a PSB image into crops based on normalized coordinates in labels_path.

    Parameters
    ----------
    image_path: Path to the PSB image file
    labels_path: Path to the labels.txt file with lines: cls x y dx dy (normalized)
    output_dir: Directory where crops will be saved
    """
    # Load the PSB file
    psd = PSDImage.open(image_path)
    image = psd.composite()
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

        if left >= right or top >= bottom:
            print(f"Skipping invalid crop coordinates for index {idx}: ({left}, {top}, {right}, {bottom})")
            continue

        # Crop the image
        crop = image.crop((left, top, right, bottom))

        # Save the crop
        output_path = os.path.join(output_dir, f"crop_{idx}_class_{cls}.png")
        crop.save(output_path)
        print(f"Saved: {output_path}")


def get_args():
    parser = argparse.ArgumentParser(description="Split PSB image into crops based on normalized labels.")
    parser.add_argument(
        '--image_path', type=str, required=True,
        help='Path to the PSB image file.'
    )
    parser.add_argument(
        '--labels_path', type=str, required=True,
        help='Path to the labels.txt file with normalized coords.'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save cropped images.'
    )
    return parser.parse_args()


def main():
    args = get_args()
    image_path = args.image_path
    labels_path = args.labels_path
    output_dir = args.output_dir

    if not os.path.isfile(image_path):
        print(f"Error: image file not found: {image_path}")
        return
    if not os.path.isfile(labels_path):
        print(f"Error: labels file not found: {labels_path}")
        return

    split_image_psb(image_path, labels_path, output_dir)


if __name__ == '__main__':
    main()
