import torch
import numpy as np
import argparse
from PIL import Image
import quantus
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import csv
from tqdm import tqdm

DEBUG = False  # Set to True to enable debug prints

def get_args():
    parser = argparse.ArgumentParser(
        description="Validate saliency maps with ROAD and IROF metrics using Quantus."
    )
    parser.add_argument(
        "--root_test", type=str, required=True,
        help="Root directory of testing data (ImageFolder structure)."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the saved model weights file."
    )
    parser.add_argument(
        "--cam_output_folder", type=str, required=True,
        help="Root folder where saliency maps (heatmaps) are stored, organized by class/method/layer."
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Path to save the resulting CSV of aggregated scores."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for test loader (default: 1)."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help='Device to use, e.g. "cuda" or "cpu". Defaults to CUDA if available.'
    )
    parser.add_argument(
        "--resize", type=int, default=256,
        help="Resize shorter side to this size for input transform (default: 256)."
    )
    parser.add_argument(
        "--crop_size", type=int, default=224,
        help="Center crop size for input transform (default: 224)."
    )
    return parser.parse_args()

def extract_score(metric_output):
    """Extracts a single numeric score from metric output."""
    if isinstance(metric_output, dict):
        if "score" in metric_output:
            return metric_output["score"]
        else:
            # average over values
            return np.mean(list(metric_output.values()))
    return metric_output

class FaithfulnessValidator:
    def __init__(self, model, debug=False):
        self.model = model
        self.metrics = {
            "ROAD": quantus.ROAD(
                noise=0.2,
                percentages=list(range(1, 50, 2)),
                display_progressbar=True,
            ),
            "IROF Score": quantus.IROF(
                segmentation_method="slic",
                perturb_baseline="mean",
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                return_aggregate=False,
                display_progressbar=True,
            ),
        }
        self.debug = debug
        if self.debug:
            dev = next(self.model.parameters()).device
            print("[Validator] Initialized with model on device:", dev)

    def load_heatmap(self, heatmap_path, target_size=(224, 224)):
        """Loads and processes a heatmap (grayscale PNG) and resizes to target."""
        if self.debug:
            print(f"[load_heatmap] Loading heatmap from {heatmap_path}")
        heatmap_image = Image.open(heatmap_path).convert('L').resize(
            target_size, Image.Resampling.LANCZOS
        )
        return np.array(heatmap_image, dtype=np.float32) / 255.0

    def validate(self, input_tensor, target_class, heatmap_path):
        """Runs ROAD and IROF metrics for a given heatmap."""
        if self.debug:
            print(f"[validate] Running validation for target class {target_class}; heatmap: {heatmap_path}")

        self.model.eval()
        # Load heatmap and resize to match input spatial dims
        _, _, H, W = input_tensor.shape
        heatmap = self.load_heatmap(heatmap_path, target_size=(W, H))

        # Move model to CPU for Quantus evaluation
        original_device = next(self.model.parameters()).device
        self.model.to(torch.device("cpu"))

        # Convert inputs to numpy arrays for Quantus
        x_cpu = input_tensor.cpu().numpy()
        y_cpu = np.array([target_class])
        a_cpu = np.expand_dims(heatmap, axis=0)  # shape (1, H, W)

        results = {}
        for metric_name, metric in self.metrics.items():
            if self.debug:
                print(f"[validate] Evaluating {metric_name}...")
            score = metric(
                model=self.model,
                x_batch=x_cpu,
                y_batch=y_cpu,
                a_batch=a_cpu,
                device=torch.device("cpu")
            )
            results[metric_name] = score
            if self.debug:
                print(f"[validate] {metric_name} output: {score}; extracted score: {extract_score(score)}")

        # Move model back to original device
        self.model.to(original_device)
        return results

def main():
    args = get_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if DEBUG:
        print(f"[Main] Using device: {device}")

    # Input transform
    transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(root=args.root_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    if DEBUG:
        print("[Main] Loading model...")
    model = models.resnet18(weights=None)
    num_classes = len(test_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    if DEBUG:
        print("[Main] Model loaded successfully!")

    validator = FaithfulnessValidator(model, debug=DEBUG)

    # Build a dict: explanation_files[class_name][method][layer] = list_of_heatmap_paths
    explanation_files = {}
    for class_name in test_dataset.classes:
        class_expl_path = os.path.join(args.cam_output_folder, class_name)
        explanation_files[class_name] = {}
        for method in ['gradcam', 'xgradcam']:
            method_path = os.path.join(class_expl_path, method)
            if os.path.exists(method_path) and os.path.isdir(method_path):
                explanation_files[class_name][method] = {}
                layers = sorted(os.listdir(method_path))
                for layer in layers:
                    layer_path = os.path.join(method_path, layer)
                    if os.path.isdir(layer_path):
                        files = sorted(os.listdir(layer_path))
                        heatmaps = [
                            os.path.join(layer_path, f)
                            for f in files if f.lower().endswith('.png')
                        ]
                        if heatmaps:
                            explanation_files[class_name][method][layer] = heatmaps

    # Track indices for each class/method/layer
    explanation_indices = {
        c: {m: {l: 0 for l in explanation_files[c].get(m, {})}
            for m in explanation_files[c]}
        for c in explanation_files
    }

    # Aggregated results
    results_agg = {}  # key: (class_name, method, layer) -> dict with lists
    # Per-class overall metrics
    class_metrics = {cls: {"accuracy": [], "confidence": [], "road": [], "irof": []}
                     for cls in test_dataset.classes}
    # Confusion counts
    confusion = {cls: {"TP": 0, "FP": 0, "FN": 0} for cls in test_dataset.classes}

    mapping = {0: 0, 1: 5, 2: 10, 3: 15}
    rev_mapping = {train_idx: test_idx for test_idx, train_idx in mapping.items()}

    total_iterations = sum(
        len(layers) for cls in explanation_files.values() for meth in cls.values() for layers in meth.values()
    )
    overall_progress = tqdm(total=total_iterations, desc="Overall heatmap evaluations", leave=True)

    print("[Main] Starting validation process...")
    for inputs, labels in tqdm(test_loader, desc="Processing test images", leave=True):
        inputs = inputs.to(device)
        test_label = labels.item()
        class_name = test_dataset.classes[test_label]
        target_training_class = mapping.get(test_label, test_label)

        # Model prediction
        with torch.no_grad():
            logits = model(inputs)
            probs = nn.functional.softmax(logits, dim=1)
            pred_prob, pred = torch.max(probs, dim=1)
            pred = pred.item()
            pred_prob = pred_prob.item()
        prediction_correct = int(pred == target_training_class)
        confidence = pred_prob

        # Determine predicted test-class if mapping applies
        if pred in rev_mapping:
            pred_test_idx = rev_mapping[pred]
            class_name_pred = test_dataset.classes[pred_test_idx]
        else:
            class_name_pred = None

        # Update confusion
        if class_name_pred == class_name:
            confusion[class_name]["TP"] += 1
        else:
            confusion[class_name]["FN"] += 1
            if class_name_pred is not None:
                confusion[class_name_pred]["FP"] += 1

        # If no explanations for this class, skip but advance progress
        if class_name not in explanation_files:
            # advance by count of all expected heatmaps for this class
            skip_count = sum(len(explanation_files.get(class_name, {}).get(m, {}))
                             for m in explanation_files.get(class_name, {}))
            overall_progress.update(skip_count)
            continue

        # Iterate methods and layers
        for method, layers_dict in explanation_files[class_name].items():
            for layer, file_list in layers_dict.items():
                idx = explanation_indices[class_name][method][layer]
                if idx >= len(file_list):
                    overall_progress.update(1)
                    continue
                heatmap_path = file_list[idx]
                explanation_indices[class_name][method][layer] += 1

                if DEBUG:
                    print(f"[Processing] Validating heatmap: {heatmap_path}")
                scores = validator.validate(inputs, target_training_class, heatmap_path)

                road_score = extract_score(scores.get("ROAD", None))
                irof_score = extract_score(scores.get("IROF Score", None))

                key = (class_name, method, layer)
                if key not in results_agg:
                    results_agg[key] = {"road_scores": [], "irof_scores": [], "correct": []}
                results_agg[key]["road_scores"].append(road_score)
                results_agg[key]["irof_scores"].append(irof_score)
                results_agg[key]["correct"].append(prediction_correct)

                # Per-class aggregates
                class_metrics[class_name]["accuracy"].append(prediction_correct)
                class_metrics[class_name]["confidence"].append(confidence)
                class_metrics[class_name]["road"].append(road_score)
                class_metrics[class_name]["irof"].append(irof_score)

                overall_progress.update(1)

    overall_progress.close()

    print(f"[Main] Validation complete. Saving results to {args.save_path}...")

    # Helper to compute averages
    def compute_avgs(metric_dict):
        """Return (avg_acc, avg_road, avg_irof) from lists in metric_dict."""
        correct_arr = np.array(metric_dict["correct"])
        road_arr = np.array(metric_dict["road_scores"])
        irof_arr = np.array(metric_dict["irof_scores"])

        avg_acc = correct_arr.mean() if correct_arr.size > 0 else None
        avg_road = road_arr.mean() if road_arr.size > 0 else None
        # For IROF, average only where correct==1
        if irof_arr.size > 0 and correct_arr.sum() > 0:
            avg_irof = irof_arr[correct_arr == 1].mean()
        else:
            avg_irof = None
        return avg_acc, avg_road, avg_irof

    # Print tables to console
    gradcam_keys = [k for k in results_agg if k[1] == "gradcam"]
    xgradcam_keys = [k for k in results_agg if k[1] == "xgradcam"]

    print("\n--- Aggregated Results: GradCAM ---")
    header_gc = f"{'Class':<15}{'Layer':<20}{'Avg Acc':<10}{'Avg ROAD':<10}{'Avg IROF':<10}"
    print(header_gc)
    print("-" * len(header_gc))
    for class_name, method, layer in sorted(gradcam_keys):
        metrics = results_agg[(class_name, method, layer)]
        avg_acc, avg_road, avg_irof = compute_avgs(metrics)
        print(f"{class_name:<15}{layer:<20}"
              f"{(avg_acc or 0):<10.4f}{(avg_road or 0):<10.4f}{(avg_irof or 0):<10.4f}")

    print("\n--- Aggregated Results: XGradCAM ---")
    header_xgc = f"{'Class':<15}{'Layer':<20}{'Avg Acc':<10}{'Avg ROAD':<10}{'Avg IROF':<10}"
    print(header_xgc)
    print("-" * len(header_xgc))
    for class_name, method, layer in sorted(xgradcam_keys):
        metrics = results_agg[(class_name, method, layer)]
        avg_acc, avg_road, avg_irof = compute_avgs(metrics)
        print(f"{class_name:<15}{layer:<20}"
              f"{(avg_acc or 0):<10.4f}{(avg_road or 0):<10.4f}{(avg_irof or 0):<10.4f}")

    # Per-class overall metrics: accuracy, confidence, ROAD, IROF, precision, recall
    print("\n--- Average Metrics per Class (Overall) ---")
    header_cls = (
        f"{'Class':<20}"
        f"{'Avg Acc':<10}"
        f"{'Avg Conf':<10}"
        f"{'Avg ROAD':<10}"
        f"{'Avg IROF':<10}"
        f"{'Avg Prec':<10}"
        f"{'Avg Rec':<10}"
    )
    print(header_cls)
    print("-" * len(header_cls))
    for cls, m in class_metrics.items():
        if m["accuracy"]:
            avg_acc = np.mean(m["accuracy"])
            avg_conf = np.mean(m["confidence"])
            avg_road = np.mean(m["road"])
            irof_arr = np.array(m["irof"])
            correct_arr = np.array(m["accuracy"])
            if irof_arr.size > 0 and correct_arr.sum() > 0:
                avg_irof = irof_arr[correct_arr == 1].mean()
            else:
                avg_irof = 0.0
        else:
            avg_acc = avg_conf = avg_road = avg_irof = 0.0

        TP = confusion[cls]["TP"]
        FP = confusion[cls]["FP"]
        FN = confusion[cls]["FN"]
        avg_prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        avg_rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        print(
            f"{cls:<20}"
            f"{avg_acc:<10.4f}"
            f"{avg_conf:<10.4f}"
            f"{avg_road:<10.4f}"
            f"{avg_irof:<10.4f}"
            f"{avg_prec:<10.4f}"
            f"{avg_rec:<10.4f}"
        )

    # Write results to CSV
    with open(args.save_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Class", "Method", "Layer", "Avg Accuracy", "Avg ROAD Score", "Avg IROF Score"])
        for (class_name, method, layer), metrics in results_agg.items():
            avg_acc, avg_road, avg_irof = compute_avgs(metrics)
            writer.writerow([class_name, method, layer, avg_acc, avg_road, avg_irof])

    print(f"[Main] Results saved to {args.save_path}!")

if __name__ == "__main__":
    main()
