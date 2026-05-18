
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from tqdm import tqdm

from src.dataset import create_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from src.model import create_model


def denormalize_image(tensor):
    """
    Converts a normalized image tensor back to displayable format.
    """
    image = tensor.cpu().numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model and returns predictions, labels, probabilities, loss, and accuracy.
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_loss = running_loss / total
    test_acc = accuracy_score(all_labels, all_preds)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs), test_loss, test_acc


def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    """
    Saves a normalized confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_top_confusions(y_true, y_pred, class_names, output_path, top_k=15):
    """
    Saves the most common wrong class predictions.
    """
    cm = confusion_matrix(y_true, y_pred)

    confusions = []

    for true_idx in range(len(class_names)):
        for pred_idx in range(len(class_names)):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                confusions.append({
                    "true_class": class_names[true_idx],
                    "predicted_class": class_names[pred_idx],
                    "count": int(cm[true_idx, pred_idx])
                })

    confusions = sorted(confusions, key=lambda x: x["count"], reverse=True)

    with open(output_path / "top_confusions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["true_class", "predicted_class", "count"])
        writer.writeheader()
        writer.writerows(confusions[:top_k])


def save_predictions_csv(y_true, y_pred, probs, class_names, test_loader, output_path):
    """
    Saves test predictions with confidence scores.
    """
    samples = test_loader.dataset.samples

    rows = []

    for i, (image_path, _) in enumerate(samples):
        top3_indices = np.argsort(probs[i])[-3:][::-1]

        rows.append({
            "image_path": image_path,
            "true_class": class_names[y_true[i]],
            "predicted_class": class_names[y_pred[i]],
            "confidence": float(probs[i][y_pred[i]]),
            "correct": bool(y_true[i] == y_pred[i]),
            "top1_class": class_names[top3_indices[0]],
            "top1_confidence": float(probs[i][top3_indices[0]]),
            "top2_class": class_names[top3_indices[1]],
            "top2_confidence": float(probs[i][top3_indices[1]]),
            "top3_class": class_names[top3_indices[2]],
            "top3_confidence": float(probs[i][top3_indices[2]])
        })

    with open(output_path / "test_predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_sample_predictions(model, test_loader, class_names, device, output_path, num_images=12):
    """
    Saves a grid of sample predictions for the report and README.
    """
    model.eval()

    images_shown = 0
    plt.figure(figsize=(16, 12))

    with torch.no_grad():
        for images, labels in test_loader:
            images_device = images.to(device)
            outputs = model(images_device)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break

                image = denormalize_image(images[i])
                true_class = class_names[labels[i].item()]
                pred_class = class_names[preds[i].item()]
                confidence = probs[i][preds[i]].item()

                plt.subplot(3, 4, images_shown + 1)
                plt.imshow(image)

                title = f"Pred: {pred_class}\nTrue: {true_class}\nConf: {confidence:.2f}"
                plt.title(title, fontsize=7)
                plt.axis("off")

                images_shown += 1

            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.savefig(output_path / "sample_predictions.png", dpi=300, bbox_inches="tight")
    plt.close()


def main(args):
    project_dir = Path(args.project_dir)
    data_dir = project_dir / "data" / "processed"
    model_path = project_dir / "models" / "best_model.pth"
    output_path = project_dir / "outputs"

    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    image_size = checkpoint.get("image_size", args.image_size)

    _, _, test_loader, _ = create_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = create_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    y_true, y_pred, probs, test_loss, test_acc = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\nFinal Test Results")
    print("------------------")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report_text)

    with open(output_path / "classification_report.txt", "w") as f:
        f.write(report_text)

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1),
        "num_test_images": int(len(y_true)),
        "num_classes": int(num_classes)
    }

    with open(output_path / "test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    save_confusion_matrix(y_true, y_pred, class_names, output_path)
    save_top_confusions(y_true, y_pred, class_names, output_path)
    save_predictions_csv(y_true, y_pred, probs, class_names, test_loader, output_path)
    save_sample_predictions(model, test_loader, class_names, device, output_path)

    print("\nSaved outputs:")
    print(output_path / "test_results.json")
    print(output_path / "classification_report.txt")
    print(output_path / "confusion_matrix.png")
    print(output_path / "top_confusions.csv")
    print(output_path / "test_predictions.csv")
    print(output_path / "sample_predictions.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_dir", type=str, default="/content/crop-disease-detection")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    main(args)
