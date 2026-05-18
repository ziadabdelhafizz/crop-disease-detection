
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.model import create_model


def format_class_name(name):
    return name.replace("___", " - ").replace("_", " ")


def main(args):
    project_dir = Path(args.project_dir)
    model_path = project_dir / "models" / "best_model.pth"
    test_dir = project_dir / "data" / "processed" / "test"
    output_path = project_dir / "outputs" / "random_sample_predictions.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    image_size = checkpoint.get("image_size", 224)

    model = create_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    all_images = []
    for class_folder in sorted(test_dir.iterdir()):
        if class_folder.is_dir():
            for img_path in class_folder.glob("*"):
                all_images.append((img_path, class_folder.name))

    random.seed(42)
    selected_images = random.sample(all_images, args.num_images)

    plt.figure(figsize=(16, 12))

    for i, (img_path, true_class) in enumerate(selected_images):
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        pred_class = class_names[pred_idx.item()]
        confidence = confidence.item()

        plt.subplot(3, 4, i + 1)
        plt.imshow(image)
        plt.title(
            f"Pred: {format_class_name(pred_class)}\n"
            f"True: {format_class_name(true_class)}\n"
            f"Conf: {confidence:.2f}",
            fontsize=7
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved random sample predictions to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=str, default="/content/crop-disease-detection")
    parser.add_argument("--num_images", type=int, default=12)

    args = parser.parse_args()
    main(args)
