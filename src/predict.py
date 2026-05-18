
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.model import create_model


def format_class_name(class_name):
    """
    Converts dataset class names into a more readable format.
    Example:
    Tomato___Late_blight -> Tomato - Late blight
    """
    return class_name.replace("___", " - ").replace("_", " ")


def load_trained_model(model_path, device):
    """
    Loads the trained model checkpoint.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    image_size = checkpoint.get("image_size", 224)

    model = create_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, image_size


def predict_image(image_path, model, class_names, image_size, device, top_k=3):
    """
    Predicts the top-k disease classes for one image.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

    predictions = []

    for prob, idx in zip(top_probs[0], top_indices[0]):
        class_name = class_names[idx.item()]
        predictions.append({
            "class_name": class_name,
            "readable_class_name": format_class_name(class_name),
            "confidence": prob.item()
        })

    return predictions


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, class_names, image_size = load_trained_model(args.model_path, device)

    predictions = predict_image(
        image_path=args.image_path,
        model=model,
        class_names=class_names,
        image_size=image_size,
        device=device,
        top_k=args.top_k
    )

    print("\nPrediction Results")
    print("------------------")

    for rank, pred in enumerate(predictions, start=1):
        print(
            f"{rank}. {pred['readable_class_name']} "
            f"({pred['confidence'] * 100:.2f}%)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/content/crop-disease-detection/models/best_model.pth")
    parser.add_argument("--top_k", type=int, default=3)

    args = parser.parse_args()
    main(args)
