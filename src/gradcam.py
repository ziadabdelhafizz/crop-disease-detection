
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.model import create_model


def format_class_name(class_name):
    return class_name.replace("___", " - ").replace("_", " ")


class GradCAM:
    """
    Grad-CAM explainability for EfficientNet-B0.
    It highlights image regions that contributed most to the predicted class.
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.target_layer = model.features[-1]

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(outputs, dim=1).item()

        score = outputs[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)

        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, outputs.detach()


def overlay_cam(original_image, cam, alpha=0.45):
    """
    Overlays a Grad-CAM heatmap on the original image.
    """
    cam_resized = Image.fromarray(np.uint8(cam * 255)).resize(
        (original_image.shape[1], original_image.shape[0]),
        resample=Image.BILINEAR
    )

    cam_resized = np.array(cam_resized) / 255.0
    heatmap = plt.get_cmap("jet")(cam_resized)[:, :, :3]

    overlay = (1 - alpha) * original_image + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def load_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    input_tensor = transform(image).unsqueeze(0)

    display_image = image.resize((image_size, image_size))
    display_image = np.array(display_image).astype(np.float32) / 255.0

    return input_tensor, display_image


def main(args):
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    image_size = checkpoint.get("image_size", 224)

    model = create_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    input_tensor, display_image = load_image(args.image_path, image_size)
    input_tensor = input_tensor.to(device)

    gradcam = GradCAM(model)
    cam, outputs = gradcam(input_tensor)

    probabilities = torch.softmax(outputs, dim=1)[0]
    top_prob, top_idx = torch.max(probabilities, dim=0)

    predicted_class = class_names[top_idx.item()]
    confidence = top_prob.item()

    overlay = overlay_cam(display_image, cam)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(display_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap="jet")
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(
        f"Prediction: {format_class_name(predicted_class)}\n"
        f"Confidence: {confidence * 100:.2f}%"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Grad-CAM saved to:", output_path)
    print("Predicted class:", format_class_name(predicted_class))
    print("Confidence:", f"{confidence * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="/content/crop-disease-detection/models/best_model.pth")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="/content/crop-disease-detection/outputs/gradcam_example.png")

    args = parser.parse_args()
    main(args)
