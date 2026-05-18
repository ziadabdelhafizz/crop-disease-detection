
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.gradcam import GradCAM, overlay_cam
from src.model import create_model


MODEL_PATH = PROJECT_DIR / "models" / "best_model.pth"


def format_class_name(class_name):
    return class_name.replace("___", " - ").replace("_", " ")


def get_confidence_message(confidence):
    if confidence >= 0.80:
        return (
            "High-confidence prediction. The model is confident, but the result should still "
            "be treated as decision-support rather than a final agricultural diagnosis."
        )
    elif confidence >= 0.60:
        return (
            "Medium-confidence prediction. The image may share symptoms with multiple diseases. "
            "Check the top-3 predictions carefully."
        )
    else:
        return (
            "Low-confidence prediction. The image may be unclear, outside the training distribution, "
            "or visually different from the dataset. Human expert review is recommended."
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]
image_size = checkpoint.get("image_size", 224)

model = create_model(num_classes=num_classes, freeze_backbone=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

gradcam_engine = GradCAM(model)

transform = transforms.Compose([
    transforms.Resize((image_size + 32, image_size + 32)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def predict(image):
    if image is None:
        return {}, "Please upload an image.", "", None

    pil_image = Image.fromarray(image).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, k=3)

    top3_results = {}

    for prob, idx in zip(top_probs, top_indices):
        readable_class = format_class_name(class_names[idx.item()])
        top3_results[readable_class] = float(prob.item())

    top1_confidence = float(top_probs[0].item())
    top1_index = int(top_indices[0].item())
    top1_class = format_class_name(class_names[top1_index])

    confidence_message = get_confidence_message(top1_confidence)

    explanation = (
        f"The model predicts '{top1_class}' with {top1_confidence * 100:.2f}% confidence. "
        "The top-3 predictions are shown because some crop diseases have visually similar symptoms, "
        "such as spots, discoloration, and blight patterns. The Grad-CAM heatmap highlights the image "
        "regions that most influenced the model's prediction."
    )

    gradcam_input = transform(pil_image).unsqueeze(0).to(device)
    cam, _ = gradcam_engine(gradcam_input, class_idx=top1_index)

    display_image = pil_image.resize((image_size, image_size))
    display_image = np.array(display_image).astype(np.float32) / 255.0

    overlay = overlay_cam(display_image, cam)
    overlay_uint8 = np.uint8(overlay * 255)

    return top3_results, confidence_message, explanation, overlay_uint8


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload a plant leaf image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top-3 Predictions"),
        gr.Textbox(label="Confidence / Reliability Message"),
        gr.Textbox(label="Prediction Explanation"),
        gr.Image(type="numpy", label="Grad-CAM Explainability Heatmap")
    ],
    title="Crop Disease Detection System",
    description=(
        "Upload a plant leaf image and the system will predict the most likely crop disease using "
        "EfficientNet-B0 transfer learning. The demo also provides top-3 predictions, a confidence-based "
        "warning, and a Grad-CAM heatmap for explainability."
    )
)

if __name__ == "__main__":
    demo.launch(share=True)
