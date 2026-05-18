
import sys
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.model import create_model


MODEL_PATH = PROJECT_DIR / "models" / "best_model.pth"


def format_class_name(class_name):
    return class_name.replace("___", " - ").replace("_", " ")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
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


def predict(image):
    """
    Receives an uploaded image and returns top class probabilities.
    """
    if image is None:
        return {}

    image = Image.fromarray(image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, k=3)

    results = {}

    for prob, idx in zip(top_probs, top_indices):
        class_name = format_class_name(class_names[idx.item()])
        results[class_name] = float(prob.item())

    return results


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload a plant leaf image"),
    outputs=gr.Label(num_top_classes=3, label="Top-3 Predictions"),
    title="Crop Disease Detection System",
    description=(
        "Upload a plant leaf image and the model will predict the most likely "
        "crop disease class using an EfficientNet-B0 transfer learning model."
    )
)

if __name__ == "__main__":
    demo.launch(share=True)
