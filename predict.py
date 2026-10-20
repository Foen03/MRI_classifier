"""
Inferenz für das Brain-Tumor-MRI-Modell (ResNet-18)

Nutzung:
    python predict.py bild.jpg
    python predict.py bild1.jpg bild2.png
    python predict.py ordner/                  # alle Bilder im Ordner
    python predict.py bild.jpg --ckpt best_resnet18.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Reihenfolge wie bei ImageFolder im Training (alphabetisch sortiert)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Identisch zur eval-Transformation im Training
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt: Path, device: str) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()


def collect_images(paths: list[str]) -> list[Path]:
    images = []
    for p in map(Path, paths):
        if p.is_dir():
            images += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        elif p.is_file():
            images.append(p)
        else:
            print(f"Übersprungen (nicht gefunden): {p}")
    return images


@torch.no_grad()
def predict(model: nn.Module, img_path: Path, device: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(device)       # (1, 3, H, W)
    logits = model(x)                                 # (1, K)
    return torch.softmax(logits, dim=1).squeeze(0).cpu()  # p_k = exp(z_k) / sum_j exp(z_j)


def main():
    parser = argparse.ArgumentParser(description="MRT-Bilder klassifizieren")
    parser.add_argument("inputs", nargs="+", help="Bilddatei(en) oder Ordner")
    parser.add_argument("--ckpt", default="best_resnet18.pt", help="Pfad zum Modell")
    args = parser.parse_args()

    device = get_device()
    model = load_model(Path(args.ckpt), device)
    images = collect_images(args.inputs)
    if not images:
        print("Keine Bilder gefunden.")
        return

    for img_path in images:
        probs = predict(model, img_path, device)
        top = int(probs.argmax())
        print(f"\n{img_path.name}  ->  {CLASSES[top]}  ({probs[top]:.1%})")
        for k in probs.argsort(descending=True):
            bar = "█" * int(probs[k] * 30)
            print(f"   {CLASSES[k]:<11} {probs[k]:6.1%}  {bar}")


if __name__ == "__main__":
    main()