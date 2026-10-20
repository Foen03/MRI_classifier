"""
Brain Tumor MRI Classification – Starter
Datensatz: Kaggle "masoudnickparvar/brain-tumor-mri-dataset"
Klassen: glioma, meningioma, notumor, pituitary

Setup:
    pip install torch torchvision kagglehub scikit-learn matplotlib
"""

import os
import random
from pathlib import Path

import kagglehub
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------- Konfiguration
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
VAL_FRACTION = 0.15

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

def run_epoch(model, criterion, optimizer, dl, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = logits.argmax(1)
            total_loss += loss.item() * x.size(0)
            correct += (preds == y).sum().item()
            n += x.size(0)
            all_preds.append(preds.cpu()); all_labels.append(y.cpu())
    return total_loss / n, correct / n, torch.cat(all_preds), torch.cat(all_labels)


def main():
    # ---------------------------------------------------------------- Daten laden
    root = Path(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"))
    train_dir, test_dir = root / "Training", root / "Testing"
    print(f"Daten unter: {root}")

    # MRT-Schnitte sind Graustufen -> auf 3 Kanäle replizieren (für ImageNet-Gewichte)
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        norm,
    ])

    # Zwei Sichten auf denselben Ordner, damit Validation keine Augmentation bekommt
    full_train_aug = datasets.ImageFolder(train_dir, transform=train_tf)
    full_train_eval = datasets.ImageFolder(train_dir, transform=eval_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
    classes = full_train_aug.classes
    print(f"Klassen: {classes}")

    idx = np.random.permutation(len(full_train_aug))
    n_val = int(VAL_FRACTION * len(idx))
    train_ds = Subset(full_train_aug, idx[n_val:])
    val_ds = Subset(full_train_eval, idx[:n_val])

    loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2, pin_memory=(device == "cuda")
    )
    train_dl, val_dl, test_dl = loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ---------------------------------------------------------------- Modell
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---------------------------------------------------------------- Training
    best_val, ckpt = 0.0, "best_resnet18.pt"
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, *_ = run_epoch(model, criterion, optimizer, train_dl, train=True)
        va_loss, va_acc, *_ = run_epoch(model, criterion, optimizer, val_dl, train=False)
        scheduler.step()
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.3f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), ckpt)

    # ---------------------------------------------------------------- Test
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, te_acc, preds, labels = run_epoch(model, criterion, optimizer, test_dl, train=False)
    print(f"\nTest-Accuracy: {te_acc:.4f}\n")
    print(classification_report(labels, preds, target_names=classes, digits=3))
    print("Konfusionsmatrix (Zeilen = wahr, Spalten = vorhergesagt):")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
