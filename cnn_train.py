import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize = (7, 6))
    plt.imshow(cm, interpolation = "nearest")
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation = 45, ha = "right")
    plt.yticks(range(len(class_names)), class_names)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi = 150)
    plt.close()


def main(img_root = "images", out_dir = "output", epochs = 8, batch_size = 32, lr = 1e-3, seed = 42, num_workers = 0):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_dir = os.path.join(img_root, "train")
    test_dir = os.path.join(img_root, "test")

    # Transforms (ImageNet pretrained expects 224x224, normalized)
    train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.2), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    test_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    train_ds = datasets.ImageFolder(train_dir, transform = train_tf)
    test_ds = datasets.ImageFolder(test_dir, transform = test_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle=True, num_workers = num_workers)
    test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle=False, num_workers = num_workers)

    # Model: MobileNetV2 (pretrained) + new classifier
    model = models.mobilenet_v2(weights = models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - train_acc: {acc:.4f}")

    # Evaluate
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim = 1).cpu().numpy()

            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    # Report
    report = classification_report(y_true, y_pred, target_names = class_names, output_dict=True)
    report_path = os.path.join(out_dir, "cnn_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent = 2)

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(out_dir, "cnn_confusion.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    # Save model
    model_path = os.path.join(out_dir, "cnn_mobilenetv2.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names
    }, model_path)

    print("\nCNN done.")
    print("Saved:", report_path)
    print("Saved:", cm_path)
    print("Saved:", model_path)
    print("Macro F1:", report["macro avg"]["f1-score"])
    print("Accuracy:", report["accuracy"])


if __name__ == "__main__":
    main()