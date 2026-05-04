import architectures as arch
import utils

import torch
import torch.optim as optim
import torch.nn as nn

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# ---------------- DATASETS ----------------
train_dataset = ImageFolder(
    root='2assign/data/train',
    transform=transform
)

test_dataset = ImageFolder(
    root='2assign/data/test',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------- MODELS ----------------
models = {
    "CNN": arch.CNN_Expresion_Recognition,
    "AlexNet48": arch.AlexNet48
}

# ---------------- TRAIN + EVAL LOOP ----------------
for model_name, model_class in models.items():

    print(f"\n===== Training {model_name} =====")

    model = model_class().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 50

    # -------- TRAIN --------
    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[{model_name}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # -------- TEST --------
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            predicted = torch.argmax(predictions, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    accuracy = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)

    print(f"\n>>> {model_name} Accuracy: {accuracy:.4f}")

    utils.show_conf(cm, train_dataset.classes)