import architectures as arch
import utils

import torch
import torch.optim as optim
import torch.nn as nn

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import numpy as np

from torch.utils.tensorboard import SummaryWriter

# ---------------- CONFIG ----------------
SEED=42
BATCH_SIZE=64
WORKERS=4
EPOCHS=100

MODELS_PATH = "./2assign/models"
LOGS_PATH = "./2assign/logs"

torch.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.benchmark=True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# ---------------- MODELS ----------------
models = {    
    "TransferVGG16":arch.TransferVGG16,
    "CNN": arch.CNN_Expresion_Recognition,
    "AlexNet48": arch.AlexNet48
}

# ---------------- TRAIN + EVAL LOOP ----------------
for model_name, model_class in models.items():

    print(f"\n===== Training {model_name} =====")
    writer=SummaryWriter(log_dir=f"./2assign/runs/{model_name}")

    model = model_class().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    #Train just last layer
    if "Transfer" in model_name:
        optimizer = optim.Adam(model.model.classifier[-1].parameters(), lr=0.001)

        for param in model.model.features.parameters():
            param.requires_grad = False

    # ---------------- TRANSFORMS ----------------
    if "Transfer" in model_name:
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), 
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #IMAGENET SCALING
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), 
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #IMAGENET SCALING
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
        )
        ])

    else:
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])

    # ---------------- DATASETS ----------------
    train_dataset = ImageFolder(
        root='2assign/data/train',
        transform=train_transform
    )

    test_dataset = ImageFolder(
        root='2assign/data/test',
        transform=test_transform
    )

    # ---------------- DATA LOADER ----------------

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    # -------- TRAIN --------

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0
        train_correct=0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*images.size(0)
            train_correct+= (torch.argmax(predictions,dim=1)==labels).sum().item()

        avg_loss = total_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)

        print(f"[{model_name}] Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f}")

    # -------- SAVE --------
    save_path=f"{MODELS_PATH}/{model_name}_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[{model_name}] Model Saved -> {save_path}")


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

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    writer.add_scalar("Accuracy/Test", accuracy, EPOCHS)
    writer.add_figure("Confusion Matrix", utils.conf_to_figure(cm,train_dataset.classes))
    writer.close()
    
    print(f"\n>>> {model_name} Accuracy: {accuracy:.4f}")

    utils.show_conf(cm, train_dataset.classes)