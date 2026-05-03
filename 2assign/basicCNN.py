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

transform = transforms.Compose([ #Ensure all images are ok and as tesnor
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# TRAIN
train_dataset = ImageFolder(
    root='2assign/data/train',
    transform=transform
)

# TEST
test_dataset = ImageFolder(
    root='2assign/data/test',
    transform=transform
)

model=arch.CNN_Expresion_Recognition().to(device)

#DATALOADERS
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

#LOSS AND OPTIMISER

criterion = nn.CrossEntropyLoss() #Good for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)


#Train

epochs = 15

for epoch in range(epochs):

    model.train()  

    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # forward pass
        predictions = model(images)

        # error
        loss = criterion(predictions, labels)
    
        #BACKWARD
        # gradient restart
        optimizer.zero_grad()

        # backpropagation
        loss.backward()

        # step ADAM
        optimizer.step()

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


#Test

model.eval()


all_preds = []
all_labels = []

with torch.no_grad():

    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)

        predicted = torch.argmax(predictions, dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

accuracy = sum([p == t for p, t in zip(all_preds, all_labels)]) / len(all_labels)
print("Accuracy:", accuracy)
utils.show_conf(cm,train_dataset.classes)
