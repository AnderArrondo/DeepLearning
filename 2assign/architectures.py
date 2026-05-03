
import torch
import torch.nn as nn

class CNN_Expresion_Recognition(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional part
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 1 channel of 3x3 kernel
            nn.ReLU(),
            nn.MaxPool2d(2),   #  Do to half /2, get most important thing

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  #Again
            nn.ReLU(),
            nn.MaxPool2d(2)    
        )

        # Fully connected
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                
            nn.Linear(32 * 12 * 12, 128),  #Total length
            nn.ReLU(),
            nn.Linear(128, 7)              # 7 emotions probabilities
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
