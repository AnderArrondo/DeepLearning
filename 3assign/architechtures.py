import torch.nn as nn

import config

class LSTM_classifier(nn.Module):

    def __init__(self, embedding, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.embedding=embedding

        self.lstm = nn.LSTM(embedding_size,hidden_size, num_layers, batch_first=True)  

        self.fc = nn.Linear(hidden_size, config.N_CLASSES)

    def forward(self, x):
        x = self.embedding(x)

        output, hidden = self.lstm(x)

        last_hidden = output[:, -1, :]

        out = self.fc(last_hidden)

        return out

class GRU_classifier(nn.Module):

    def __init__(self, embedding, embedding_size, hidden_size, num_layers):
        super().__init__()

        self.embedding=embedding

        self.gru = nn.GRU(embedding_size,hidden_size, num_layers,batch_first=True)  

        self.fc = nn.Linear(hidden_size, config.N_CLASSES)

    def forward(self, x):
        x = self.embedding(x)

        output, hidden = self.gru(x)

        last_hidden = output[:, -1, :]

        out = self.fc(last_hidden)

        return out