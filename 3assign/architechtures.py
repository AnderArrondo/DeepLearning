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
    
# Structurally this is a correct implementation of both an PyTorch LSTM and GRU classifier.

# Your flow is:

# tokens → embedding → RNN → last timestep → linear classifier

# which is the standard setup for sequence classification.

# A few important notes/improvements though.

# 2. Better to use hidden state directly

# Currently you do:

# last_hidden = output[:, -1, :]

# This works if:

# sequences are padded consistently
# or all same length

# But for padded NLP batches, the last timestep may just be padding.

# Better:

# LSTM
# output, (hidden, cell) = self.lstm(x)

# last_hidden = hidden[-1]
# GRU
# output, hidden = self.gru(x)

# last_hidden = hidden[-1]

# Why?

# hidden[-1] = final hidden state from top layer
# more robust
# standard practice
# 3. Potential issue with bidirectional models

# If later you add:

# bidirectional=True

# then dimensions change and your linear layer must become:

# nn.Linear(hidden_size * 2, config.N_CLASSES)

# Currently fine since not bidirectional.

# 4. Add dropout (recommended)

# Without dropout, overfitting is likely.

# Example:

# self.lstm = nn.LSTM(
#     embedding_size,
#     hidden_size,
#     num_layers,
#     batch_first=True,
#     dropout=0.3
# )

# Note:

# dropout only works if num_layers > 1
# 5. Naming convention

# Python style usually uses:

# class LSTMClassifier(nn.Module):

# instead of:

# LSTM_classifier

# Not required though.