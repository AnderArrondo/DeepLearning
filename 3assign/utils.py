from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pandas as pd

import optuna

import config
import architechtures



X=pd.read_csv(config.DATA_DIR+"tweet_emotions_X.csv")
y=pd.read_csv(config.DATA_DIR+"tweet_emotions_y.csv")
#NO HAY BATCHES; SI QUITAS ESTO MUERE
X = X.head(200)
y = y[:200]

encoder = LabelEncoder()

y = encoder.fit_transform(
    y.values.ravel()
)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


def objective_function(trial):
    writer=SummaryWriter(f"runs/{config.STUDY_NAME}")

    classifier=None
    classifier_type=trial.suggest_categorical("classifier type", ["LSTM","GRU"])
    
    embedding_size=trial.suggest_int("embedding size",65,256)
    hidden_size=trial.suggest_int("hidden layer size", 64, 256)
    num_layers=trial.suggest_int("num_layers", 1,4)
    learning_rate=trial.suggest_float("lr", 1e-4, 5e-3,log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)


    X_tensor, vocab= create_embeddings(X_train["content"], max_length=100)#CREO QUE MAX EN DEL TWEET
    X_tensor=X_tensor.to(config.DEVICE)

    embedding=nn.Embedding(
        num_embeddings=len(vocab),
        embedding_dim=embedding_size    
    )

    if classifier_type=="LSTM":
        classifier=architechtures.LSTM_classifier(
            embedding,
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout
        )

    elif classifier_type=="GRU":
        classifier = architechtures.GRU_classifier(
            embedding,
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout)
    else:
        classifier=None


    classifier=classifier.to(config.DEVICE)

    X_test_tensor, _ = create_embeddings(
        X_test["content"],
        vocab=vocab,
        max_length=100
    )
    X_test_tensor = X_test_tensor.to(config.DEVICE)

    y_train_tensor = torch.tensor(y_train).long().to(config.DEVICE)
    train_dataset = TensorDataset(
        X_tensor,
        y_train_tensor
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    y_test_tensor = torch.tensor(y_test).long().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=learning_rate
    )

    epochs = config.n_epochs

    for epoch in range(epochs):

        classifier.train()
        epoch_loss=0
        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()

            outputs = classifier(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/Train", mean_loss,epoch)
        trial.report(mean_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()
        
    classifier.eval()

    with torch.no_grad():

        outputs = classifier(X_test_tensor)

        y_pred = torch.argmax(
            outputs,
            dim=1
        )
    fold_score = accuracy_score(
    y_test_tensor.cpu().numpy(),
    y_pred.cpu().numpy()
    )
    writer.add_scalar("Accuracy/Test",fold_score,trial.number)
    
    return fold_score





def create_embeddings(texts,vocab=None, max_length=None):
    """
    Convierte una lista de strings en tensores preparados
    para una LSTM/GRU.

    Pasos:
    1. Tokenización
    2. Vocabulario
    3. Texto -> índices
    4. Padding

    Parameters
    ----------
    texts : list[str]
        Lista de frases/tweets

    max_length : int | None
        Longitud máxima de secuencia

    Returns
    -------
    padded_sequences : torch.Tensor
        Tensor shape:
        (batch_size, sequence_length)

    vocab : dict
        Diccionario palabra -> índice
    """

    # ---------------------------------------------------
    # TOKENIZACIÓN
    # ---------------------------------------------------

    tokenized_texts = []

    for text in texts:

        tokens = text.lower().split()

        tokenized_texts.append(tokens)

    # ---------------------------------------------------
    # CREAR VOCABULARIO
    # ---------------------------------------------------

    all_words = []

    for tokens in tokenized_texts:
        all_words.extend(tokens)

    counter = Counter(all_words)

    vocab = {
            "<PAD>": 0,
            "<UNK>": 1
        }
    
    for idx, (word, _) in enumerate(counter.items()):
        vocab[word] = idx + 2

    # ---------------------------------------------------
    # TEXTO -> ÍNDICES
    # ---------------------------------------------------

    encoded_sequences = []

    for tokens in tokenized_texts:

        encoded = [
            vocab.get(word, vocab["<UNK>"])
            for word in tokens
        ]

        encoded_sequences.append(
            torch.tensor(encoded)
        )

    # ---------------------------------------------------
    # PADDING
    # ---------------------------------------------------

    padded_sequences = pad_sequence(
        encoded_sequences,
        batch_first=True,
        padding_value=0
    )

    # ---------------------------------------------------
    # RECORTAR / EXTENDER
    # ---------------------------------------------------

    if max_length is not None:

        current_length = padded_sequences.shape[1]

        # recortar
        if current_length > max_length:

            padded_sequences = padded_sequences[:, :max_length]

        # extender
        elif current_length < max_length:

            pad_size = max_length - current_length

            extra_padding = torch.zeros(
                (padded_sequences.shape[0], pad_size),
                dtype=torch.long
            )

            padded_sequences = torch.cat(
                [padded_sequences, extra_padding],
                dim=1
            )

    return padded_sequences.long(), vocab


