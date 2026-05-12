from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import pandas as pd

import optuna

import config
import architechtures

X=pd.read_csv(config.DATA_DIR+"tweet_emotions_X.csv")
y=pd.read_csv(config.DATA_DIR+"tweet_emotions_y.csv")
#NO HAY BATCHES; SI QUEITAS ESTO MUERE
X = X.head(200)
y = y[:200]

encoder = LabelEncoder()

y = encoder.fit_transform(
    y.values.ravel()
)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


def objective_function(trial):
    #CLASSIFIERS: LSTM GRU
    classifier=None
    classifier_type=trial.suggest_categorical("classifier type", ["LSTM","GRU"])
    
    embedding_size=trial.suggest_int("embedding size",20,300)
    hidden_size=trial.suggest_int("hidden layer size", 10,30)
    num_layers=trial.suggest_int("num neurons per layer", 2,10)

    X_tensor, vocab= create_embeddings(X_train["content"], max_length=100)#CREO QUE MAX EN DEL TWEET
    embedding=nn.Embedding(
        num_embeddings=len(vocab),
        embedding_dim=embedding_size    
    )
    if classifier_type=="LSTM":
        classifier=architechtures.LSTM_classifier(
            embedding,
            embedding_size,
            hidden_size,
            num_layers
        )

    elif classifier_type=="GRU":
        classifier = architechtures.GRU_classifier(
            embedding,
            embedding_size,
            hidden_size,
            num_layers)
    else:
        classifier=None



    # ME FALTA PREDECIR AQUI
    X_test_tensor, _ = create_embeddings(
        X_test["content"],
        max_length=100
    )

    #VOCAB TRAIN Y TEST DSITINTO--> MALISIMO
    y_train_tensor = torch.tensor(y_test).long()

    y_test_tensor = torch.tensor(y_test).long()

    criterion = nn.CrossEntropyLoss()
    learning_rate=trial.suggest_float("lr", 10e-5,10e-2,log=True)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=learning_rate
    )

    epochs = config.n_epochs

    for epoch in range(epochs):

        classifier.train()

        optimizer.zero_grad()

        outputs = classifier(X_tensor)

        print(outputs.shape)
        print(y_train_tensor.shape)

        loss = criterion(
            outputs,
            y_train_tensor
        )

        loss.backward()

        optimizer.step()

        trial.report(loss.item(), epoch)

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
    y_test_tensor.numpy(),
    y_pred.numpy()
    )
    
    return fold_score





def create_embeddings(texts, max_length=None):
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
        word: idx + 1
        for idx, (word, _) in enumerate(counter.items())
    }

    # 0 reservado para padding
    vocab["<PAD>"] = 0

    # ---------------------------------------------------
    # TEXTO -> ÍNDICES
    # ---------------------------------------------------

    encoded_sequences = []

    for tokens in tokenized_texts:

        encoded = [
            vocab[word]
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

# The biggest issue is this:

# X_test_tensor, _ = create_embeddings(X_test["content"])

# You are creating a different vocabulary for test data.

# That is a major NLP mistake.

# A word may map to:

# "happy" -> 15 in train
# "happy" -> 82 in test

# which completely breaks inference.

# Main Problems
# 1. Different vocabularies for train/test ❌

# This is the biggest issue.

# Your current code:

# X_tensor, vocab = create_embeddings(X_train["content"])

# X_test_tensor, _ = create_embeddings(X_test["content"])

# creates:

# train vocab
# test vocab

# independently.

# This makes embeddings meaningless.

# FIX

# You need:

# build vocab ONLY from train
# reuse same vocab for test
# 2. Critical bug: wrong labels ❌

# You wrote:

# y_train_tensor = torch.tensor(y_test).long()

# That is wrong.

# Should be:

# y_train_tensor = torch.tensor(y_train).long()

# Right now you are training on test labels.

# 3. OOV words (unknown words) ❌

# Your current encoding assumes every word exists:

# vocab[word]

# But test data may contain unseen words.

# This will crash.

# You need:

# <UNK> token

# Example:

# vocab = {
#     "<PAD>": 0,
#     "<UNK>": 1
# }

# then:

# vocab.get(word, vocab["<UNK>"])
# 4. You are training on the entire dataset as one batch ⚠️

# This works for tiny experiments:

# outputs = classifier(X_tensor)

# But:

# no minibatching
# no shuffling
# poor scalability

# Not wrong for debugging though.

# 5. Hyperparameter names are misleading ⚠️
# num_layers=trial.suggest_int("num neurons per layer", 2,10)

# This is not neurons per layer.

# It is:

# number of recurrent layers

# Better:

# "num_layers"
# 6. Your sequence handling is weak ⚠️

# You currently do:

# last_hidden = output[:, -1, :]

# But you use padding.

# So the final token may be padding.

# You should use:

# hidden[-1]

# as discussed before.

# 7. No device handling (CPU/GPU) ⚠️

# You should eventually add:

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# and move tensors/models.