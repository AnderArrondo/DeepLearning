
#######
# Libraries
#######

from config import DATA_DIR, PLOTS_DIR, MODEL, ZERO_SHOT_BATCH_SIZE, DEVICE

import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

torch.backends.cudnn.benchmark = True

#######
# Data collection + splitting
#######
X = pd.read_csv(str(DATA_DIR + "tweet_emotions_X.csv"))
y = pd.read_csv(str(DATA_DIR + "tweet_emotions_y.csv"))["sentiment"]
candidate_labels = y.unique()

X = X.head(1000)
y = y.head(1000)
candidate_labels = y.unique()
#######
# Pipeline
#######
classifier = pipeline("zero-shot-classification", model=MODEL)

preds = []
for i in tqdm(range(0, len(X), ZERO_SHOT_BATCH_SIZE)):
    batch = X.iloc[i:i+ZERO_SHOT_BATCH_SIZE]["content"].tolist()

    results = classifier(
        batch,
        candidate_labels=candidate_labels,
        batch_size=len(batch),
        device=DEVICE
    )
    preds.extend([res["labels"][0] for res in results])

cm = confusion_matrix(y, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=sorted(list(set(y))))
disp.plot(cmap="Blues")
plt.savefig(PLOTS_DIR + "zero_shot_conf_matrix.png")

