
#######
# Libraries
#######

from config import (
    DATA_DIR, PLOTS_DIR, RESULTS_DIR,
    MODEL, ZERO_SHOT_BATCH_SIZE, DEVICE
)

import json
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report

)
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

pred_df = pd.DataFrame({
    "text": X["content"],
    "true": y,
    "pred": preds
})

pred_df.to_csv(
    RESULTS_DIR + "zero_shot_predictions.csv",
    index=False
)

cm = confusion_matrix(y, preds, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=sorted(list(set(y))))
disp.plot(cmap="Blues", values_format=".2f")
plt.tight_layout()
plt.savefig(PLOTS_DIR + "zero_shot_conf_matrix.png")

metrics = {
    "acc": float(accuracy_score(y, preds)),
    "balanced_acc": float(balanced_accuracy_score(y, preds)),
    
    "f1_macro": float(f1_score(y, preds, average="macro")),
    "f1_weighted": float(f1_score(y, preds, average="weighted")),

    "precision_macro": float(precision_score(y, preds, average="macro")),
    "precision_weighted": float(precision_score(y, preds, average="weighted")),

    "recall_macro": float(recall_score(y, preds, average="macro")),
    "recall_weighted": float(recall_score(y, preds, average="weighted")),

    "report": classification_report(y, preds, output_dict=True)
}

with open(str(RESULTS_DIR + "zero_shot.json"), "w") as f:
    json.dump(metrics, f, indent=4)
