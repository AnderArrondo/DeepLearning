import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("./3assign/results/zero_shot_predictions.csv")

def simplify(x):
    x = x.lower()
    if x in ["sadness", "worry"]:
        return "distress"
    if x in ["happiness", "enthusiasm", "fun"]:
        return "joy"
    return x

df["true_s"] = df["true"].apply(simplify)
df["pred_s"] = df["pred"].apply(simplify)

labels = sorted(set(df["true_s"]) | set(df["pred_s"]))
cm = confusion_matrix(df["true_s"], df["pred_s"], labels=labels)

plt.imshow(cm, cmap="Blues")
plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
plt.yticks(range(len(labels)), labels)
plt.colorbar()
plt.tight_layout()
plt.show()