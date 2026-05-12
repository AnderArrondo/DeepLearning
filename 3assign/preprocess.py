
from config import ORIG_CSV, DATA_DIR, PLOTS_DIR

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

PLOT_OR_SAVE = "save" # ["save", "plot"] alternatives

df = pd.read_csv(ORIG_CSV)
df.drop(columns=["tweet_id"], inplace=True)

empty_idx = df["sentiment"] == "empty"

X = df.loc[-empty_idx, "content"]
X_to_predict = df.loc[empty_idx, "content"]
y = df.loc[-empty_idx, "sentiment"]

print("# Full dataset")
print("Shape:", df.shape)
print(df.head())

print("\n# X")
print("Shape:", X.shape)
print(X.head())

print("\n# X to predict")
print("Shape:", X_to_predict.shape)
print(X_to_predict.head())

print("\n# y")
print("Shape:", y.shape)
print(y.head())

print("\n# Labels")
print(y.unique())

X.to_csv(str(DATA_DIR + "tweet_emotions_X.csv"), index=False)
X_to_predict.to_csv(str(DATA_DIR + "tweet_emotions_X_to_predict.csv"), index=False)
y.to_csv(str(DATA_DIR + "tweet_emotions_y.csv"), index=False)

(
    df["sentiment"]
    .value_counts()
    .sort_values(ascending=True)
    .plot(kind="barh", figsize=(12, 6))
)
plt.title("Sentiment Distribution")
plt.ylabel("Sentiment labels")
plt.xlabel("# of ocurrences")

plt.tight_layout()
if PLOT_OR_SAVE == "plot":
    plt.show()
else:
    plt.savefig(PLOTS_DIR + "label_distribution.png")