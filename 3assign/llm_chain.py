########
# Libraries
########

from config import (
    DATA_DIR, PLOTS_DIR, RESULTS_DIR,
    LOCAL_MODEL,        # e.g. "llama3.2", "mistral", "gemma3"
    OLLAMA_HOST,        # e.g. "http://localhost:11434"
    OLLAMA_MAX_WORKERS, # e.g. 4
)

import re
import json
import ollama
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report,
)

########
# Data collection + cleaning
########

def clean_tweet(text: str) -> str:
    text = re.sub(r"http\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    return text.strip()

X = pd.read_csv(DATA_DIR + "tweet_emotions_X.csv")
X["content"] = X["content"].astype(str).apply(clean_tweet)
y = pd.read_csv(DATA_DIR + "tweet_emotions_y.csv")["sentiment"]

# X = X.head(200)
# y = y.head(200)

########
# Structured output schema
########

SentimentLabel = Literal[
    "sadness", "enthusiasm", "neutral", "worry",
    "surprise", "love", "fun", "hate",
    "happiness", "boredom", "relief", "anger",
]

LABEL_DESCRIPTIONS = {
    "sadness":    "a feeling of sadness, sorrow, or unhappiness",
    "enthusiasm": "enthusiasm, excitement, or eagerness about something",
    "neutral":    "a neutral, objective, or emotionless tone",
    "worry":      "worry, concern, or anxious thoughts about something",
    "surprise":   "surprise, shock, or unexpected discovery",
    "love":       "love, affection, or deep care for someone or something",
    "fun":        "fun, playfulness, or lighthearted amusement",
    "hate":       "hate, strong dislike, or disgust toward something",
    "happiness":  "happiness, joy, or a positive and cheerful feeling",
    "boredom":    "boredom, disinterest, or a lack of engagement",
    "relief":     "relief, comfort, or the lifting of stress or worry",
    "anger":      "anger, frustration, or strong irritation",
}

LABELS_BLOCK = "\n".join(
    f"  - {label}: {desc}" for label, desc in LABEL_DESCRIPTIONS.items()
)

SYSTEM_PROMPT = (
    "You are a tweet sentiment classifier.\n"
    "Classify the tweet into exactly one of these emotion labels:\n\n"
    f"{LABELS_BLOCK}\n\n"
    "Return only valid JSON matching the required schema."
)


class TweetSentiment(BaseModel):
    reasoning: str = Field(description="One sentence justification for the chosen label.")
    label: SentimentLabel = Field(description="The single best-matching sentiment label.")


########
# Client + inference
########

client = ollama.Client(host=OLLAMA_HOST)


def classify_tweet(idx_text: tuple[int, str]) -> tuple[int, str]:
    idx, text = idx_text
    response = client.chat(
        model=LOCAL_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Tweet: {text}"},
        ],
        format=TweetSentiment.model_json_schema(),  # Ollama enforces schema at sampler level
        options={"temperature": 0},
    )
    result = TweetSentiment.model_validate_json(response.message.content)
    return idx, result.label


tweets = list(enumerate(X["content"].tolist()))
preds_indexed: list[tuple[int, str]] = []

with ThreadPoolExecutor(max_workers=OLLAMA_MAX_WORKERS) as executor:
    futures = {executor.submit(classify_tweet, item): item for item in tweets}
    for future in tqdm(as_completed(futures), total=len(tweets), desc="Classifying"):
        preds_indexed.append(future.result())

# Restore original order (ThreadPoolExecutor doesn't guarantee it)
preds_indexed.sort(key=lambda x: x[0])
preds = [label for _, label in preds_indexed]

########
# Save predictions
########

pred_df = pd.DataFrame({
    "text": X["content"].tolist(),
    "true": y.tolist(),
    "pred": preds,
})
pred_df.to_csv(RESULTS_DIR + "structured_llm_predictions.csv", index=False)

########
# Evaluation
########

labels = sorted(LABEL_DESCRIPTIONS.keys())

cm = confusion_matrix(y, preds, labels=labels, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format=".2f")
plt.tight_layout()
plt.savefig(PLOTS_DIR + "structured_llm_conf_matrix.png")

metrics = {
    "acc":                float(accuracy_score(y, preds)),
    "balanced_acc":       float(balanced_accuracy_score(y, preds)),
    "f1_macro":           float(f1_score(y, preds, average="macro")),
    "f1_weighted":        float(f1_score(y, preds, average="weighted")),
    "precision_macro":    float(precision_score(y, preds, average="macro")),
    "precision_weighted": float(precision_score(y, preds, average="weighted")),
    "recall_macro":       float(recall_score(y, preds, average="macro")),
    "recall_weighted":    float(recall_score(y, preds, average="weighted")),
    "report":             classification_report(y, preds, output_dict=True),
}

with open(RESULTS_DIR + "structured_llm.json", "w") as f:
    json.dump(metrics, f, indent=4)