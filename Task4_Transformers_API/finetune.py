import os
import time
import re
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "saved_model"


def clean_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def make_data():
    positive = [
        "I love this product", "excellent quality and fast delivery",
        "very happy with my purchase", "the service was wonderful",
        "this works perfectly", "great support from the team",
        "easy to use and reliable", "thank you for the quick help",
        "the item is fantastic", "friendly staff and good service",
        "I am satisfied with the result", "superb experience overall"
    ]
    negative = [
        "I hate this product", "terrible quality and slow delivery",
        "very unhappy with my purchase", "the service was awful",
        "this does not work", "poor support from the team",
        "hard to use and unreliable", "no one helped me",
        "the item is disappointing", "rude staff and bad service",
        "I am dissatisfied with the result", "worst experience overall"
    ]
    rows = []
    suffixes = [" today", " for me", " overall", " so far", " really"]
    for label, texts in [(1, positive), (0, negative)]:
        for i in range(15):
            for text in texts:
                rows.append({"text": clean_text(text + suffixes[i % len(suffixes)]), "label": label})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def main():
    df = make_data()
    train_df, eval_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize, batched=True)
    eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True)).map(tokenize, batched=True)
    train_ds = train_ds.remove_columns(["text"])
    eval_ds = eval_ds.remove_columns(["text"])
    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1}
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")
        }

    args = TrainingArguments(
        output_dir="training_output",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none"
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds,
        eval_dataset=eval_ds, compute_metrics=compute_metrics
    )
    start = time.time()
    trainer.train()
    train_seconds = time.time() - start
    metrics = trainer.evaluate()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tuned model saved to:", OUTPUT_DIR)
    print("Fine-tuning seconds:", round(train_seconds, 2))
    print("Evaluation metrics:", metrics)
    print("\nUse these measured metrics in the comparison table in your report.")
    print("Compare against Task 3's measured TF-IDF accuracy, macro F1 and times.")

if __name__ == "__main__":
    main()
