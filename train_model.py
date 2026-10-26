"""
Fine-tune DistilBERT for fake-news classification.

What this script does, in order:
1. Loads data/Fake.csv and data/True.csv.
2. Cleans the text, fixes labels, and removes exact duplicates.
3. Creates reproducible 80/10/10 train/validation/test splits.
4. Saves the held-out test set to data/test.csv (it is NOT used for training).
5. Tokenizes the text with the DistilBERT tokenizer.
6. Fine-tunes distilbert-base-uncased with Hugging Face Trainer.
7. Saves the best model and tokenizer to saved_model/.

Run this script only when you want to train the model:

    python train_model.py

For a quick CPU check, expect training to be slow. A GPU is strongly
recommended. See README.md for the Google Colab instructions.
"""

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Settings you can change
# ---------------------------------------------------------------------------
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256          # Longer articles are cut to this many tokens.
EPOCHS = 2                # Two epochs are a good starting point for DistilBERT.
BATCH_SIZE = 8            # Reduce to 4 if you run out of memory.
LEARNING_RATE = 2e-5
RANDOM_SEED = 42
MIN_TEXT_LENGTH = 20      # Ignore empty or almost-empty rows.

# ---------------------------------------------------------------------------
# Project paths (all paths are relative to this file, not the terminal)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FAKE_CSV = DATA_DIR / "Fake.csv"
TRUE_CSV = DATA_DIR / "True.csv"
TEST_CSV = DATA_DIR / "test.csv"
MODEL_DIR = PROJECT_ROOT / "saved_model"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"


def read_news_csv(path: Path) -> pd.DataFrame:
    """Read one Kaggle CSV file and give a helpful error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find {path}.\n"
            "Download the 'Fake and Real News Dataset' from Kaggle:\n"
            "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            f"Then put Fake.csv and True.csv inside: {DATA_DIR}"
        )

    # utf-8 is normal. latin-1 is a safe fallback for occasional Windows CSVs.
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def load_and_clean_data() -> pd.DataFrame:
    """
    Load both classes, combine title + article, and remove exact duplicates.

    Labels: 0 = FAKE, 1 = REAL.
    """
    fake_df = read_news_csv(FAKE_CSV)
    real_df = read_news_csv(TRUE_CSV)

    for name, frame in (("Fake.csv", fake_df), ("True.csv", real_df)):
        missing_columns = {"title", "text"} - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{name} is missing these columns: {sorted(missing_columns)}. "
                "Make sure you downloaded the correct Kaggle dataset."
            )

    # Add the label before joining the two files.
    fake_df["label"] = 0
    real_df["label"] = 1

    # Keep only the columns we need.
    columns = ["title", "text", "label"]
    combined = pd.concat(
        [fake_df[columns], real_df[columns]],
        ignore_index=True,
    )

    # Replace missing values with empty strings and remove extra whitespace.
    combined["title"] = combined["title"].fillna("").astype(str)
    combined["text"] = combined["text"].fillna("").astype(str)
    combined["text"] = (
        combined["title"].str.strip() + "\n\n" + combined["text"].str.strip()
    )
    combined["text"] = combined["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    combined = combined[["text", "label"]]

    # Drop rows that have no useful text.
    combined = combined[combined["text"].str.len() >= MIN_TEXT_LENGTH].copy()

    # A lowercase, whitespace-normalized key catches duplicates that differ
    # only by capitalization or repeated spaces. This happens BEFORE splitting,
    # so the same article cannot appear in both training and test data.
    combined["text_key"] = (
        combined["text"].str.lower().str.replace(r"\s+", " ", regex=True)
    )

    # If an identical article somehow has both labels, it is ambiguous.
    # Removing it is safer than teaching the model two answers for one text.
    labels_per_text = combined.groupby("text_key")["label"].nunique()
    conflicting_keys = set(labels_per_text[labels_per_text > 1].index)
    if conflicting_keys:
        print(f"Removing {len(conflicting_keys)} texts with conflicting labels...")
        combined = combined[~combined["text_key"].isin(conflicting_keys)]

    before_deduplication = len(combined)
    combined = combined.drop_duplicates(subset="text_key", keep="first")
    duplicates_removed = before_deduplication - len(combined)
    combined = combined.drop(columns=["text_key"]).reset_index(drop=True)

    # Shuffle in a reproducible way. The split functions also use the seed.
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if combined["label"].nunique() != 2:
        raise ValueError("Both FAKE and REAL examples are required for training.")

    print("\nDataset summary")
    print(f"  Total usable rows: {len(combined):,}")
    print(f"  Fake (0): {(combined['label'] == 0).sum():,}")
    print(f"  Real (1): {(combined['label'] == 1).sum():,}")
    print(f"  Exact duplicates removed: {duplicates_removed:,}")
    print(
        "  Note: exact duplicates are removed, but near-duplicate news stories "
        "can still make results look better than they really are.\n"
    )

    return combined


def make_splits(data: pd.DataFrame):
    """
    Create an 80% train / 10% validation / 10% test stratified split.

    Stratified means both classes keep roughly the same proportions in each set.
    """
    train_df, remaining_df = train_test_split(
        data,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=data["label"],
    )
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=remaining_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def tokenize_dataset(frame: pd.DataFrame, tokenizer):
    """Convert a pandas DataFrame into a tokenized Hugging Face Dataset."""
    dataset = Dataset.from_pandas(
        frame[["text", "label"]],
        preserve_index=False,
    )
    dataset = dataset.rename_column("label", "labels")

    def tokenize_batch(batch):
        # DistilBERT expects token IDs, not raw words. Truncation keeps long
        # articles within the model's input limit; padding is done later in
        # batches by DataCollatorWithPadding.
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(tokenize_batch, batched=True)
    dataset = dataset.remove_columns(["text"])

    # Trainer/DataCollator can convert these columns to PyTorch tensors.
    dataset.set_format("torch")
    return dataset


def compute_metrics(eval_prediction):
    """Calculate validation metrics after each epoch."""
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    labels = eval_prediction.label_ids
    predictions = np.argmax(logits, axis=-1)

    # Class 1 (REAL) is treated as the positive class here. This is only a
    # reporting convention; accuracy and the confusion matrix do not depend on it.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    # Show whether training will run on the GPU or the CPU.
    device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")
    if not torch.cuda.is_available():
        print(
            "WARNING: CPU training can take several hours and is not recommended "
            "for the full dataset. Use Google Colab with a GPU if possible."
        )

    data = load_and_clean_data()
    train_df, val_df, test_df = make_splits(data)

    # Save the test set now, but do not train on it. evaluate_model.py will
    # load this exact file after training is complete.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_df[["text", "label"]].to_csv(TEST_CSV, index=False)
    print(f"Test set saved to: {TEST_CSV}")
    print(f"  Train: {len(train_df):,} | Validation: {len(val_df):,} | Test: {len(test_df):,}\n")

    print(f"Loading tokenizer and model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1},
    )

    print("Tokenizing the text...")
    train_dataset = tokenize_dataset(train_df, tokenizer)
    val_dataset = tokenize_dataset(val_df, tokenizer)

    # Dynamic padding uses the longest sequence in each batch. It is faster
    # than padding every article to MAX_LENGTH.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TrainingArguments tells Trainer how to train: learning rate, batch size,
    # how often to evaluate, where to save checkpoints, and so on.
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=16,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",             # validate after every epoch
        save_strategy="epoch",             # save a checkpoint after every epoch
        load_best_model_at_end=True,        # restore the best validation F1
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",                  # do not connect to experiment trackers
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        dataloader_num_workers=0,          # avoids multiprocessing issues on Windows
        fp16=torch.cuda.is_available(),    # faster training on most modern GPUs
    )

    # Trainer is Hugging Face's ready-made training loop. It handles batching,
    # moving data to the GPU, backpropagation, evaluation, and checkpointing.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nFinal validation results:")
    validation_metrics = trainer.evaluate()
    for metric_name, metric_value in validation_metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"  {metric_name}: {metric_value:.4f}")

    # Save the best model in the standard Hugging Face format. This same folder
    # can be loaded by app.py or uploaded to the Hugging Face Hub.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    training_info = {
        "base_model": BASE_MODEL,
        "labels": {"0": "FAKE", "1": "REAL"},
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "validation_metrics": {
            name: round(float(value), 4)
            for name, value in validation_metrics.items()
            if isinstance(value, (int, float))
        },
    }
    (MODEL_DIR / "training_info.json").write_text(
        json.dumps(training_info, indent=2),
        encoding="utf-8",
    )

    # Remove temporary checkpoints so the model folder is easier to upload.
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

    print(f"\nBest model saved to: {MODEL_DIR}")
    print("Next step: run  python evaluate_model.py")


if __name__ == "__main__":
    main()
