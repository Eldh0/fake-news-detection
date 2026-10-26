"""
Evaluate a DistilBERT model on the held-out test set.

This script:
1. Loads saved_model/ (or a model specified with --model) and data/test.csv.
2. Runs predictions on the test set.
3. Calculates accuracy, precision, recall, F1-score, and a confusion matrix.
4. Saves the results to outputs/ so the Streamlit app can display them.

Run this after train_model.py:

    python evaluate_model.py

To evaluate the public Hub model instead of a locally trained model:

    python evaluate_model.py --model therealcyberlord/fake-news-classification-distilbert

You can also point at a different test file:

    python evaluate_model.py --test-csv path/to/test.csv
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for saving image files.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "saved_model"
TEST_CSV = PROJECT_ROOT / "data" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_JSON = OUTPUT_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PNG = OUTPUT_DIR / "confusion_matrix.png"
REPORT_TXT = OUTPUT_DIR / "classification_report.txt"
MAX_LENGTH = 256


def check_required_files(
    model_source: str,
    test_csv: Path,
    use_local_model: bool,
):
    """Stop early with a clear message if required files are missing."""
    if use_local_model and not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_DIR}.\n"
            "Run  python train_model.py  first, or pass --model with a Hub repo id."
        )
    if not test_csv.exists():
        raise FileNotFoundError(
            f"No test split found at {test_csv}.\n"
            "Run  python train_model.py  first, or create the split with "
            "--test-csv."
        )


def load_test_dataset(tokenizer, test_csv: Path, max_samples=None):
    """Load the saved test CSV and tokenize it."""
    test_df = pd.read_csv(test_csv)

    missing_columns = {"text", "label"} - set(test_df.columns)
    if missing_columns:
        raise ValueError(
            f"{test_csv} is missing these columns: {sorted(missing_columns)}"
        )

    if max_samples and len(test_df) > max_samples:
        test_df = test_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Using a random sample of {max_samples:,} test articles.")

    dataset = Dataset.from_pandas(
        test_df[["text", "label"]],
        preserve_index=False,
    )
    dataset = dataset.rename_column("label", "labels")

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(tokenize_batch, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")
    return dataset, test_df


def create_confusion_matrix_plot(matrix):
    """Save a clearly labeled confusion-matrix image for the app."""
    plt.figure(figsize=(7, 5.6), facecolor="#0b1220")
    axis = plt.gca()
    axis.set_facecolor("#0b1220")

    sns.heatmap(
        matrix,
        annot=False,
        cmap="Blues",
        cbar=False,
        xticklabels=["FAKE", "REAL"],
        yticklabels=["FAKE", "REAL"],
        linewidths=2,
        linecolor="#0b1220",
        ax=axis,
    )

    # Write the counts manually so text stays readable on both light and dark cells.
    darkest_cell = int(matrix.max()) if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            text_color = "#f8fbff" if value > darkest_cell * 0.55 else "#0b1220"
            axis.text(
                column + 0.5,
                row + 0.5,
                f"{value:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=17,
                fontweight="bold",
            )

    axis.set_xlabel("Predicted label", color="#dbeafe", fontsize=12, labelpad=12)
    axis.set_ylabel("True label", color="#dbeafe", fontsize=12, labelpad=12)
    axis.set_title(
        "Test-set Confusion Matrix",
        color="white",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    axis.tick_params(colors="#dbeafe", labelsize=11)

    plt.setp(axis.get_xticklabels(), color="#dbeafe")
    plt.setp(axis.get_yticklabels(), color="#dbeafe")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=200, facecolor="#0b1220")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fake-news classifier on a labelled test CSV."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model directory or Hugging Face Hub repo id. Defaults to the "
            "locally trained saved_model/ directory."
        ),
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        help="Path to the labelled test CSV. Defaults to data/test.csv.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Evaluate a random sample of this many test rows. Useful on a slow "
            "CPU; omit to evaluate the full test set."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_local_model = args.model is None
    model_source = str(MODEL_DIR) if use_local_model else args.model
    test_csv = Path(args.test_csv).expanduser().resolve() if args.test_csv else TEST_CSV

    check_required_files(model_source, test_csv, use_local_model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    print(f"Evaluating on: {device_name}")

    print(f"Loading model from: {model_source}")
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)

    test_dataset, test_df = load_test_dataset(
        tokenizer,
        test_csv,
        max_samples=args.max_samples,
    )
    print(f"Test articles: {len(test_df):,}")

    # Trainer.predict runs the model in evaluation mode and returns logits
    # and the true labels. No gradient updates happen here.
    eval_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "_evaluation_temp"),
        per_device_eval_batch_size=32,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    prediction_output = trainer.predict(test_dataset)
    logits = prediction_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    y_true = prediction_output.label_ids
    y_pred = np.argmax(logits, axis=1)

    # Accuracy = percentage of all test articles classified correctly.
    accuracy = accuracy_score(y_true, y_pred)

    # Class 1 (REAL) is the positive class for the headline metrics.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    # Also calculate per-class details. This is useful because a model can
    # perform well overall while doing poorly on one class.
    class_precision, class_recall, class_f1, class_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["FAKE", "REAL"],
        digits=4,
        zero_division=0,
    )

    create_confusion_matrix_plot(matrix)

    report_path = OUTPUT_DIR / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    results = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": matrix.tolist(),
        "labels": ["FAKE", "REAL"],
        "positive_class": "REAL",
        "test_samples": int(len(y_true)),
        "class_counts": {
            "FAKE": int((y_true == 0).sum()),
            "REAL": int((y_true == 1).sum()),
        },
        "per_class": {
            "FAKE": {
                "precision": round(float(class_precision[0]), 4),
                "recall": round(float(class_recall[0]), 4),
                "f1_score": round(float(class_f1[0]), 4),
                "support": int(class_support[0]),
            },
            "REAL": {
                "precision": round(float(class_precision[1]), 4),
                "recall": round(float(class_recall[1]), 4),
                "f1_score": round(float(class_f1[1]), 4),
                "support": int(class_support[1]),
            },
        },
        "model": (
            "distilbert-base-uncased (fine-tuned locally)"
            if use_local_model
            else model_source
        ),
        "model_source": model_source,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not use_local_model:
        results["evaluation_note"] = (
            "This public model was evaluated on this project's local test split. "
            "It may have seen this dataset during its own training, so treat these "
            "numbers as a reference rather than a clean held-out result."
        )

    METRICS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nTest-set results")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}  (positive class: REAL)")
    print(f"  Recall   : {recall:.4f}  (positive class: REAL)")
    print(f"  F1-score : {f1:.4f}  (positive class: REAL)")
    print("\nConfusion matrix [[true FAKE, false REAL], [false FAKE, true REAL]]:")
    print(matrix)
    print("\nPer-class report:")
    print(report_text)
    print(f"Metrics saved to: {METRICS_JSON}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PNG}")
    print(f"Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
