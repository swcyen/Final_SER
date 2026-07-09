from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def make_classification_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_names: List[str],
) -> str:
    return classification_report(
        list(y_true),
        list(y_pred),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )


def save_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_names: List[str],
    output_path: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(list(y_true), list(y_pred))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
