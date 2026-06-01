from typing import Dict, Iterable

from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
