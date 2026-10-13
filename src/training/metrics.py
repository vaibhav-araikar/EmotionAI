"""Evaluation metrics and confusion matrix utilities."""
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.data.dataset import EMOTION_LABELS

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "report": classification_report(
            y_true, y_pred, labels=list(range(len(EMOTION_LABELS))),
            target_names=EMOTION_LABELS, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(EMOTION_LABELS)))
        ),
    }
