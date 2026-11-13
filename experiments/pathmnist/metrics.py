from collections.abc import Sequence

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor


def classification_metrics(
    logits: Tensor, labels: Tensor, class_names: Sequence[str] = ()
) -> dict[str, float]:
    """Loss, accuracy, balanced accuracy, macro F1, macro one-vs-rest AUC.

    With `class_names`, also `recall_{name}` for every class.
    """
    probabilities = logits.double().softmax(dim=1).numpy()
    truth = labels.numpy()
    predicted = probabilities.argmax(axis=1)
    classes = np.arange(probabilities.shape[1])
    scores = {
        "loss": F.cross_entropy(logits, labels).item(),
        "accuracy": accuracy_score(truth, predicted),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, average="macro", labels=classes, zero_division=0),
        "auc": roc_auc_score(truth, probabilities, multi_class="ovr", average="macro", labels=classes),
    }
    if class_names:
        recalls = recall_score(truth, predicted, average=None, labels=classes, zero_division=0)
        scores |= {f"recall_{name}": float(recall) for name, recall in zip(class_names, recalls, strict=True)}
    return scores


def confusion(logits: Tensor, labels: Tensor) -> list[list[int]]:
    classes = np.arange(logits.shape[1])
    return confusion_matrix(labels.numpy(), logits.argmax(dim=1).numpy(), labels=classes).tolist()
