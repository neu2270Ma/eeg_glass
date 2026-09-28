from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from simple_utils import (
    MODEL_PATH,
    NORMALIZATION_PATH,
    TEST_DATA_PATH,
    binary_metrics,
    bipolarize,
    head_agreement,
    level_metrics,
    load_data,
    load_final_model,
    load_normalization,
    normalize,
    selection_f1,
)


BATCH_SIZE = 8
NUM_WORKERS = 0


def _number(value: float) -> str:
    return f"{float(value):.4f}" if np.isfinite(value) else "N/A"


def main() -> None:
    raw_signals, binary_labels, level_labels = load_data(TEST_DATA_PATH)
    signals = bipolarize(raw_signals)
    del raw_signals
    mean, std, eps = load_normalization(NORMALIZATION_PATH)
    signals = normalize(signals, mean, std, eps)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        TensorDataset(signals, binary_labels, level_labels),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    model = load_final_model(MODEL_PATH).to(device)
    model.eval()

    binary_true, binary_pred, binary_scores = [], [], []
    level_true, level_pred, level_scores = [], [], []
    with torch.no_grad():
        for batch_signals, batch_binary, batch_level in loader:
            _, binary_logits, level_logits = model(
                batch_signals.to(device, non_blocking=True)
            )
            binary_probability = torch.softmax(binary_logits, dim=1)
            level_probability = torch.softmax(level_logits, dim=1)
            binary_true.append(batch_binary.numpy())
            binary_pred.append(binary_probability.argmax(1).cpu().numpy())
            binary_scores.append(binary_probability[:, 1].cpu().numpy())
            level_true.append(batch_level.numpy())
            level_pred.append(level_probability.argmax(1).cpu().numpy())
            level_scores.append(level_probability.cpu().numpy())

    binary_true = np.concatenate(binary_true)
    binary_pred = np.concatenate(binary_pred)
    level_true = np.concatenate(level_true)
    level_pred = np.concatenate(level_pred)
    binary = binary_metrics(binary_true, binary_pred, np.concatenate(binary_scores))
    level = level_metrics(level_true, level_pred, np.concatenate(level_scores))
    agreement = head_agreement(binary_pred, level_pred)
    selection = selection_f1(binary["f1"], level["macro_f1"])

    print("=" * 50)
    print("BINARY RESULT")
    print("=" * 50)
    for key in ("tp", "fp", "tn", "fn"):
        print(f"{key.upper():<8}: {binary[key]}")
    for label, key in (
        ("Acc", "accuracy"), ("BA", "balanced_accuracy"),
        ("Prec.", "precision"), ("Recall", "recall"), ("F1", "f1"),
        ("AUC", "auc"), ("AP", "average_precision"),
    ):
        print(f"{label:<8}: {_number(binary[key])}")

    print("\n" + "=" * 50)
    print("LEVEL RESULT")
    print("=" * 50)
    for label, key in (
        ("Acc", "accuracy"), ("BA", "balanced_accuracy"),
        ("MacroPrec.", "macro_precision"), ("MacroRecall", "macro_recall"),
        ("MacroF1", "macro_f1"), ("MacroAUC", "macro_auc"),
        ("MacroAP", "macro_average_precision"), ("MAE", "mae"),
    ):
        print(f"{label:<12}: {_number(level[key])}")
    distances = level["distance_counts"]
    print(f"Exact       : {distances['exact']}")
    print(f"Off-by-1    : {distances['off_by_1']}")
    print(f"Off-by-2    : {distances['off_by_2']}")
    print(f"Off-by-3    : {distances['off_by_3']}")
    print("\nLevel Confusion Matrix")
    print("True\\Pred      0      1      2      3")
    for class_index, row in enumerate(level["confusion_matrix"]):
        print(f"{class_index:<10}" + "".join(f"{int(value):>7}" for value in row))
    print("\nClass  Precision  Recall  F1")
    for class_index in range(4):
        values = level["per_class"][class_index]
        print(
            f"{class_index:<5}  {values['precision']:.4f}     "
            f"{values['recall']:.4f}  {values['f1']:.4f}"
        )

    print("\n" + "=" * 50)
    print("DUAL-HEAD")
    print("=" * 50)
    print(f"Binary F1      : {_number(binary['f1'])}")
    print(f"Level Macro-F1 : {_number(level['macro_f1'])}")
    print(f"Selection F1   : {_number(selection)}")
    print(f"Head Agreement : {_number(agreement)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
