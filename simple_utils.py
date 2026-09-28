from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from model import DualHeadBIOT


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "train.pt"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "test.pt"
PRETRAINED_CHECKPOINT = PROJECT_ROOT / "pretrained_models" / "EEG-PREST-16-channels.ckpt"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model_best.pth"
NORMALIZATION_PATH = MODEL_DIR / "normalization.npz"

PROFESSIONAL_CHANNEL_NAMES = (
    "P7", "P4", "Cz", "Pz", "P3", "P8", "O1", "O2",
    "T8", "F8", "C4", "F4", "Fp2", "Fz", "C3", "F3",
    "Fp1", "T7", "F7", "Oz", "PO4", "FC6", "FC2", "AF4",
    "CP6", "CP2", "CP1", "CP5", "FC1", "FC5", "AF3", "PO3",
)
BIPOLAR_PAIRS = (
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
)
_CHANNEL_INDEX = {
    name.casefold(): index for index, name in enumerate(PROFESSIONAL_CHANNEL_NAMES)
}
_ANODES = tuple(_CHANNEL_INDEX[left.casefold()] for left, _ in BIPOLAR_PAIRS)
_CATHODES = tuple(_CHANNEL_INDEX[right.casefold()] for _, right in BIPOLAR_PAIRS)


def load_data(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file does not exist: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {"x", "y_binary", "y_level"}:
        raise ValueError(f"Data must contain only x/y_binary/y_level: {path}")
    x, y_binary, y_level = payload["x"], payload["y_binary"], payload["y_level"]
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float32:
        raise ValueError(f"x must be a float32 tensor: {path}")
    if x.ndim != 3 or tuple(x.shape[1:]) != (32, 2000):
        raise ValueError(f"x must have shape [N,32,2000], got {tuple(x.shape)}: {path}")
    for name, labels in (("y_binary", y_binary), ("y_level", y_level)):
        if not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64:
            raise ValueError(f"{name} must be an int64 tensor: {path}")
        if labels.ndim != 1 or labels.shape[0] != x.shape[0]:
            raise ValueError(f"{name} must have shape [N]: {path}")
    binary_values = set(int(value) for value in torch.unique(y_binary).tolist())
    level_values = set(int(value) for value in torch.unique(y_level).tolist())
    if not binary_values or not binary_values.issubset({0, 1}):
        raise ValueError(f"y_binary must contain only 0/1: {path}")
    if not level_values or not level_values.issubset({0, 1, 2, 3}):
        raise ValueError(f"y_level must contain only 0/1/2/3: {path}")
    if not torch.equal(y_binary, (y_level > 0).to(torch.int64)):
        raise ValueError(f"y_binary and y_level are inconsistent: {path}")
    return x, y_binary, y_level


def bipolarize(signals: torch.Tensor) -> torch.Tensor:
    if signals.ndim != 3 or signals.shape[1] != 32:
        raise ValueError(f"Expected [N,32,T], got {tuple(signals.shape)}")
    output = torch.empty(
        (signals.shape[0], 16, signals.shape[2]),
        dtype=signals.dtype,
        device=signals.device,
    )
    for index, (anode, cathode) in enumerate(zip(_ANODES, _CATHODES)):
        torch.sub(signals[:, anode], signals[:, cathode], out=output[:, index])
    return output


def fit_normalization(signals: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    if signals.ndim != 3 or signals.shape[1] != 16:
        raise ValueError(f"Expected [N,16,T], got {tuple(signals.shape)}")
    mean = signals.mean(dim=(0, 2))
    std = signals.std(dim=(0, 2), unbiased=False)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("Training normalization is non-finite or has zero-variance channels")
    return mean.numpy().astype(np.float32), std.numpy().astype(np.float32)


def save_normalization(path: Path, mean: np.ndarray, std: np.ndarray, eps: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
        eps=np.asarray(eps, dtype=np.float64),
    )


def load_normalization(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Normalization file does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"mean", "std", "eps"}:
            raise ValueError(f"Normalization must contain mean/std/eps only: {path}")
        mean = payload["mean"].astype(np.float32)
        std = payload["std"].astype(np.float32)
        eps = float(payload["eps"])
    if mean.shape != (16,) or std.shape != (16,):
        raise ValueError(f"Normalization mean/std must have shape [16]: {path}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"Invalid normalization values: {path}")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError(f"Invalid normalization eps: {path}")
    return mean, std, eps


def normalize(
    signals: torch.Tensor, mean: np.ndarray, std: np.ndarray, eps: float
) -> torch.Tensor:
    if signals.ndim != 3 or signals.shape[1] != 16:
        raise ValueError(f"Expected [N,16,T], got {tuple(signals.shape)}")
    mean_tensor = torch.as_tensor(mean, dtype=signals.dtype).reshape(1, 16, 1)
    std_tensor = torch.as_tensor(std, dtype=signals.dtype).reshape(1, 16, 1)
    normalized = ((signals - mean_tensor) / (std_tensor + eps)).contiguous()
    if not torch.isfinite(normalized).all():
        raise ValueError("Normalized signals contain NaN/Inf")
    return normalized


def build_model(
    load_pretrained: bool = False,
    *,
    emb_size: int = 256,
    heads: int = 8,
    depth: int = 4,
    n_channels: int = 16,
    n_fft: int = 200,
    hop_length: int = 100,
) -> DualHeadBIOT:
    model = DualHeadBIOT(
        emb_size=emb_size,
        heads=heads,
        depth=depth,
        n_channels=n_channels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    if load_pretrained:
        if not PRETRAINED_CHECKPOINT.is_file():
            raise FileNotFoundError(f"PREST checkpoint does not exist: {PRETRAINED_CHECKPOINT}")
        encoder_state = torch.load(PRETRAINED_CHECKPOINT, map_location="cpu")
        if type(encoder_state) is not OrderedDict:
            raise RuntimeError("PREST checkpoint must be a pure encoder OrderedDict")
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model


def save_final_model(model: torch.nn.Module, path: Path = MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"model_state_dict": state}, path)


def load_final_model(path: Path = MODEL_PATH) -> DualHeadBIOT:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model does not exist: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {"model_state_dict"}:
        raise RuntimeError(f"Model checkpoint must contain only model_state_dict: {path}")
    model = build_model(load_pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def _vector(values: Sequence, name: str, dtype) -> np.ndarray:
    output = np.asarray(values, dtype=dtype)
    if output.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return output


def _rank_metrics(binary_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    positives = int(np.sum(binary_true == 1))
    negatives = int(binary_true.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_true = binary_true[order]
    indices = np.r_[np.where(np.diff(sorted_scores))[0], binary_true.size - 1]
    cumulative_tp = np.cumsum(sorted_true == 1)[indices]
    cumulative_fp = 1 + indices - cumulative_tp
    recall = cumulative_tp / positives
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)
    auc = float(np.trapz(np.r_[0.0, recall], np.r_[0.0, cumulative_fp / negatives]))
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return auc, ap


def binary_metrics(true, predicted, probability_1) -> Dict[str, float]:
    true = _vector(true, "binary true labels", np.int64)
    predicted = _vector(predicted, "binary predictions", np.int64)
    scores = _vector(probability_1, "binary probabilities", np.float64)
    if true.size == 0 or predicted.shape != true.shape or scores.shape != true.shape:
        raise ValueError("Binary metric inputs must have equal nonzero length")
    tp = int(np.sum((true == 1) & (predicted == 1)))
    fp = int(np.sum((true == 0) & (predicted == 1)))
    tn = int(np.sum((true == 0) & (predicted == 0)))
    fn = int(np.sum((true == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    auc, ap = _rank_metrics((true == 1).astype(np.int64), scores)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": float((tp + tn) / true.size),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "auc": auc, "average_precision": ap,
    }


def level_metrics(true, predicted, probabilities) -> Dict[str, object]:
    true = _vector(true, "level true labels", np.int64)
    predicted = _vector(predicted, "level predictions", np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if true.size == 0 or predicted.shape != true.shape or scores.shape != (true.size, 4):
        raise ValueError("Level metric inputs have incompatible shapes")
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (true, predicted), 1)
    per_class = {}
    precisions, recalls, f1_values = [], [], []
    auc_values, ap_values = [], []
    for class_index in range(4):
        tp = int(matrix[class_index, class_index])
        fp = int(matrix[:, class_index].sum() - tp)
        fn = int(matrix[class_index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_index] = {
            "precision": float(precision), "recall": float(recall), "f1": float(f1)
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
        one_vs_rest = (true == class_index).astype(np.int64)
        auc, ap = _rank_metrics(one_vs_rest, scores[:, class_index])
        if np.isfinite(auc):
            auc_values.append(auc)
        if np.isfinite(ap):
            ap_values.append(ap)
    distances = np.abs(predicted - true)
    return {
        "accuracy": float(np.mean(predicted == true)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "macro_auc": float(np.mean(auc_values)) if auc_values else float("nan"),
        "macro_average_precision": float(np.mean(ap_values)) if ap_values else float("nan"),
        "mae": float(np.mean(distances)),
        "distance_counts": {
            "exact": int(np.sum(distances == 0)),
            "off_by_1": int(np.sum(distances == 1)),
            "off_by_2": int(np.sum(distances == 2)),
            "off_by_3": int(np.sum(distances == 3)),
        },
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def head_agreement(binary_predictions, level_predictions) -> float:
    binary = _vector(binary_predictions, "binary predictions", np.int64)
    level = _vector(level_predictions, "level predictions", np.int64)
    if binary.size == 0 or level.shape != binary.shape:
        raise ValueError("Head agreement inputs must have equal nonzero length")
    return float(np.mean(binary == (level > 0).astype(np.int64)))


def selection_f1(binary_f1: float, level_macro_f1: float) -> float:
    return float((binary_f1 + level_macro_f1) / 2.0)
