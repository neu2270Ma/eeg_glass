from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from simple_utils import (
    MODEL_PATH,
    NORMALIZATION_PATH,
    TRAIN_DATA_PATH,
    binary_metrics,
    bipolarize,
    build_model,
    fit_normalization,
    level_metrics,
    load_data,
    normalize,
    save_final_model,
    save_normalization,
)


SAMPLING_RATE = 200
SIGNAL_LENGTH = 2000
N_CHANNELS = 16
N_FFT = 200
HOP_LENGTH = 100
EMB_SIZE = 256
DEPTH = 4
HEADS = 8
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BINARY_LOSS_WEIGHT = 0.3
LEVEL_LOSS_WEIGHT = 0.7
NUM_WORKERS = 0
EPS = 1e-6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(SEED)
    raw_signals, binary_labels, level_labels = load_data(TRAIN_DATA_PATH)
    signals = bipolarize(raw_signals)
    del raw_signals
    mean, std = fit_normalization(signals)
    save_normalization(NORMALIZATION_PATH, mean, std, EPS)
    signals = normalize(signals, mean, std, EPS)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        TensorDataset(signals, binary_labels, level_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    model = build_model(
        load_pretrained=True,
        emb_size=EMB_SIZE,
        heads=HEADS,
        depth=DEPTH,
        n_channels=N_CHANNELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Train data: {TRAIN_DATA_PATH}")
    print(f"Samples: {len(binary_labels)}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sample_count = 0
        total_sum = binary_sum = level_sum = 0.0
        binary_true, binary_pred, binary_scores = [], [], []
        level_true, level_pred, level_scores = [], [], []
        for batch_signals, batch_binary, batch_level in loader:
            batch_signals = batch_signals.to(device, non_blocking=True)
            batch_binary = batch_binary.to(device, non_blocking=True)
            batch_level = batch_level.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, binary_logits, level_logits = model(batch_signals)
            binary_loss = criterion(binary_logits, batch_binary)
            level_loss = criterion(level_logits, batch_level)
            total_loss = BINARY_LOSS_WEIGHT * binary_loss + LEVEL_LOSS_WEIGHT * level_loss
            total_loss.backward()
            optimizer.step()

            count = batch_binary.shape[0]
            sample_count += count
            total_sum += float(total_loss.item()) * count
            binary_sum += float(binary_loss.item()) * count
            level_sum += float(level_loss.item()) * count
            binary_probability = torch.softmax(binary_logits.detach(), dim=1)
            level_probability = torch.softmax(level_logits.detach(), dim=1)
            binary_true.append(batch_binary.cpu().numpy())
            binary_pred.append(binary_probability.argmax(1).cpu().numpy())
            binary_scores.append(binary_probability[:, 1].cpu().numpy())
            level_true.append(batch_level.cpu().numpy())
            level_pred.append(level_probability.argmax(1).cpu().numpy())
            level_scores.append(level_probability.cpu().numpy())

        binary_result = binary_metrics(
            np.concatenate(binary_true), np.concatenate(binary_pred), np.concatenate(binary_scores)
        )
        level_result = level_metrics(
            np.concatenate(level_true), np.concatenate(level_pred), np.concatenate(level_scores)
        )
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Loss: {total_sum / sample_count:.6f}")
        print(f"Binary Loss: {binary_sum / sample_count:.6f}")
        print(f"Level Loss: {level_sum / sample_count:.6f}")
        print(f"Binary F1: {binary_result['f1']:.4f}")
        print(f"Level Macro-F1: {level_result['macro_f1']:.4f}")

    save_final_model(model, MODEL_PATH)
    print(f"Normalization saved: {NORMALIZATION_PATH}")
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
