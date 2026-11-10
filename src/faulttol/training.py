"""Training loops and checkpointing (ports of the notebook helpers)."""
from __future__ import annotations

import copy
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import (
    classification_clean_metrics,
    classification_outputs,
    regression_clean_metrics,
    regression_outputs,
)


class EarlyStopping:
    """Port of the notebook EarlyStopping."""

    def __init__(self, patience: int = 6, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.stop = False

    def step(self, score: float) -> bool:
        if self.best is None:
            self.best = score
            return False
        improved = score > self.best if self.mode == "max" else score < self.best
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return self.stop


def save_checkpoint(path: pathlib.Path, model, optimizer, epoch: int,
                    best_score: float, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: pathlib.Path, model) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int,
                shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=False)


def train_classifier(model, X_train, y_train, X_val, y_val, device, *,
                     lr: float, weight_decay: float, epochs: int, batch_size: int,
                     patience: int, ckpt_path: pathlib.Path, config: dict,
                     selection_metric: str = "roc_auc", grad_clip: float | None = None,
                     cosine: bool = False, label_smoothing: float = 0.0,
                     augment=None) -> dict:
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if cosine else None
    )
    stopper = EarlyStopping(patience=patience, mode="max")
    best_state = None

    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    loader = make_loader(X_train, y_train_t, batch_size, shuffle=True)

    history, best = [], -np.inf
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for xb, yb in loader:
            if augment is not None:
                xb = augment(xb)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        if scheduler is not None:
            scheduler.step()

        val = classification_clean_metrics(
            classification_outputs(model, X_val, y_val, device)
        )
        score = val[selection_metric]
        history.append({"epoch": epoch, "train_loss": total / max(n, 1),
                        "secs": time.time() - t0, **{"val_" + k: v for k, v in val.items()}})
        print(
            "    epoch %2d  loss %.4f  val_acc %.4f  val_f1 %.4f  val_auc %.4f  (%.0fs)"
            % (epoch, total / max(n, 1), val["accuracy"], val["f1"], val["roc_auc"],
               time.time() - t0),
            flush=True,
        )

        if score > best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(ckpt_path, model, optimizer, epoch, best, config)
        if stopper.step(score):
            print("    early stopping at epoch %d" % epoch, flush=True)
            break

    # Restore from memory, not from disk: another process running the same
    # notebook can overwrite the checkpoint file between save and reload.
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_score": float(best), "selection_metric": selection_metric}


def train_regressor(model, X_train, y_train, X_val, y_val, device, *,
                    lr: float, weight_decay: float, epochs: int, batch_size: int,
                    patience: int, ckpt_path: pathlib.Path, config: dict,
                    grad_clip: float | None = 1.0) -> dict:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopping(patience=patience, mode="min")
    best_state = None

    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    loader = make_loader(X_train, y_train_t, batch_size, shuffle=True)

    history, best = [], np.inf
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        val = regression_clean_metrics(regression_outputs(model, X_val, y_val, device))
        score = val["mae"]
        history.append({"epoch": epoch, "train_mse": total / max(n, 1),
                        "secs": time.time() - t0, **{"val_" + k: v for k, v in val.items()}})
        print(
            "    epoch %2d  train_mse %.6f  val_mae %.6f  val_rmse %.6f  (%.0fs)"
            % (epoch, total / max(n, 1), val["mae"], val["rmse"], time.time() - t0),
            flush=True,
        )

        if score < best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(ckpt_path, model, optimizer, epoch, best, config)
        if stopper.step(score):
            print("    early stopping at epoch %d" % epoch, flush=True)
            break

    # Restore from memory, not from disk (see train_classifier).
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_score": float(best), "selection_metric": "val_mae"}
